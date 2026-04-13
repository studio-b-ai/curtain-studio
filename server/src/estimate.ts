/**
 * Dimension estimation via ONNX Runtime.
 * Builds a feature vector from window bounds + EXIF + sensor specs,
 * runs inference through an MLP model, and returns estimated dimensions.
 */

import { InferenceSession, Tensor } from 'onnxruntime-node';
import { lookupSensor } from './phone-db.js';

let session: InferenceSession | null = null;

/**
 * Load ONNX model into an InferenceSession (CPU provider).
 * Called once at server startup.
 */
export async function loadModel(modelPath: string): Promise<void> {
  session = await InferenceSession.create(modelPath, {
    executionProviders: ['cpu'],
  });
}

export interface WindowBounds {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface ExifData {
  focalLength?: number;
  phoneModel?: string;
  imageWidth: number;
  imageHeight: number;
}

export interface EstimationInput {
  imageBase64: string;
  windowBounds: WindowBounds;
  exif: ExifData;
}

export interface EstimationResult {
  widthInches: number;
  heightInches: number;
  confidence: number;
  sizeCategory?: string;
}

/** Round to nearest 0.5 increment. */
function roundToHalf(value: number): number {
  return Math.round(value * 2) / 2;
}

/** Clamp a value between min and max. */
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Build a 16-element feature vector for the estimation MLP.
 *
 * Features:
 *  [0]  window pixel width
 *  [1]  window pixel height
 *  [2]  area ratio (window area / image area)
 *  [3-6]  depth corner values (zeros until Metric3D integrated)
 *  [7-10] depth edge midpoint values (zeros)
 *  [11] depth mean window (zero)
 *  [12] depth mean surround (zero)
 *  [13] focal length (mm, from EXIF or 26mm default)
 *  [14] sensor width (mm, from phone DB)
 *  [15] image width (pixels)
 */
function buildFeatureVector(bounds: WindowBounds, exif: ExifData): Float32Array {
  const sensor = lookupSensor(exif.phoneModel ?? '');
  const imageArea = exif.imageWidth * exif.imageHeight;
  const windowArea = bounds.w * bounds.h;
  const areaRatio = imageArea > 0 ? windowArea / imageArea : 0;
  const focalLength = exif.focalLength ?? 26; // typical phone wide lens

  const features = new Float32Array(16);
  features[0] = bounds.w;
  features[1] = bounds.h;
  features[2] = areaRatio;
  // features[3..6] = depth corners (zeros — Metric3D not yet integrated)
  // features[7..10] = depth edge midpoints (zeros)
  // features[11] = depth mean window (zero)
  // features[12] = depth mean surround (zero)
  features[13] = focalLength;
  features[14] = sensor.sensorWidthMm;
  features[15] = exif.imageWidth;

  return features;
}

/**
 * Check if any depth features are present (non-zero) in the feature vector.
 */
function hasDepthFeatures(features: Float32Array): boolean {
  for (let i = 3; i <= 12; i++) {
    if (features[i] !== 0) return true;
  }
  return false;
}

/**
 * Estimate window dimensions from image data + EXIF + window bounds.
 * Runs ONNX inference, rounds to nearest 0.5", clamps to plausible range.
 */
export async function estimateDimensions(input: EstimationInput): Promise<EstimationResult> {
  if (!session) {
    throw new Error('Model not loaded. Call loadModel() first.');
  }

  const features = buildFeatureVector(input.windowBounds, input.exif);
  const inputTensor = new Tensor('float32', features, [1, 16]);

  const feeds: Record<string, Tensor> = {};
  const inputName = session.inputNames[0];
  feeds[inputName] = inputTensor;

  const results = await session.run(feeds);
  const outputName = session.outputNames[0];
  const output = results[outputName];
  const data = output.data as Float32Array;

  // Model outputs [width_inches, height_inches]
  const rawWidth = data[0];
  const rawHeight = data[1];

  const widthInches = clamp(roundToHalf(rawWidth), 12, 120);
  const heightInches = clamp(roundToHalf(rawHeight), 12, 96);
  const confidence = hasDepthFeatures(features) ? 0.85 : 0.65;

  return { widthInches, heightInches, confidence };
}

/**
 * Find the closest size bracket from available standard sizes.
 * Returns a human-readable string like `48"`.
 */
export function categorizeSize(widthInches: number, availableSizes: number[]): string {
  if (availableSizes.length === 0) {
    return `${widthInches}"`;
  }

  let closest = availableSizes[0];
  let minDiff = Math.abs(widthInches - closest);

  for (const size of availableSizes) {
    const diff = Math.abs(widthInches - size);
    if (diff < minDiff) {
      minDiff = diff;
      closest = size;
    }
  }

  return `${closest}"`;
}
