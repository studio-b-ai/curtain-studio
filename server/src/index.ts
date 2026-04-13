/**
 * Measure API — Fastify server entry point.
 * Receives window photos + EXIF + bounds, returns estimated dimensions.
 */

import Fastify from 'fastify';
import cors from '@fastify/cors';
import { loadModel, estimateDimensions, categorizeSize } from './estimate.js';
import { storeFeedback } from './feedback.js';
import type { EstimationInput, WindowBounds, ExifData } from './estimate.js';
import type { FeedbackData } from './feedback.js';

const PORT = parseInt(process.env.PORT ?? '3000', 10);
const HOST = '0.0.0.0';
const CORS_ORIGIN = process.env.CORS_ORIGIN ?? 'https://measure.asthetik.com';
const MODEL_PATH = process.env.MODEL_PATH ?? 'models/estimation_mlp.onnx';

const app = Fastify({ logger: true });

await app.register(cors, { origin: CORS_ORIGIN });

// --- Model loading ---

let modelLoaded = false;

try {
  await loadModel(MODEL_PATH);
  modelLoaded = true;
  app.log.info(`ONNX model loaded from ${MODEL_PATH}`);
} catch (err) {
  app.log.warn(`Failed to load ONNX model from ${MODEL_PATH} — estimation endpoint will fail gracefully`);
  app.log.warn(err);
}

// --- Routes ---

/** Health check. */
app.get('/health', async () => {
  return { status: 'ok', model: modelLoaded };
});

/** Estimate window dimensions from image + bounds + EXIF. */
interface EstimateBody {
  imageBase64: string;
  windowBounds: WindowBounds;
  exif: ExifData;
  sizes?: number[];
}

app.post<{ Body: EstimateBody }>('/estimate', async (request, reply) => {
  if (!modelLoaded) {
    return reply.status(503).send({
      error: 'Model not loaded',
      message: 'The estimation model is not available. Please try again later.',
    });
  }

  const { imageBase64, windowBounds, exif, sizes } = request.body;

  if (!imageBase64 || !windowBounds || !exif) {
    return reply.status(400).send({
      error: 'Missing required fields',
      message: 'imageBase64, windowBounds, and exif are required.',
    });
  }

  const input: EstimationInput = { imageBase64, windowBounds, exif };

  try {
    const result = await estimateDimensions(input);

    if (sizes && sizes.length > 0) {
      result.sizeCategory = categorizeSize(result.widthInches, sizes);
    }

    return result;
  } catch (err) {
    request.log.error(err, 'Estimation failed');
    return reply.status(500).send({
      error: 'Estimation failed',
      message: err instanceof Error ? err.message : 'Unknown error',
    });
  }
});

/** Store confirmed measurement feedback to GCS. */
app.post<{ Body: FeedbackData }>('/feedback', async (request, reply) => {
  const data = request.body;

  if (!data.sessionId || !data.confirmedDimensions) {
    return reply.status(400).send({
      error: 'Missing required fields',
      message: 'sessionId and confirmedDimensions are required.',
    });
  }

  try {
    await storeFeedback(data);
    return { stored: true };
  } catch (err) {
    request.log.error(err, 'Feedback storage failed');
    return reply.status(500).send({
      error: 'Storage failed',
      message: err instanceof Error ? err.message : 'Unknown error',
    });
  }
});

// --- Start ---

await app.listen({ port: PORT, host: HOST });
app.log.info(`Measure API listening on ${HOST}:${PORT}`);
