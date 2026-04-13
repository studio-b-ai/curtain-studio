/**
 * Confirmed measurement storage to GCS.
 * Stores user-confirmed dimensions alongside AI estimates
 * to build training data for model improvement.
 */

import { Storage } from '@google-cloud/storage';
import type { WindowBounds, ExifData } from './estimate.js';

const BUCKET = process.env.GCS_BUCKET ?? 'aesthetik-measure-data';
const PROJECT = process.env.GCS_PROJECT ?? 'aesthetik-production-488816';

let storageClient: Storage | null = null;

function getStorage(): Storage {
  if (!storageClient) {
    storageClient = new Storage({ projectId: PROJECT });
  }
  return storageClient;
}

export interface FeedbackData {
  sessionId: string;
  imageBase64?: string;
  consentToStore: boolean;
  aiEstimate: { width: number; height: number };
  confirmedDimensions: { width: number; height: number };
  adjustedByUser: boolean;
  measurementMethod: 'ai_confirmed' | 'ai_lidar' | 'manual';
  confidence: number;
  exif: ExifData;
  windowBounds: WindowBounds;
  timestamp: string;
}

/**
 * Store feedback data to GCS.
 *
 * Always stores metadata JSON (without imageBase64) under:
 *   customer/{YYYY-MM-DD}/{sessionId}.json
 *
 * Only stores the photo if consentToStore is true and imageBase64 is present:
 *   customer/{YYYY-MM-DD}/{sessionId}.jpg
 */
export async function storeFeedback(data: FeedbackData): Promise<void> {
  const storage = getStorage();
  const bucket = storage.bucket(BUCKET);
  const date = data.timestamp.slice(0, 10); // YYYY-MM-DD
  const prefix = `customer/${date}/${data.sessionId}`;

  // Strip imageBase64 from metadata JSON
  const { imageBase64, ...metadata } = data;

  // Store metadata JSON
  const metadataFile = bucket.file(`${prefix}.json`);
  await metadataFile.save(JSON.stringify(metadata, null, 2), {
    contentType: 'application/json',
    metadata: {
      sessionId: data.sessionId,
      measurementMethod: data.measurementMethod,
      adjustedByUser: String(data.adjustedByUser),
    },
  });

  // Store photo only if user consented and image data is present
  if (data.consentToStore && imageBase64) {
    const imageBuffer = Buffer.from(imageBase64, 'base64');
    const imageFile = bucket.file(`${prefix}.jpg`);
    await imageFile.save(imageBuffer, {
      contentType: 'image/jpeg',
      metadata: {
        sessionId: data.sessionId,
      },
    });
  }
}
