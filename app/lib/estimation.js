/**
 * estimation.js — Server API client for AI-powered window measurement estimation
 */

const API_URL = (typeof window !== 'undefined' && window.__MEASURE_API_URL)
  || 'https://measure-api-production.up.railway.app';

/**
 * Send window photo + detection bounds to the estimation API
 * @param {string} imageBase64 - Base64-encoded JPEG image (without data URL prefix)
 * @param {{ x: number, y: number, w: number, h: number }} windowBounds - Normalized window bounds (0-1)
 * @param {{ focalLength: number, phoneModel: string|null, imageWidth: number, imageHeight: number }} exif - EXIF metadata
 * @param {number[]|null} sizes - Available standard sizes (inches), or null for custom
 * @returns {Promise<{ widthInches: number, heightInches: number, confidence: number, sizeCategory?: string }>}
 */
export async function estimateDimensions(imageBase64, windowBounds, exif, sizes) {
  const body = {
    image: imageBase64,
    windowBounds,
    exif: exif || {},
    sizes: sizes || null,
  };

  const response = await fetch(`${API_URL}/estimate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errText = await response.text().catch(() => 'Unknown error');
    throw new Error(`Estimation API error (${response.status}): ${errText}`);
  }

  return response.json();
}

/**
 * Submit user feedback on measurement accuracy (non-blocking)
 * @param {{ estimatedWidth: number, estimatedHeight: number, actualWidth?: number, actualHeight?: number, confidence: number, method: string, accepted: boolean, adjusted: boolean, consentToStore: boolean, imageBase64?: string, windowBounds?: object, exif?: object }} feedbackData
 */
export function submitFeedback(feedbackData) {
  const url = `${API_URL}/feedback`;
  const body = JSON.stringify(feedbackData);

  // Use sendBeacon if available — survives page navigation
  if (navigator.sendBeacon) {
    const blob = new Blob([body], { type: 'application/json' });
    navigator.sendBeacon(url, blob);
    return;
  }

  // Fallback to fetch (may be aborted on navigation)
  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body,
  }).catch(err => console.warn('Feedback submission failed:', err));
}
