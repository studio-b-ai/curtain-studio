/**
 * segmentation.js — Window detection via classical CV pipeline and optional YOLO model
 *
 * The classical pipeline uses a 7-factor scoring system with integral images,
 * directional Sobel edges, coarse+fine search, and sill detection.
 * The YOLO path provides a faster, higher-accuracy alternative when a model is loaded.
 */

let yoloModel = null;

/**
 * Load a YOLO model from a TensorFlow.js model URL
 * @param {string} modelUrl - URL to the tfjs model.json
 * @returns {Promise<boolean>} true if model loaded successfully
 */
export async function loadYOLOModel(modelUrl) {
  try {
    if (typeof tf === 'undefined') {
      console.warn('[Segmentation] TensorFlow.js not available, skipping YOLO model load');
      return false;
    }
    yoloModel = await tf.loadGraphModel(modelUrl);
    console.log('[Segmentation] YOLO model loaded from', modelUrl);
    return true;
  } catch (e) {
    console.warn('[Segmentation] Failed to load YOLO model:', e.message);
    yoloModel = null;
    return false;
  }
}

/**
 * Detect window bounds in a photo. Uses YOLO if available, falls back to classical.
 * @param {string} photoDataUrl - JPEG data URL of the photo
 * @returns {Promise<{ x: number, y: number, w: number, h: number, confidence: number, polygon?: number[][] } | null>}
 */
export async function detectWindow(photoDataUrl) {
  if (yoloModel) {
    try {
      const result = await detectWindowYOLO(photoDataUrl);
      if (result && result.confidence > 0.3) return result;
    } catch (e) {
      console.warn('[Segmentation] YOLO detection failed, falling back to classical:', e.message);
    }
  }
  return detectWindowBoundsClassical(photoDataUrl);
}

/**
 * YOLO-based window detection
 * @private
 */
async function detectWindowYOLO(photoDataUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = async () => {
      try {
        // Resize to 640x640 for YOLO input
        const inputSize = 640;
        const canvas = document.createElement('canvas');
        canvas.width = inputSize;
        canvas.height = inputSize;
        const ctx = canvas.getContext('2d');

        // Maintain aspect ratio with letterboxing
        const scale = Math.min(inputSize / img.width, inputSize / img.height);
        const sw = img.width * scale;
        const sh = img.height * scale;
        const offsetX = (inputSize - sw) / 2;
        const offsetY = (inputSize - sh) / 2;

        ctx.fillStyle = '#808080';
        ctx.fillRect(0, 0, inputSize, inputSize);
        ctx.drawImage(img, offsetX, offsetY, sw, sh);

        // Create tensor: [1, 640, 640, 3], normalized to 0-1
        const tensor = tf.browser.fromPixels(canvas)
          .toFloat()
          .div(255.0)
          .expandDims(0);

        // Run inference
        const predictions = await yoloModel.predict(tensor);
        tensor.dispose();

        // Parse YOLO output — format: [1, numDetections, 6] (x, y, w, h, confidence, classId)
        // Window class is typically classId 0 in a window-detection model
        const data = await predictions.data();
        predictions.dispose();

        // Find best window detection
        let bestConf = 0;
        let bestBox = null;
        const stride = 6; // x, y, w, h, conf, classId
        const numDetections = data.length / stride;

        for (let i = 0; i < numDetections; i++) {
          const offset = i * stride;
          const conf = data[offset + 4];
          const classId = data[offset + 5];

          // Accept window class (0) with confidence > 0.25
          if (classId === 0 && conf > 0.25 && conf > bestConf) {
            bestConf = conf;
            // Convert from 640x640 coords back to normalized 0-1
            const cx = data[offset];
            const cy = data[offset + 1];
            const bw = data[offset + 2];
            const bh = data[offset + 3];

            // Remove letterbox offset and scale
            bestBox = {
              x: Math.max(0, (cx - bw / 2 - offsetX) / sw),
              y: Math.max(0, (cy - bh / 2 - offsetY) / sh),
              w: Math.min(1, bw / sw),
              h: Math.min(1, bh / sh),
              confidence: conf,
            };
          }
        }

        resolve(bestBox);
      } catch (e) {
        reject(e);
      }
    };
    img.onerror = () => reject(new Error('Failed to load image for YOLO detection'));
    img.src = photoDataUrl;
  });
}

// ============================================================
// Classical window detection pipeline
// Moved from index.html — this is the entire existing pipeline
// with the 7-factor scoring, integral images, Sobel edges,
// coarse+fine passes, and sill detection.
// ============================================================

/**
 * Classical CV window detection using brightness, blue ratio, edge scoring,
 * uniformity, aspect ratio, position, and size heuristics.
 * @param {string} photoDataUrl - JPEG data URL
 * @returns {Promise<{ x: number, y: number, w: number, h: number, confidence: number } | null>}
 */
export function detectWindowBoundsClassical(photoDataUrl) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      // 1. Downscale to ~200px wide for speed
      const maxDim = 200;
      const scale = maxDim / Math.max(img.width, img.height);
      const sw = Math.round(img.width * scale);
      const sh = Math.round(img.height * scale);
      const off = document.createElement('canvas');
      off.width = sw; off.height = sh;
      const octx = off.getContext('2d');
      octx.drawImage(img, 0, 0, sw, sh);
      const pixels = octx.getImageData(0, 0, sw, sh).data;

      // 2. Extract channels: luminance + blue ratio (sky detection)
      const lum = new Float32Array(sw * sh);
      const blueRatio = new Float32Array(sw * sh);
      for (let i = 0; i < sw * sh; i++) {
        const r = pixels[i * 4], g = pixels[i * 4 + 1], b = pixels[i * 4 + 2];
        lum[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        const total = r + g + b;
        blueRatio[i] = total > 30 ? (b / total) * 255 : 0;
      }

      // 3. Build integral images: brightness, blue channel, directional edges
      const intImg = buildIntegralImage(lum, sw, sh);
      const intBlue = buildIntegralImage(blueRatio, sw, sh);
      const { horiz: edgesH, vert: edgesV, combined: edgesC } = computeDirectionalEdges(lum, sw, sh);
      const intEdge = buildIntegralImage(edgesC, sw, sh);
      const intEdgeH = buildIntegralImage(edgesH, sw, sh);
      const intEdgeV = buildIntegralImage(edgesV, sw, sh);

      // 4. Total image stats for contrast scoring
      const totalPixels = sw * sh;
      const totalBrightness = queryRect(intImg, 0, 0, sw - 1, sh - 1, sw);
      const totalBlue = queryRect(intBlue, 0, 0, sw - 1, sh - 1, sw);

      // 5. PASS 1: Coarse scan (5% grid) — find top 5 candidates
      const coarseStepX = Math.max(2, Math.round(sw * 0.05));
      const coarseStepY = Math.max(2, Math.round(sh * 0.05));
      const topN = [];

      function scoreCandidate(x, y, rw, rh) {
        const x2 = x + rw - 1, y2 = y + rh - 1;
        if (x2 >= sw || y2 >= sh) return -1;

        const area = rw * rh;
        const insideBright = queryRect(intImg, x, y, x2, y2, sw);
        const outsideBright = totalBrightness - insideBright;
        const outsideArea = totalPixels - area;
        const avgIn = insideBright / area;
        const avgOut = outsideArea > 0 ? outsideBright / outsideArea : 128;
        const contrast = avgOut > 0 ? avgIn / avgOut : 0;
        const brightnessScore = Math.min(contrast / 2.0, 1.0);

        const insideBlue = queryRect(intBlue, x, y, x2, y2, sw);
        const outsideBlue = totalBlue - insideBlue;
        const avgBlueIn = insideBlue / area;
        const avgBlueOut = outsideArea > 0 ? outsideBlue / outsideArea : 85;
        const blueContrast = avgBlueOut > 0 ? avgBlueIn / avgBlueOut : 1;
        const blueScore = Math.min(Math.max(blueContrast - 0.9, 0) / 0.5, 1.0);

        const edgeBand = Math.max(2, Math.round(Math.min(rw, rh) * 0.08));
        let dirEdgeScore = 0;

        if (y + edgeBand - 1 < sh) {
          const topHoriz = queryRect(intEdgeH, x, y, x2, Math.min(y + edgeBand - 1, y2), sw);
          dirEdgeScore += topHoriz / (rw * edgeBand);
        }
        if (y2 - edgeBand + 1 >= y) {
          const botHoriz = queryRect(intEdgeH, x, Math.max(y2 - edgeBand + 1, y), x2, y2, sw);
          dirEdgeScore += botHoriz / (rw * edgeBand);
        }
        if (x + edgeBand - 1 <= x2) {
          const leftVert = queryRect(intEdgeV, x, y, Math.min(x + edgeBand - 1, x2), y2, sw);
          dirEdgeScore += leftVert / (edgeBand * rh);
        }
        if (x2 - edgeBand + 1 >= x) {
          const rightVert = queryRect(intEdgeV, Math.max(x2 - edgeBand + 1, x), y, x2, y2, sw);
          dirEdgeScore += rightVert / (edgeBand * rh);
        }
        const edgeScore = Math.min(dirEdgeScore / 200, 1.0);

        const meanIn = avgIn;
        let varianceApprox = 0;
        const sampleStep = Math.max(1, Math.floor(Math.min(rw, rh) / 5));
        let sampleCount = 0;
        for (let sy = y; sy <= y2; sy += sampleStep) {
          for (let sx = x; sx <= x2; sx += sampleStep) {
            const v = lum[sy * sw + sx] - meanIn;
            varianceApprox += v * v;
            sampleCount++;
          }
        }
        varianceApprox = sampleCount > 0 ? Math.sqrt(varianceApprox / sampleCount) : 50;
        const uniformScore = Math.max(0, 1.0 - varianceApprox / 80);

        const ar = rw / rh;
        let aspectScore;
        if (ar >= 0.6 && ar <= 2.5) aspectScore = 1.0;
        else if (ar >= 0.4 && ar <= 3.5) aspectScore = 0.5;
        else aspectScore = 0.1;

        const cx = (x + rw / 2) / sw;
        const cy = (y + rh / 2) / sh;
        const posScore = Math.exp(-2 * ((cx - 0.5) ** 2 + (cy - 0.38) ** 2));

        const areaFrac = area / totalPixels;
        const sizeScore = areaFrac >= 0.05 && areaFrac <= 0.6 ? 1.0 :
                          areaFrac >= 0.02 ? 0.5 : 0.2;

        // Weighted total — 7 factors
        return brightnessScore * 0.28
             + blueScore * 0.12
             + edgeScore * 0.20
             + uniformScore * 0.07
             + aspectScore * 0.10
             + posScore * 0.10
             + sizeScore * 0.13;
      }

      // Coarse scan
      for (let y = 0; y < sh * 0.65; y += coarseStepY) {
        for (let x = 0; x < sw * 0.75; x += coarseStepX) {
          for (let rw = Math.round(sw * 0.15); rw <= Math.min(sw * 0.85, sw - x); rw += coarseStepX) {
            for (let rh = Math.round(sh * 0.15); rh <= Math.min(sh * 0.7, sh - y); rh += coarseStepY) {
              const score = scoreCandidate(x, y, rw, rh);
              if (score < 0) continue;
              if (topN.length < 5 || score > topN[topN.length - 1].score) {
                topN.push({ x, y, rw, rh, score });
                topN.sort((a, b) => b.score - a.score);
                if (topN.length > 5) topN.length = 5;
              }
            }
          }
        }
      }

      // 6. PASS 2: Fine refinement around each top candidate
      const fineStepX = Math.max(1, Math.round(sw * 0.015));
      const fineStepY = Math.max(1, Math.round(sh * 0.015));
      let best = null;
      let bestScore = -1;

      for (const cand of topN) {
        const padXY = Math.round(sw * 0.08);
        const padSize = Math.round(sw * 0.12);
        const yMin = Math.max(0, cand.y - padXY);
        const yMax = Math.min(Math.round(sh * 0.65), cand.y + padXY);
        const xMin = Math.max(0, cand.x - padXY);
        const xMax = Math.min(Math.round(sw * 0.75), cand.x + padXY);
        const rwMin = Math.max(Math.round(sw * 0.15), cand.rw - padSize);
        const rwMax = Math.min(sw, cand.rw + padSize);
        const rhMin = Math.max(Math.round(sh * 0.15), cand.rh - padSize);
        const rhMax = Math.min(sh, cand.rh + Math.round(sh * 0.25));

        for (let y = yMin; y <= yMax; y += fineStepY) {
          for (let x = xMin; x <= xMax; x += fineStepX) {
            for (let rw = rwMin; rw <= Math.min(rwMax, sw - x); rw += fineStepX) {
              for (let rh = rhMin; rh <= Math.min(rhMax, sh - y); rh += fineStepY) {
                const score = scoreCandidate(x, y, rw, rh);
                if (score > bestScore) {
                  bestScore = score;
                  best = { x: x / sw, y: y / sh, w: rw / sw, h: rh / sh, confidence: score };
                }
              }
            }
          }
        }
      }

      // 7. PASS 3: Sill detection — extend bottom edge to nearest strong horizontal edge
      if (best) {
        const bx = Math.round(best.x * sw), by = Math.round(best.y * sh);
        const bw = Math.round(best.w * sw), bh = Math.round(best.h * sh);
        const bot = by + bh;
        let bestSillY = bot;
        let bestSillStrength = 0;
        for (let sy = bot; sy < Math.min(bot + Math.round(sh * 0.3), sh - 1); sy++) {
          const edgeStr = queryRect(intEdgeH, bx, sy, Math.min(bx + bw - 1, sw - 1), Math.min(sy + 2, sh - 1), sw);
          const avgStr = edgeStr / (bw * 3);
          if (avgStr > bestSillStrength && avgStr > 15) {
            bestSillStrength = avgStr;
            bestSillY = sy;
          }
        }
        if (bestSillY > bot + 2) {
          const extH = bestSillY - by;
          const extScore = scoreCandidate(bx, by, bw, extH);
          if (extScore > bestScore * 0.85) {
            best = { x: bx / sw, y: by / sh, w: bw / sw, h: extH / sh, confidence: Math.max(extScore, bestScore) };
            bestScore = Math.max(extScore, bestScore);
          }
        }
      }

      resolve(best);
    };
    img.onerror = () => resolve(null);
    img.src = photoDataUrl;
  });
}

/**
 * Build a summed-area table (integral image) from a flat 2D array
 * @param {Float32Array} data - Input pixel data
 * @param {number} w - Width
 * @param {number} h - Height
 * @returns {Float64Array}
 */
export function buildIntegralImage(data, w, h) {
  const int = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    let rowSum = 0;
    for (let x = 0; x < w; x++) {
      rowSum += data[y * w + x];
      int[y * w + x] = rowSum + (y > 0 ? int[(y - 1) * w + x] : 0);
    }
  }
  return int;
}

/**
 * Query the sum of a rectangular region from an integral image
 * @param {Float64Array} int - Integral image
 * @param {number} x1 - Left
 * @param {number} y1 - Top
 * @param {number} x2 - Right
 * @param {number} y2 - Bottom
 * @param {number} w - Image width
 * @returns {number}
 */
export function queryRect(int, x1, y1, x2, y2, w) {
  const d = int[y2 * w + x2];
  const a = (x1 > 0 && y1 > 0) ? int[(y1 - 1) * w + (x1 - 1)] : 0;
  const b = (y1 > 0) ? int[(y1 - 1) * w + x2] : 0;
  const c = (x1 > 0) ? int[y2 * w + (x1 - 1)] : 0;
  return d - b - c + a;
}

/**
 * Compute directional Sobel edge responses
 * Horizontal edges (gy) detect window frame top/bottom + sill
 * Vertical edges (gx) detect window frame sides
 * @param {Float32Array} lum - Luminance data
 * @param {number} w - Width
 * @param {number} h - Height
 * @returns {{ horiz: Float32Array, vert: Float32Array, combined: Float32Array }}
 */
export function computeDirectionalEdges(lum, w, h) {
  const horiz = new Float32Array(w * h);
  const vert = new Float32Array(w * h);
  const combined = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const gx = -lum[(y - 1) * w + (x - 1)] + lum[(y - 1) * w + (x + 1)]
               - 2 * lum[y * w + (x - 1)] + 2 * lum[y * w + (x + 1)]
               - lum[(y + 1) * w + (x - 1)] + lum[(y + 1) * w + (x + 1)];
      const gy = -lum[(y - 1) * w + (x - 1)] - 2 * lum[(y - 1) * w + x] - lum[(y - 1) * w + (x + 1)]
               + lum[(y + 1) * w + (x - 1)] + 2 * lum[(y + 1) * w + x] + lum[(y + 1) * w + (x + 1)];
      vert[y * w + x] = Math.abs(gx);
      horiz[y * w + x] = Math.abs(gy);
      combined[y * w + x] = Math.sqrt(gx * gx + gy * gy);
    }
  }
  return { horiz, vert, combined };
}
