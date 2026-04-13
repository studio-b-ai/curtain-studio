/**
 * camera.js — Camera capture, flip, retake, file upload, and EXIF extraction
 */

/**
 * Initialize camera with environment-facing preference
 * @param {object} state - Global app state (S)
 * @returns {Promise<MediaStream|null>}
 */
export async function initCamera(state) {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: state.cameraFacing,
        width: { ideal: 1080 },
        height: { ideal: 1440 },
      },
    });
    state.stream = stream;
    document.getElementById('cameraFeed').srcObject = stream;
    return stream;
  } catch (e) {
    document.getElementById('cameraSection').classList.add('hidden');
    document.getElementById('noCamera').classList.remove('hidden');
    return null;
  }
}

/**
 * Toggle between front and rear cameras
 * @param {object} state - Global app state
 */
export async function flipCamera(state) {
  state.cameraFacing = state.cameraFacing === 'environment' ? 'user' : 'environment';
  if (state.stream) state.stream.getTracks().forEach(t => t.stop());
  await initCamera(state);
}

/**
 * Capture current video frame as JPEG data URL
 * @param {object} state - Global app state
 * @returns {string} JPEG data URL
 */
export function capturePhoto(state) {
  const v = document.getElementById('cameraFeed');
  const c = document.createElement('canvas');
  c.width = v.videoWidth;
  c.height = v.videoHeight;
  c.getContext('2d').drawImage(v, 0, 0);
  const url = c.toDataURL('image/jpeg', 0.85);
  state.photo = url;

  const img = document.getElementById('capturedPhoto');
  img.src = url;
  img.style.display = 'block';
  document.getElementById('cameraFeed').style.display = 'none';
  document.getElementById('cameraGuideOverlay').style.display = 'none';
  document.getElementById('shutterBtn').classList.add('hidden');
  document.getElementById('flipCameraBtn').classList.add('hidden');
  document.getElementById('retakeBtn').classList.remove('hidden');
  document.getElementById('nextBtn').textContent = 'Continue';

  return url;
}

/**
 * Clear captured photo, reset UI to live camera
 * @param {object} state - Global app state
 */
export function retakePhoto(state) {
  state.photo = null;
  state.windowBounds = null;
  state.windowDetectionConfidence = 0;
  document.getElementById('capturedPhoto').style.display = 'none';
  document.getElementById('cameraFeed').style.display = 'block';
  document.getElementById('cameraGuideOverlay').style.display = 'flex';
  document.getElementById('shutterBtn').classList.remove('hidden');
  document.getElementById('flipCameraBtn').classList.remove('hidden');
  document.getElementById('retakeBtn').classList.add('hidden');
  document.getElementById('nextBtn').textContent = 'Skip Photo';
}

/**
 * Handle file input selection (upload fallback)
 * @param {Event} event - File input change event
 * @param {object} state - Global app state
 * @returns {Promise<string|null>} Data URL or null
 */
export function handleFileSelect(event, state) {
  return new Promise((resolve) => {
    const f = event.target.files[0];
    if (!f) { resolve(null); return; }
    const r = new FileReader();
    r.onload = ev => {
      state.photo = ev.target.result;
      resolve(ev.target.result);
    };
    r.readAsDataURL(f);
  });
}

/**
 * Extract EXIF metadata from a JPEG data URL.
 * Parses the APP1 marker to find IFD entries for focal length, phone model, and dimensions.
 * @param {string} dataUrl - JPEG data URL
 * @returns {{ focalLength: number, phoneModel: string|null, imageWidth: number, imageHeight: number }}
 */
export function extractExif(dataUrl) {
  const result = { focalLength: 26, phoneModel: null, imageWidth: 0, imageHeight: 0 };

  try {
    // Decode base64 to binary
    const base64 = dataUrl.split(',')[1];
    if (!base64) return result;
    const binary = atob(base64);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);

    // Get image dimensions via an offscreen decode
    const img = new Image();
    img.src = dataUrl;
    if (img.naturalWidth) {
      result.imageWidth = img.naturalWidth;
      result.imageHeight = img.naturalHeight;
    }

    // Verify JPEG SOI marker
    if (bytes[0] !== 0xFF || bytes[1] !== 0xD8) return result;

    // Find APP1 marker (0xFFE1) containing EXIF
    let offset = 2;
    while (offset < len - 4) {
      if (bytes[offset] !== 0xFF) break;
      const marker = bytes[offset + 1];
      const segLen = (bytes[offset + 2] << 8) | bytes[offset + 3];

      if (marker === 0xE1) {
        // Check for "Exif\0\0" header
        if (bytes[offset + 4] === 0x45 && bytes[offset + 5] === 0x78 &&
            bytes[offset + 6] === 0x69 && bytes[offset + 7] === 0x66 &&
            bytes[offset + 8] === 0x00 && bytes[offset + 9] === 0x00) {
          parseExifIFD(bytes, offset + 10, result);
        }
        break;
      }

      offset += 2 + segLen;
    }
  } catch (e) {
    // EXIF parsing is best-effort; return defaults on any error
  }

  return result;
}

/**
 * Parse TIFF IFD entries from EXIF data
 * @private
 */
function parseExifIFD(bytes, tiffStart, result) {
  const len = bytes.length;
  if (tiffStart + 8 > len) return;

  // Determine byte order
  const bigEndian = bytes[tiffStart] === 0x4D && bytes[tiffStart + 1] === 0x4D;

  function read16(off) {
    if (off + 1 >= len) return 0;
    return bigEndian
      ? (bytes[off] << 8) | bytes[off + 1]
      : bytes[off] | (bytes[off + 1] << 8);
  }

  function read32(off) {
    if (off + 3 >= len) return 0;
    return bigEndian
      ? (bytes[off] << 24) | (bytes[off + 1] << 16) | (bytes[off + 2] << 8) | bytes[off + 3]
      : bytes[off] | (bytes[off + 1] << 8) | (bytes[off + 2] << 16) | (bytes[off + 3] << 24);
  }

  function readString(off, count) {
    let s = '';
    for (let i = 0; i < count && off + i < len; i++) {
      const c = bytes[off + i];
      if (c === 0) break;
      s += String.fromCharCode(c);
    }
    return s.trim();
  }

  function readRational(off) {
    const num = read32(off);
    const den = read32(off + 4);
    return den > 0 ? num / den : 0;
  }

  // IFD0 offset
  const ifd0Off = tiffStart + read32(tiffStart + 4);
  if (ifd0Off + 2 > len) return;
  const numEntries = read16(ifd0Off);
  let exifIFDOffset = 0;

  for (let i = 0; i < numEntries; i++) {
    const entryOff = ifd0Off + 2 + i * 12;
    if (entryOff + 12 > len) break;
    const tag = read16(entryOff);
    const type = read16(entryOff + 2);
    const count = read32(entryOff + 4);
    const valueOff = entryOff + 8;

    // 0x010F = Make, 0x0110 = Model
    if (tag === 0x0110) {
      // Model: ASCII string
      const strOff = count > 4 ? tiffStart + read32(valueOff) : valueOff;
      result.phoneModel = readString(strOff, count);
    }

    // 0x8769 = ExifIFD pointer
    if (tag === 0x8769) {
      exifIFDOffset = tiffStart + read32(valueOff);
    }
  }

  // Parse ExifIFD for focal length tags
  if (exifIFDOffset && exifIFDOffset + 2 < len) {
    const exifEntries = read16(exifIFDOffset);
    for (let i = 0; i < exifEntries; i++) {
      const entryOff = exifIFDOffset + 2 + i * 12;
      if (entryOff + 12 > len) break;
      const tag = read16(entryOff);
      const type = read16(entryOff + 2);
      const count = read32(entryOff + 4);
      const valueOff = entryOff + 8;

      // 0x920A = FocalLength (RATIONAL)
      if (tag === 0x920A) {
        const ratOff = tiffStart + read32(valueOff);
        if (ratOff + 8 <= len) {
          const fl = readRational(ratOff);
          if (fl > 0) result.focalLength = fl;
        }
      }

      // 0xA405 = FocalLengthIn35mmFilm (SHORT)
      if (tag === 0xA405) {
        const fl35 = read16(valueOff);
        if (fl35 > 0) result.focalLength = fl35;
      }

      // 0xA002 = PixelXDimension, 0xA003 = PixelYDimension
      if (tag === 0xA002) {
        result.imageWidth = type === 3 ? read16(valueOff) : read32(valueOff);
      }
      if (tag === 0xA003) {
        result.imageHeight = type === 3 ? read16(valueOff) : read32(valueOff);
      }
    }
  }
}
