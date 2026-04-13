/**
 * lidar.js — LiDAR depth sensor support for precise window measurement
 *
 * LiDAR-equipped devices (iPhone 12 Pro+, iPad Pro) can measure real-world
 * dimensions directly using the WebXR Device API with depth sensing.
 */

/**
 * Check if the device supports LiDAR/AR depth sensing via WebXR
 * @returns {Promise<boolean>} true if immersive-ar sessions are supported
 */
export async function checkLiDARSupport() {
  try {
    if (!navigator.xr) return false;
    const supported = await navigator.xr.isSessionSupported('immersive-ar');
    return supported;
  } catch (e) {
    return false;
  }
}

/**
 * Attempt to measure window dimensions using LiDAR depth sensing.
 *
 * Full implementation would:
 * 1. Request an immersive-ar XRSession with 'depth-sensing' feature
 * 2. Create an XRWebGLLayer for the session
 * 3. Use XRFrame.getDepthInformation() to get per-pixel depth data
 * 4. Combine depth data with window bounds from segmentation to
 *    calculate real-world width/height of the detected window
 * 5. Convert depth map points at window corners to 3D coordinates
 *    using the camera intrinsics and XRView projection matrix
 * 6. Compute Euclidean distance between corner pairs for width/height
 *
 * This is a fast-follow feature — returns null for now.
 * When implemented, it will provide inch-level accuracy without any
 * server round-trip, making it the preferred measurement method on
 * supported devices.
 *
 * @returns {Promise<{ widthInches: number, heightInches: number, confidence: number } | null>}
 */
export async function measureWithLiDAR() {
  // Placeholder — full implementation is a fast-follow
  return null;
}
