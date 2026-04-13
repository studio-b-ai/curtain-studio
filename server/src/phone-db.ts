/**
 * Phone model sensor size database.
 * Maps EXIF phone model strings to physical sensor dimensions
 * for photogrammetric dimension estimation.
 */

export interface SensorInfo {
  sensorWidthMm: number;
  sensorHeightMm: number;
  cropFactor: number;
}

/**
 * Typical phone sensor fallback (1/2.55" sensor).
 * Used when phone model is unknown or not in the database.
 */
export const DEFAULT_SENSOR: SensorInfo = {
  sensorWidthMm: 7.6,
  sensorHeightMm: 5.7,
  cropFactor: 4.74,
};

/**
 * Phone model -> sensor spec lookup table.
 * Main (wide) camera sensor for each model.
 * Sources: manufacturer specs, DxOMark, CIPA sensor size standards.
 */
const PHONE_DB: Record<string, SensorInfo> = {
  // --- Apple iPhone 16 series (2024) ---
  'iPhone 16 Pro Max': { sensorWidthMm: 9.8, sensorHeightMm: 7.35, cropFactor: 3.67 },
  'iPhone 16 Pro': { sensorWidthMm: 9.8, sensorHeightMm: 7.35, cropFactor: 3.67 },
  'iPhone 16 Plus': { sensorWidthMm: 7.6, sensorHeightMm: 5.7, cropFactor: 4.74 },
  'iPhone 16': { sensorWidthMm: 7.6, sensorHeightMm: 5.7, cropFactor: 4.74 },

  // --- Apple iPhone 15 series (2023) ---
  'iPhone 15 Pro Max': { sensorWidthMm: 9.8, sensorHeightMm: 7.35, cropFactor: 3.67 },
  'iPhone 15 Pro': { sensorWidthMm: 9.8, sensorHeightMm: 7.35, cropFactor: 3.67 },
  'iPhone 15 Plus': { sensorWidthMm: 7.6, sensorHeightMm: 5.7, cropFactor: 4.74 },
  'iPhone 15': { sensorWidthMm: 7.6, sensorHeightMm: 5.7, cropFactor: 4.74 },

  // --- Apple iPhone 14 series (2022) ---
  'iPhone 14 Pro Max': { sensorWidthMm: 9.8, sensorHeightMm: 7.35, cropFactor: 3.67 },
  'iPhone 14 Pro': { sensorWidthMm: 9.8, sensorHeightMm: 7.35, cropFactor: 3.67 },
  'iPhone 14 Plus': { sensorWidthMm: 7.6, sensorHeightMm: 5.7, cropFactor: 4.74 },
  'iPhone 14': { sensorWidthMm: 7.6, sensorHeightMm: 5.7, cropFactor: 4.74 },

  // --- Apple iPhone 13 series (2021) ---
  'iPhone 13 Pro Max': { sensorWidthMm: 7.6, sensorHeightMm: 5.7, cropFactor: 4.74 },
  'iPhone 13 Pro': { sensorWidthMm: 7.6, sensorHeightMm: 5.7, cropFactor: 4.74 },
  'iPhone 13': { sensorWidthMm: 7.6, sensorHeightMm: 5.7, cropFactor: 4.74 },

  // --- Apple iPhone 12 series (2020) ---
  'iPhone 12 Pro Max': { sensorWidthMm: 7.6, sensorHeightMm: 5.7, cropFactor: 4.74 },
  'iPhone 12 Pro': { sensorWidthMm: 7.6, sensorHeightMm: 5.7, cropFactor: 4.74 },
  'iPhone 12': { sensorWidthMm: 7.6, sensorHeightMm: 5.7, cropFactor: 4.74 },

  // --- Samsung Galaxy S24 series (2024) ---
  'SM-S928': { sensorWidthMm: 9.8, sensorHeightMm: 7.35, cropFactor: 3.67 }, // S24 Ultra
  'SM-S926': { sensorWidthMm: 7.6, sensorHeightMm: 5.7, cropFactor: 4.74 },  // S24+
  'SM-S921': { sensorWidthMm: 7.6, sensorHeightMm: 5.7, cropFactor: 4.74 },  // S24

  // --- Samsung Galaxy S23 series (2023) ---
  'SM-S918': { sensorWidthMm: 9.8, sensorHeightMm: 7.35, cropFactor: 3.67 }, // S23 Ultra
  'SM-S916': { sensorWidthMm: 7.6, sensorHeightMm: 5.7, cropFactor: 4.74 },  // S23+
  'SM-S911': { sensorWidthMm: 7.6, sensorHeightMm: 5.7, cropFactor: 4.74 },  // S23

  // --- Google Pixel 8 series (2023) ---
  'Pixel 8 Pro': { sensorWidthMm: 9.8, sensorHeightMm: 7.35, cropFactor: 3.67 },
  'Pixel 8': { sensorWidthMm: 7.6, sensorHeightMm: 5.7, cropFactor: 4.74 },

  // --- Google Pixel 7 series (2022) ---
  'Pixel 7 Pro': { sensorWidthMm: 9.8, sensorHeightMm: 7.35, cropFactor: 3.67 },
  'Pixel 7': { sensorWidthMm: 7.6, sensorHeightMm: 5.7, cropFactor: 4.74 },
};

/**
 * Look up sensor info for a phone model string (from EXIF).
 * Tries exact match first, then case-insensitive partial match, then falls back to default.
 */
export function lookupSensor(phoneModel: string): SensorInfo {
  // Exact match
  if (PHONE_DB[phoneModel]) {
    return PHONE_DB[phoneModel];
  }

  // Case-insensitive partial match
  const lower = phoneModel.toLowerCase();
  for (const [key, value] of Object.entries(PHONE_DB)) {
    if (lower.includes(key.toLowerCase()) || key.toLowerCase().includes(lower)) {
      return value;
    }
  }

  return DEFAULT_SENSOR;
}
