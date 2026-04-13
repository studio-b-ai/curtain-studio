/**
 * shopify.js — Shopify URL parameter handling, context display, and cart URL construction
 */

/**
 * Parse Shopify-related query parameters from the current URL
 * @returns {object|null} Shopify context object or null if not from Shopify
 */
export function parseShopifyParams() {
  const p = new URLSearchParams(location.search);
  if (!p.has('product') && !p.has('from')) return null;

  return {
    product: p.get('product') || '',
    color: p.get('color') || '',
    variant: p.get('variant') || '',
    handle: p.get('handle') || '',
    returnUrl: p.get('return_url') || '',
    from: p.get('from') || 'shopify',
    type: p.get('type') || 'standard', // 'custom' or 'standard'
    sizes: p.get('sizes') ? p.get('sizes').split(',').map(Number).filter(n => !isNaN(n)) : null,
  };
}

/**
 * Show the Shopify context badge on the splash screen and step 1
 * @param {object} shopify - Shopify context from parseShopifyParams
 * @param {string} activeColor - Current color hex value
 */
export function showShopifyContext(shopify, activeColor) {
  if (!shopify) return;
  document.getElementById('shopifyBadge').style.display = '';
  document.getElementById('shopifyContext').style.display = '';
  document.getElementById('shopifySwatch').style.background = shopify.color || activeColor;
  document.getElementById('shopifyProduct').textContent = shopify.product || 'Selected Fabric';
  document.getElementById('shopifyDetail').textContent = shopify.variant || 'From your Shopify cart';
}

/**
 * Build a Shopify cart URL with measurement line item properties
 * @param {object} shopify - Shopify context
 * @param {{ width: number, height: number, method: string, confidence: number, style: string, length: string, fullness: number, yards: number }} measurements - Measurement and style data
 * @returns {string|null} Cart URL or null if no return URL configured
 */
export function buildCartUrl(shopify, measurements) {
  if (!shopify || !shopify.returnUrl) return null;

  const params = new URLSearchParams();
  params.set('yards', measurements.yards.toFixed(1));
  params.set('window_width', String(measurements.width));
  if (measurements.height) params.set('window_height', String(measurements.height));
  params.set('measurement_method', measurements.method || 'manual');
  if (measurements.confidence != null) params.set('measurement_confidence', measurements.confidence.toFixed(2));
  params.set('style', measurements.style);
  params.set('length', measurements.length);
  params.set('fullness', String(measurements.fullness));

  const separator = shopify.returnUrl.includes('?') ? '&' : '?';
  return shopify.returnUrl + separator + params.toString();
}
