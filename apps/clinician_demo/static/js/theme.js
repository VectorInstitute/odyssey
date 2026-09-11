/**
 * Read theme colours from CSS custom properties, so charts drawn in SVG
 * and canvas follow the light/dark palette defined in styles.css.
 */

/**
 * The current value of a CSS custom property on :root.
 * @param {string} name e.g. "--accent"
 * @param {string} [fallback]
 * @returns {string}
 */
export function cssVar(name, fallback = '') {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

/**
 * The colour assigned to a forecast event (falls back to the accent).
 * @param {string} event
 * @returns {string}
 */
export function eventColor(event) {
  return cssVar(`--ev-${event}`, cssVar('--accent', '#0B6E77'));
}

/**
 * Parse "#RRGGBB" (or "#RGB") into [r, g, b].
 * @param {string} hex
 * @returns {number[]}
 */
export function hexToRgb(hex) {
  let h = hex.replace('#', '').trim();
  if (h.length === 3) h = [...h].map((c) => c + c).join('');
  const n = Number.parseInt(h, 16);
  if (Number.isNaN(n) || h.length !== 6) return [11, 110, 119];
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/**
 * Run fn whenever the OS light/dark preference changes.
 * @param {() => void} fn
 * @returns {() => void} unsubscribe
 */
export function onThemeChange(fn) {
  const query = window.matchMedia('(prefers-color-scheme: dark)');
  query.addEventListener('change', fn);
  return () => query.removeEventListener('change', fn);
}
