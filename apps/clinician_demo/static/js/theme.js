/**
 * Theme: light by default (the look clinicians know from the chart), dark
 * on request. Charts drawn in SVG and canvas read their colours from the
 * CSS custom properties defined in styles.css, so they follow the theme.
 */

const STORAGE_KEY = 'odyssey-demo-theme';
const EVENT = 'odyssey-themechange';

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

/** @returns {boolean} whether the dark theme is on */
export function isDark() {
  return document.documentElement.dataset.theme === 'dark';
}

function apply(dark) {
  if (dark) document.documentElement.dataset.theme = 'dark';
  else delete document.documentElement.dataset.theme;
  document.dispatchEvent(new CustomEvent(EVENT));
}

/** Restore the viewer's theme choice (only the choice is stored, never data). */
export function initTheme() {
  let stored = null;
  try {
    stored = localStorage.getItem(STORAGE_KEY);
  } catch {
    stored = null;
  }
  apply(stored === 'dark');
}

/** Switch between light and dark and remember the choice. */
export function toggleTheme() {
  const dark = !isDark();
  apply(dark);
  try {
    localStorage.setItem(STORAGE_KEY, dark ? 'dark' : 'light');
  } catch {
    /* storage may be unavailable; the theme still switches for this page */
  }
}

/**
 * Run fn whenever the theme changes.
 * @param {() => void} fn
 * @returns {() => void} unsubscribe
 */
export function onThemeChange(fn) {
  document.addEventListener(EVENT, fn);
  return () => document.removeEventListener(EVENT, fn);
}
