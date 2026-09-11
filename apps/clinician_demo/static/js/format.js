/**
 * Formatting for clinicians: percentages, visit clock times, durations,
 * "times typical" wording and trend arrows. Pure functions, no DOM.
 */

/**
 * A probability as a percentage: 0.031 -> "3.1%", 0.45 -> "45%".
 * @param {number|null|undefined} p
 * @returns {string}
 */
export function pct(p) {
  if (p == null || Number.isNaN(p)) return '—';
  const v = p * 100;
  if (v > 0 && v < 0.1) return '<0.1%';
  return v < 10 ? `${v.toFixed(1)}%` : `${Math.round(v)}%`;
}

/**
 * A probability difference in percentage points: 0.021 -> "+2.1 pts".
 * @param {number|null|undefined} d
 * @returns {string}
 */
export function points(d) {
  if (d == null || Number.isNaN(d)) return '—';
  const v = d * 100;
  if (Math.abs(v) < 0.05) return '±0.0 pts';
  return `${v > 0 ? '+' : '−'}${Math.abs(v).toFixed(1)} pts`;
}

/**
 * Visit-relative hours as a clock: 30.5 -> "Day 2 · 06:30" (hour 0 = admission).
 * @param {number|null|undefined} h
 * @returns {string}
 */
export function clock(h) {
  if (h == null || Number.isNaN(h)) return '—';
  if (h < 0) return `${duration(-h)} before admission`;
  let day = Math.floor(h / 24);
  let minutes = Math.round((h - day * 24) * 60);
  if (minutes >= 1440) {
    minutes -= 1440;
    day += 1;
  }
  const hh = String(Math.floor(minutes / 60)).padStart(2, '0');
  const mm = String(minutes % 60).padStart(2, '0');
  return `Day ${day + 1} · ${hh}:${mm}`;
}

/**
 * A duration in hours as words: 0.5 -> "30 min", 14 -> "14 h", 60 -> "2.5 days".
 * @param {number|null|undefined} h
 * @returns {string}
 */
export function duration(h) {
  if (h == null || Number.isNaN(h)) return '—';
  if (h < 1) return `${Math.round(h * 60)} min`;
  if (h < 48) return `${Math.round(h)} h`;
  return `${(h / 24).toFixed(1)} days`;
}

/**
 * How a risk compares with the typical at-risk patient: "3.4× typical".
 * @param {number|null|undefined} p
 * @param {number|null|undefined} base
 * @returns {string|null}
 */
export function timesTypical(p, base) {
  if (p == null || !base) return null;
  const ratio = p / base;
  if (ratio < 0.1) return 'far below typical';
  if (ratio < 0.95) return `${ratio.toFixed(1)}× typical (lower)`;
  if (ratio <= 1.05) return 'about typical';
  return ratio >= 10 ? `${Math.round(ratio)}× typical` : `${ratio.toFixed(1)}× typical`;
}

/**
 * Direction of change between two risks, with a short label.
 * @param {number|null|undefined} now
 * @param {number|null|undefined} before
 * @param {number} hours window the change is over
 * @returns {{arrow: string, label: string, dir: -1|0|1}}
 */
export function trend(now, before, hours) {
  if (now == null || before == null) return { arrow: '', label: '', dir: 0 };
  const d = now - before;
  const rel = before > 0 ? Math.abs(d) / before : Math.abs(d) > 0 ? Infinity : 0;
  if (Math.abs(d) < 0.002 || rel < 0.1) return { arrow: '→', label: `steady over ${hours} h`, dir: 0 };
  return d > 0
    ? { arrow: '↑', label: `${points(d)} in ${hours} h`, dir: 1 }
    : { arrow: '↓', label: `${points(d)} in ${hours} h`, dir: -1 };
}

/**
 * An AUROC to three decimals.
 * @param {number|null|undefined} x
 * @returns {string}
 */
export function auroc(x) {
  return x == null ? '—' : x.toFixed(3);
}

/**
 * A 95% interval "0.867–0.874".
 * @param {number[]|null|undefined} pair
 * @returns {string}
 */
export function interval(pair) {
  return pair ? `${pair[0].toFixed(3)}–${pair[1].toFixed(3)}` : '';
}

/**
 * Index of the last time at or before t (0 if t precedes them all).
 * @param {number[]} times ascending
 * @param {number} t
 * @returns {number}
 */
export function indexAtOrBefore(times, t) {
  let lo = 0;
  let hi = times.length - 1;
  if (hi < 0 || t <= times[0]) return 0;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (times[mid] <= t) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/**
 * Index of the time closest to t.
 * @param {number[]} times ascending
 * @param {number} t
 * @returns {number}
 */
export function nearestIndex(times, t) {
  const i = indexAtOrBefore(times, t);
  if (i + 1 < times.length && Math.abs(times[i + 1] - t) < Math.abs(times[i] - t)) return i + 1;
  return i;
}

/**
 * Clamp a number into [lo, hi].
 * @param {number} x
 * @param {number} lo
 * @param {number} hi
 * @returns {number}
 */
export function clamp(x, lo, hi) {
  return Math.min(hi, Math.max(lo, x));
}
