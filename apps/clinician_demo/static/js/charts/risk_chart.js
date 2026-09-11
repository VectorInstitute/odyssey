/**
 * Risk-over-time chart (SVG): one line per event with its dashed alert
 * line, onset markers, care-transition ticks, an optional GBM overlay, a
 * scrub cursor, a hover tooltip and click-to-scrub. What lies after the
 * cursor is drawn faded, so the eye stays on "now" and what led to it.
 */

import { el, svg } from '../dom.js';
import { clock, pct, nearestIndex } from '../format.js';

const HEIGHT = 280;
const MARGIN = { left: 48, right: 16, top: 26, bottom: 28 };
const Y_STEPS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25];
const FUTURE_OPACITY = 0.22;
let clipCounter = 0;

function niceMax(maxValue) {
  const target = Math.min(1, Math.max(0.02, maxValue * 1.15));
  for (const step of Y_STEPS) {
    const top = Math.ceil(target / step) * step;
    if (top / step <= 5) return { top: Math.min(1, top), step };
  }
  return { top: 1, step: 0.2 };
}

function xTickStep(spanHours) {
  if (spanHours <= 36) return 6;
  if (spanHours <= 96) return 12;
  if (spanHours <= 24 * 10) return 24;
  return 48;
}

function xTickLabel(t, step) {
  if (step >= 24) return `Day ${Math.floor(t / 24) + 1}`;
  const hh = String(Math.round(t % 24)).padStart(2, '0');
  return `D${Math.floor(t / 24) + 1} ${hh}:00`;
}

function linePath(times, values, x, y) {
  let d = '';
  let open = false;
  for (let i = 0; i < times.length; i += 1) {
    const v = values[i];
    if (v == null) {
      open = false;
      continue;
    }
    d += `${open ? 'L' : 'M'}${x(times[i]).toFixed(1)},${y(v).toFixed(1)}`;
    open = true;
  }
  return d;
}

/**
 * Create the chart inside a container.
 * @param {HTMLElement} container
 * @param {{onScrub: (index: number) => void, label?: string}} opts
 * @returns {{update: (data: object) => void, setCursor: (i: number) => void,
 *   redraw: () => void, destroy: () => void}}
 */
export function createRiskChart(container, { onScrub, label = 'Risk over the admission' }) {
  const root = svg('svg', { class: 'risk-chart__svg', role: 'img', 'aria-label': label });
  const tip = el('div', { class: 'chart-tip', hidden: true });
  const wrap = el('div', { class: 'risk-chart' }, [root, tip]);
  container.append(wrap);
  const clipId = `risk-clip-${++clipCounter}`;

  let data = null;
  let cursor = 0;
  let scale = null;
  let cursorLine = null;
  let clipRect = null;

  function drawSeries(group, x, y, times, series, onsets, width) {
    // Onset labels are stacked in rows so events that begin close together
    // never print on top of each other; a label near the right edge flips
    // to the left of its line.
    const placed = [];
    for (const o of onsets) {
      const px = x(o.t);
      const text = `${o.label} began`;
      const w = text.length * 6.6 + 10;
      const flip = px + w > width - MARGIN.right;
      const span = flip ? [px - w, px] : [px, px + w];
      let row = 0;
      while (placed.some((p) => p.row === row && span[0] < p.span[1] && span[1] > p.span[0])) row += 1;
      placed.push({ row, span });
      group.append(
        svg('line', {
          x1: px, x2: px, y1: MARGIN.top, y2: HEIGHT - MARGIN.bottom,
          stroke: o.color, 'stroke-width': 2, 'stroke-dasharray': '1 3',
        }),
        svg('text', {
          class: 'onset-label',
          x: flip ? px - 4 : px + 4,
          y: MARGIN.top + 12 + row * 14,
          'text-anchor': flip ? 'end' : 'start',
          fill: o.color,
          text,
        }),
      );
    }
    for (const s of series) {
      group.append(svg('path', {
        d: linePath(times, s.values, x, y), fill: 'none', stroke: s.color,
        'stroke-width': 2.2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round',
      }));
    }
  }

  function draw() {
    root.replaceChildren();
    if (!data || !data.times.length) return;
    const width = Math.max(320, wrap.clientWidth || 800);
    root.setAttribute('viewBox', `0 0 ${width} ${HEIGHT}`);
    root.setAttribute('width', String(width));
    root.setAttribute('height', String(HEIGHT));
    const { times, series } = data;
    const t0 = times[0];
    const t1 = times[times.length - 1] > t0 ? times[times.length - 1] : t0 + 1;
    const plotW = width - MARGIN.left - MARGIN.right;
    const plotH = HEIGHT - MARGIN.top - MARGIN.bottom;
    let maxValue = 0;
    for (const s of series) {
      for (const v of s.values) if (v != null && v > maxValue) maxValue = v;
      if (s.threshold != null && s.threshold > maxValue) maxValue = s.threshold;
    }
    const { top, step } = niceMax(maxValue);
    const x = (t) => MARGIN.left + ((t - t0) / (t1 - t0)) * plotW;
    const y = (v) => MARGIN.top + plotH - (Math.min(v, top) / top) * plotH;
    scale = { x, t0, t1, plotW };

    clipRect = svg('rect', { x: MARGIN.left, y: 0, width: 0, height: HEIGHT });
    root.append(svg('defs', {}, [svg('clipPath', { id: clipId }, [clipRect])]));

    const grid = svg('g', { class: 'grid' });
    const axis = svg('g', { class: 'axis' });
    for (let v = 0; v <= top + 1e-9; v += step) {
      grid.append(svg('line', { x1: MARGIN.left, x2: width - MARGIN.right, y1: y(v), y2: y(v) }));
      axis.append(svg('text', { x: MARGIN.left - 8, y: y(v) + 4, 'text-anchor': 'end', text: pct(v) }));
    }
    const xStep = xTickStep(t1 - t0);
    for (let t = Math.ceil(t0 / xStep) * xStep; t <= t1; t += xStep) {
      grid.append(svg('line', { x1: x(t), x2: x(t), y1: MARGIN.top, y2: MARGIN.top + plotH }));
      const nearRight = x(t) > width - MARGIN.right - 30;
      axis.append(svg('text', {
        x: nearRight ? width - MARGIN.right : x(t),
        y: HEIGHT - 8,
        'text-anchor': nearRight ? 'end' : 'middle',
        text: xTickLabel(t, xStep),
      }));
    }
    root.append(grid, axis);

    const markers = svg('g', { class: 'marker' });
    for (const m of data.markers ?? []) {
      if (m.t < t0 || m.t > t1) continue;
      markers.append(
        svg('line', { x1: x(m.t), x2: x(m.t), y1: MARGIN.top, y2: MARGIN.top + plotH }, [
          svg('title', { text: `${m.label} · ${clock(m.t)}` }),
        ]),
      );
    }
    root.append(markers);

    for (const s of series) {
      if (s.threshold == null) continue;
      root.append(
        svg('line', {
          x1: MARGIN.left, x2: width - MARGIN.right, y1: y(s.threshold), y2: y(s.threshold),
          stroke: s.color, 'stroke-width': 1, 'stroke-dasharray': '5 4', opacity: 0.5,
        }, [svg('title', { text: `${s.label}: alert line at ${pct(s.threshold)}` })]),
      );
    }

    if (data.overlay && data.overlay.points.length) {
      const pts = data.overlay.points.filter((p) => p.v != null && p.t >= t0 && p.t <= t1);
      const g = svg('g', { opacity: 0.85 });
      g.append(svg('path', {
        d: linePath(pts.map((p) => p.t), pts.map((p) => p.v), x, y),
        fill: 'none', stroke: data.overlay.color, 'stroke-width': 1.2, 'stroke-dasharray': '3 3',
      }));
      for (const p of pts) g.append(svg('circle', { cx: x(p.t), cy: y(p.v), r: 2.2, fill: data.overlay.color }));
      g.append(svg('title', { text: data.overlay.label }));
      root.append(g);
    }

    const onsets = (data.onsets ?? [])
      .filter((o) => o.t != null && o.t >= t0 && o.t <= t1)
      .sort((a, b) => a.t - b.t);
    const future = svg('g', { class: 'onset', opacity: FUTURE_OPACITY });
    const past = svg('g', { class: 'onset', 'clip-path': `url(#${clipId})` });
    drawSeries(future, x, y, times, series, onsets, width);
    drawSeries(past, x, y, times, series, onsets, width);
    root.append(future, past);

    cursorLine = svg('line', { class: 'cursor', y1: MARGIN.top - 4, y2: MARGIN.top + plotH });
    root.append(cursorLine);
    const hit = svg('rect', {
      x: MARGIN.left, y: MARGIN.top, width: plotW, height: plotH, fill: 'transparent',
    });
    root.append(hit);
    positionCursor();
  }

  function positionCursor() {
    if (!cursorLine || !scale || !data) return;
    const t = data.times[Math.min(cursor, data.times.length - 1)];
    const px = scale.x(t);
    cursorLine.setAttribute('x1', String(px));
    cursorLine.setAttribute('x2', String(px));
    clipRect.setAttribute('width', String(Math.max(0, px - MARGIN.left + 1)));
  }

  function indexAt(event) {
    if (!scale || !data) return null;
    const rect = root.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * Number(root.getAttribute('width'));
    const t = scale.t0 + ((px - MARGIN.left) / scale.plotW) * (scale.t1 - scale.t0);
    return nearestIndex(data.times, t);
  }

  function showTip(event) {
    const i = indexAt(event);
    if (i == null) return;
    const rows = data.series.map((s) =>
      el('div', { class: 'chart-tip__row' }, [
        el('span', { class: 'dot', style: { background: s.color } }),
        el('span', { text: s.label }),
        el('strong', { text: s.values[i] == null ? '—' : pct(s.values[i]) }),
      ]),
    );
    tip.replaceChildren(el('div', { class: 'chart-tip__time', text: clock(data.times[i]) }), ...rows);
    tip.hidden = false;
    const wrapRect = wrap.getBoundingClientRect();
    const left = event.clientX - wrapRect.left + 14;
    const maxLeft = wrapRect.width - tip.offsetWidth - 4;
    tip.style.setProperty('left', `${Math.min(left, maxLeft)}px`);
    tip.style.setProperty('top', `${Math.max(0, event.clientY - wrapRect.top - 20)}px`);
  }

  root.addEventListener('mousemove', showTip);
  root.addEventListener('mouseleave', () => { tip.hidden = true; });
  root.addEventListener('click', (event) => {
    const i = indexAt(event);
    if (i != null) onScrub(i);
  });
  const observer = new ResizeObserver(() => draw());
  observer.observe(wrap);

  return {
    update(next) {
      data = next;
      draw();
    },
    setCursor(i) {
      cursor = i;
      positionCursor();
    },
    redraw: draw,
    destroy() {
      observer.disconnect();
      wrap.remove();
    },
  };
}
