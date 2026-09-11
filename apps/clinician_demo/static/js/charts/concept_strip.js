/**
 * Concept heat strip (canvas): one row per named concept, time left to
 * right, darker = the model believes the concept is more likely present
 * this visit. Row labels are HTML so they stay crisp and readable by
 * screen readers.
 */

import { el } from '../dom.js';
import { clock, pct, nearestIndex } from '../format.js';
import { cssVar, hexToRgb } from '../theme.js';

const ROW_HEIGHT = 14;

/**
 * Create the strip inside a container.
 * @param {HTMLElement} container
 * @param {{concepts: object[], onScrub: (index: number) => void}} opts
 * @returns {{update: (data: {times: number[], values: number[][]}) => void,
 *   setCursor: (i: number) => void, redraw: () => void, destroy: () => void}}
 */
export function createConceptStrip(container, { concepts, onScrub }) {
  const labels = el(
    'ul',
    { class: 'concept-strip__labels', 'aria-hidden': 'true' },
    concepts.map((c) => el('li', { title: c.description, text: c.display })),
  );
  const canvas = el('canvas', {
    role: 'img',
    'aria-label': 'How strongly the model believes each clinical concept is present, over time',
  });
  const cursorLine = el('div', { class: 'concept-strip__cursor' });
  const tip = el('div', { class: 'chart-tip', hidden: true });
  const plot = el('div', { class: 'concept-strip__plot' }, [canvas, cursorLine, tip]);
  const wrap = el('div', { class: 'concept-strip' }, [labels, plot]);
  container.append(wrap);

  const height = concepts.length * ROW_HEIGHT;
  let data = null;
  let cursor = 0;

  function columnEdges(width) {
    const { times } = data;
    const t0 = times[0];
    const t1 = times[times.length - 1] > t0 ? times[times.length - 1] : t0 + 1;
    const x = (t) => ((t - t0) / (t1 - t0)) * width;
    return { x, t0, t1 };
  }

  function draw() {
    if (!data || !data.times.length) return;
    const width = Math.max(200, plot.clientWidth || 600);
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    canvas.style.setProperty('height', `${height}px`);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.fillStyle = cssVar('--surface-2', '#ECF1F2');
    ctx.fillRect(0, 0, width, height);
    const [r, g, b] = hexToRgb(cssVar('--accent', '#0B6E77'));
    const { times, values } = data;
    const { x } = columnEdges(width);
    for (let i = 0; i < times.length; i += 1) {
      const left = i === 0 ? 0 : (x(times[i - 1]) + x(times[i])) / 2;
      const right = i === times.length - 1 ? width : (x(times[i]) + x(times[i + 1])) / 2;
      const w = Math.max(1, right - left + 0.5);
      const row = values[i] ?? [];
      for (let c = 0; c < concepts.length; c += 1) {
        const v = row[c];
        if (v == null || v < 0.02) continue;
        // Squared so a 0.9 belief reads clearly darker than a 0.6 one.
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${(0.06 + 0.94 * v * v).toFixed(3)})`;
        ctx.fillRect(left, c * ROW_HEIGHT + 1, w, ROW_HEIGHT - 2);
      }
    }
    positionCursor();
  }

  function positionCursor() {
    if (!data || !data.times.length) return;
    const width = plot.clientWidth || 600;
    const { x } = columnEdges(width);
    cursorLine.style.setProperty('left', `${x(data.times[Math.min(cursor, data.times.length - 1)]) - 1}px`);
    cursorLine.style.setProperty('height', `${height}px`);
    const row = data.values[cursor] ?? [];
    labels.querySelectorAll('li').forEach((li, c) => li.classList.toggle('is-hot', (row[c] ?? 0) >= 0.5));
  }

  function locate(event) {
    const rect = canvas.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    const { t0, t1 } = columnEdges(rect.width);
    const i = nearestIndex(data.times, t0 + (px / rect.width) * (t1 - t0));
    const c = Math.min(concepts.length - 1, Math.max(0, Math.floor(py / ROW_HEIGHT)));
    return { i, c, px, py };
  }

  canvas.addEventListener('mousemove', (event) => {
    if (!data) return;
    const { i, c, px, py } = locate(event);
    const concept = concepts[c];
    tip.replaceChildren(
      el('div', { class: 'chart-tip__time', text: `${concept.display} · ${pct(data.values[i]?.[c])}` }),
      el('div', { class: 'muted', text: concept.description }),
      el('div', { class: 'faint', text: clock(data.times[i]) }),
    );
    tip.hidden = false;
    const maxLeft = plot.clientWidth - tip.offsetWidth - 4;
    tip.style.setProperty('left', `${Math.min(px + 14, maxLeft)}px`);
    tip.style.setProperty('top', `${py + 12}px`);
  });
  canvas.addEventListener('mouseleave', () => { tip.hidden = true; });
  canvas.addEventListener('click', (event) => {
    if (data) onScrub(locate(event).i);
  });
  const observer = new ResizeObserver(() => draw());
  observer.observe(plot);

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
