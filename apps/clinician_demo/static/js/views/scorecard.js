/**
 * Scorecard: how well the model ranks patients for each event and
 * horizon, against the tuned GBM, plus calibration and concept readouts.
 */

import { api } from '../api.js';
import { el, svg, card, loadingBlock, errorBlock, emptyBlock } from '../dom.js';
import { auroc, interval, pct } from '../format.js';
import { eventInfo } from '../meta.js';
import { eventColor } from '../theme.js';

function verdict(cell) {
  if (cell.delta == null) return null;
  if (!cell.separated) return el('span', { class: 'verdict verdict--tie', text: 'No clear difference' });
  return cell.delta < 0
    ? el('span', { class: 'verdict verdict--gbm', text: 'GBM clearly better' })
    : el('span', { class: 'verdict verdict--model', text: 'Model clearly better' });
}

function calibration(bins) {
  if (!bins.length) return null;
  const size = 64;
  const top = Math.max(0.01, ...bins.flatMap((b) => [b.predicted, b.observed]));
  const s = (v) => (v / top) * (size - 6) + 3;
  const pts = bins.map((b) => `${s(b.predicted).toFixed(1)},${(size - s(b.observed)).toFixed(1)}`);
  return svg('svg', { class: 'calib', width: size, height: size, viewBox: `0 0 ${size} ${size}`, role: 'img',
    'aria-label': 'Calibration: predicted against observed risk by decile' }, [
    svg('title', { text: 'Predicted (across) against observed (up), by decile. On the dashed line = well calibrated.' }),
    svg('line', { class: 'diag', x1: 3, y1: size - 3, x2: size - 3, y2: 3 }),
    svg('polyline', { class: 'curve', points: pts.join(' ') }),
    ...bins.map((b) => svg('circle', { class: 'pt', cx: s(b.predicted), cy: size - s(b.observed), r: 1.8 })),
  ]);
}

function cellView(cell) {
  const line = (who, value, ci) =>
    el('div', { class: 'score-cell__line' }, [
      el('span', { class: 'score-cell__who', text: who }),
      el('span', { class: 'score-cell__auroc', text: auroc(value) }),
      el('span', { class: 'score-cell__ci', text: interval(ci) }),
    ]);
  return el('div', { class: 'score-cell' }, [
    line('Model', cell.hazard_auroc, cell.hazard_ci),
    line('GBM', cell.gbm_auroc, cell.gbm_ci),
    verdict(cell),
    el('div', { class: 'faint', text: `${pct(cell.base_rate)} of ${cell.n_at_risk.toLocaleString()} moments` }),
  ]);
}

function table(meta, cells) {
  const horizons = [...new Set(cells.map((c) => c.horizon_hours))].sort((a, b) => a - b);
  const events = [...new Set(cells.map((c) => c.event))];
  const find = (e, h) => cells.find((c) => c.event === e && c.horizon_hours === h);
  return el('div', { class: 'card' }, [
    el('table', { class: 'score-table' }, [
      el('thead', {}, el('tr', {}, [
        el('th', { text: 'Event' }),
        ...horizons.map((h) => el('th', { text: `Within ${h} h` })),
        el('th', { text: 'Calibration (24 h)' }),
      ])),
      el('tbody', {}, events.map((e) => {
        const info = eventInfo(meta, e);
        const c24 = find(e, 24);
        return el('tr', {}, [
          el('td', {}, [
            el('div', { class: 'risk-card__name' }, [el('span', { class: 'dot', style: { background: eventColor(e) } }), info.display]),
            el('div', { class: 'faint', text: info.definition }),
          ]),
          ...horizons.map((h) => el('td', {}, find(e, h) ? cellView(find(e, h)) : '—')),
          el('td', {}, c24 ? calibration(c24.calibration) : '—'),
        ]);
      })),
    ]),
  ]);
}

function conceptBars(concepts) {
  const scored = concepts.filter((c) => c.readout_auroc != null).sort((a, b) => b.readout_auroc - a.readout_auroc);
  if (!scored.length) return emptyBlock('No concept readouts are banked for this run.');
  return el('div', { class: 'auroc-bars' }, scored.map((c) => {
    const fill = el('div', { class: 'auroc-bar__fill' });
    fill.style.setProperty('width', `${Math.max(0, (c.readout_auroc - 0.5) / 0.5) * 100}%`);
    return el('div', { class: 'auroc-bar', title: c.description }, [
      el('span', { text: c.display }),
      el('div', { class: 'auroc-bar__track' }, [fill]),
      el('span', { class: 'num', text: auroc(c.readout_auroc) }),
    ]);
  }));
}

/**
 * Render the scorecard page.
 * @param {HTMLElement} root
 * @param {{meta: object}} ctx
 * @returns {Promise<() => void>} cleanup
 */
export async function renderScorecard(root, { meta }) {
  root.replaceChildren(loadingBlock('Loading the report card…'));
  let sc;
  try {
    sc = await api.scorecard();
  } catch (err) {
    root.replaceChildren(errorBlock(err));
    return () => {};
  }
  root.replaceChildren(
    el('div', { class: 'page-head' }, [
      el('div', {}, [
        el('h1', { text: 'How good is it?' }),
        el('p', { text: 'Measured on held-out patients the model never trained on. Bars start at 0.5, which is chance.' }),
      ]),
    ]),
    el('div', { class: 'card headline-card', text: sc.headline }),
    el('div', { class: 'two-col' }, [
      el('div', {}, [sc.cells.length ? table(meta, sc.cells) : emptyBlock('No alert evaluation is banked for this run.')]),
      el('div', { class: 'replay__main' }, [
        card('Reading the chart: concept accuracy', conceptBars(sc.concepts), {
          sub: 'AUROC of each named concept against its rule',
        }),
        card('How to read this', el('ul', { class: 'notes' }, sc.notes.map((n) => el('li', { text: n })))),
      ]),
    ]),
  );
  return () => {};
}
