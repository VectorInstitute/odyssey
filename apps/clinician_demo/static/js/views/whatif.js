/**
 * "What if…" panel: change up to three recent readings and see how the
 * model's forecast at the current moment moves. This shows the model's
 * sensitivity, not the effect of a treatment.
 */

import { api } from '../api.js';
import { el, errorBlock, loadingBlock } from '../dom.js';
import { clock, pct, points } from '../format.js';
import { eventColor } from '../theme.js';

const MAX_EDITS = 3;
let presetsPromise = null;

function loadPresets() {
  presetsPromise ??= api.presets().catch((err) => {
    presetsPromise = null;
    throw err;
  });
  return presetsPromise;
}

function formatValue(preset, value) {
  if (preset.mode === 'scale') return `×${Number(value).toFixed(1)}`;
  const sign = preset.mode === 'add' && value > 0 ? '+' : '';
  return `${sign}${Number(value)}${preset.unit ? ` ${preset.unit}` : ''}`;
}

function deltaClass(d) {
  if (d == null || Math.abs(d) < 0.0005) return 'delta-flat';
  return d > 0 ? 'delta-up' : 'delta-down';
}

function resultView(result, events, onsets) {
  const happened = (e) => onsets?.[e.name] != null && onsets[e.name] <= result.t_hours;
  const open = events.filter((e) => result.factual.risk[e.name] && !happened(e));
  const max = Math.max(
    0.01,
    ...open.flatMap((e) => [result.factual.risk[e.name]['24h'] ?? 0, result.counterfactual.risk[e.name]?.['24h'] ?? 0]),
  );
  const done = events
    .filter(happened)
    .map((e) => el('div', { class: 'compare-row faint' }, [
      el('strong', { text: e.display }),
      el('span', { text: `Already happened at ${clock(onsets[e.name])}. Nothing left to forecast.` }),
      el('span'),
    ]));
  const rows = open
    .map((e) => {
      const before = result.factual.risk[e.name]['24h'];
      const after = result.counterfactual.risk[e.name]['24h'];
      const d = result.delta.risk[e.name]?.['24h'];
      const color = eventColor(e.name);
      const minor = ['8h', '72h']
        .map((k) => `${k.replace('h', ' h')} ${points(result.delta.risk[e.name]?.[k])}`)
        .join(' · ');
      return el('div', { class: 'compare-row' }, [
        el('div', {}, [el('strong', { text: e.display }), el('div', { class: 'muted', text: minor })]),
        el('div', { class: 'compare-bars', 'aria-label': `${e.display}: ${pct(before)} now, ${pct(after)} with the change` }, [
          el('div', { class: 'compare-bar' }, [
            el('div', { class: 'compare-bar__fill compare-bar__fill--before', style: { width: `${(before / max) * 100}%` } }),
          ]),
          el('div', { class: 'compare-bar' }, [
            el('div', { class: 'compare-bar__fill', style: { width: `${(after / max) * 100}%`, background: color } }),
          ]),
        ]),
        el('div', { class: `compare-row__delta ${deltaClass(d)}` }, [
          `${pct(before)} → ${pct(after)}`,
          el('div', { text: points(d) }),
        ]),
      ]);
    });
  return el('div', { class: 'panel' }, [
    el('div', { class: 'muted' }, [
      `Forecast at ${clock(result.t_hours)} · ${result.rows_edited} reading${result.rows_edited === 1 ? '' : 's'} changed. `,
      'Grey bar: as charted. Coloured bar: with your change. Risk in the next 24 h.',
    ]),
    ...result.warnings.map((w) => el('div', { class: 'note', text: w })),
    result.rows_edited > 0 ? el('div', { class: 'compare' }, [...rows, ...done]) : null,
  ]);
}

/**
 * Create the what-if panel.
 * @param {HTMLElement} container
 * @param {{meta: object, sid: string, vid: string, getT: () => number, events: object[],
 *   onsets?: Object<string, number|null>}} ctx onsets: visit hours each event happened
 * @returns {{destroy: () => void}}
 */
export function createWhatIfPanel(container, { meta, sid, vid, getT, events, onsets = {} }) {
  let destroyed = false;
  const edits = [];
  const rowsBox = el('div', { class: 'panel' });
  const resultBox = el('div');
  const select = el('select', { 'aria-label': 'Choose a reading to change' });
  const addBtn = el('button', { class: 'btn btn--ghost btn--small', type: 'button', text: 'Add' });
  const runBtn = el('button', { class: 'btn', type: 'button', text: 'Re-run the forecast', disabled: true });
  const controls = el('div', { class: 'panel__controls' }, [select, addBtn, el('span', { class: 'spacer' }), runBtn]);
  const panel = el('div', { class: 'panel' }, [
    el('div', { class: 'note', text: meta.disclaimers.whatif }),
    controls,
    rowsBox,
    resultBox,
  ]);
  container.replaceChildren(panel);
  let presets = [];

  function renderRows() {
    rowsBox.replaceChildren(
      ...edits.map((edit, i) => {
        const valueLabel = el('span', { class: 'edit-row__value', text: formatValue(edit.preset, edit.value) });
        const slider = el('input', {
          type: 'range', min: edit.preset.min, max: edit.preset.max, step: edit.preset.step,
          value: edit.value, 'aria-label': `${edit.preset.label} value`,
        });
        slider.addEventListener('input', () => {
          edit.value = Number(slider.value);
          valueLabel.textContent = formatValue(edit.preset, edit.value);
        });
        return el('div', { class: 'edit-row' }, [
          el('div', {}, [
            el('div', { class: 'edit-row__name', text: edit.preset.label }),
            el('div', { class: 'edit-row__hint', text: edit.preset.description }),
          ]),
          slider,
          valueLabel,
          el('button', {
            class: 'btn btn--ghost btn--icon', type: 'button', 'aria-label': `Remove ${edit.preset.label}`, text: '✕',
            onClick: () => { edits.splice(i, 1); renderRows(); },
          }),
        ]);
      }),
    );
    if (!edits.length) rowsBox.append(el('div', { class: 'muted', text: 'Pick a reading above and press Add. Up to three changes.' }));
    addBtn.disabled = edits.length >= MAX_EDITS || !presets.length;
    runBtn.disabled = !edits.length;
  }

  addBtn.addEventListener('click', () => {
    const preset = presets.find((p) => p.id === select.value);
    if (!preset || edits.length >= MAX_EDITS) return;
    edits.push({ preset, value: preset.value });
    renderRows();
  });

  runBtn.addEventListener('click', async () => {
    const t = getT();
    runBtn.disabled = true;
    resultBox.replaceChildren(loadingBlock(`Re-running the model at ${clock(t)}…`));
    try {
      const result = await api.whatif(sid, vid, {
        t_hours: t,
        edits: edits.map((e) => ({ preset: e.preset.id, value: e.value })),
      });
      if (!destroyed) resultBox.replaceChildren(resultView(result, events, onsets));
    } catch (err) {
      if (!destroyed) resultBox.replaceChildren(errorBlock(err));
    } finally {
      runBtn.disabled = !edits.length;
    }
  });

  loadPresets()
    .then((list) => {
      if (destroyed) return;
      presets = list;
      select.replaceChildren(...list.map((p) => el('option', { value: p.id, text: p.label, title: p.description })));
      renderRows();
    })
    .catch((err) => {
      if (!destroyed) rowsBox.replaceChildren(errorBlock(err));
    });
  renderRows();

  return {
    destroy() {
      destroyed = true;
      panel.remove();
    },
  };
}
