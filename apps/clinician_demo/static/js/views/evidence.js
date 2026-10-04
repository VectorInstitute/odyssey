/**
 * "Why?" panel: which recent recorded events the forecast leans on. The
 * server removes each recent code in turn and re-runs the model; this
 * panel starts that job, shows progress, and ranks the results.
 */

import { api } from '../api.js';
import { el, errorBlock, emptyBlock } from '../dom.js';
import { clock, pct, points } from '../format.js';

const POLL_MS = 1000;
const LOOKBACK_HOURS = 24;

function resultRow(item, maxAbs) {
  const effect = -item.delta; // how much this code raised the target
  const width = maxAbs > 0 ? (Math.abs(effect) / maxAbs) * 50 : 0;
  const raises = effect > 0;
  return el('li', { class: 'evidence-row' }, [
    el('div', { class: 'evidence-row__label', title: item.code }, [
      el('strong', { text: item.label }),
      el('span', { class: 'faint', text: ` · ${item.n_rows} reading${item.n_rows === 1 ? '' : 's'}` }),
    ]),
    el('div', { class: 'diverge', 'aria-hidden': 'true' }, [
      el('div', {
        class: `diverge__fill ${raises ? 'diverge__fill--up' : 'diverge__fill--down'}`,
        style: { width: `${width}%` },
      }),
    ]),
    el('div', { class: raises ? 'delta-up' : 'delta-down', text: `${raises ? 'Raises' : 'Lowers'} ${points(Math.abs(effect)).replace('+', '')}` }),
  ]);
}

function doneView(job, targetLabel) {
  const items = job.result;
  if (!items.length) return emptyBlock('Nothing charted recently moved this forecast.');
  const maxAbs = Math.max(...items.map((i) => Math.abs(i.delta)));
  return el('div', { class: 'panel' }, [
    el('div', { class: 'muted' }, [
      `${targetLabel}: ${pct(items[0].baseline)} as recorded. `,
      'Each bar shows how the forecast would change if that item had not been charted.',
    ]),
    job.note ? el('div', { class: 'muted', text: job.note }) : null,
    el('ol', { class: 'evidence-list' }, items.map((i) => resultRow(i, maxAbs))),
  ]);
}

/**
 * Create the evidence panel.
 * @param {HTMLElement} container
 * @param {{meta: object, sid: string, vid: string, getT: () => number, events: object[],
 *   onsets?: Object<string, number|null>}} ctx onsets: visit hours each event happened
 * @returns {{destroy: () => void}}
 */
export function createEvidencePanel(container, { meta, sid, vid, getT, events, onsets = {} }) {
  let destroyed = false;
  let timer = null;
  const select = el('select', { 'aria-label': 'What to explain' }, [
    el('optgroup', { label: 'Risk in the next 24 h' }, events.map((e) => el('option', { value: `event:${e.name}`, text: e.display }))),
    el('optgroup', { label: 'What the model thinks is going on' }, meta.concepts.map((c) => el('option', { value: `concept:${c.name}`, text: c.display }))),
  ]);
  const runBtn = el('button', { class: 'btn', type: 'button', text: 'Explain' });
  const out = el('div');
  const panel = el('div', { class: 'panel' }, [
    el('div', { class: 'note', text: meta.disclaimers.evidence }),
    el('div', { class: 'panel__controls' }, [select, runBtn, el('span', { class: 'muted', text: `Looks at what was charted in the last ${LOOKBACK_HOURS} h.` })]),
    out,
  ]);
  container.replaceChildren(panel);

  function progress(job) {
    const share = job.total ? job.done / job.total : 0;
    const bar = el('div', { class: 'progress__bar' });
    bar.style.setProperty('width', `${Math.round(share * 100)}%`);
    return el('div', { class: 'panel', role: 'status' }, [
      el('div', { class: 'muted', text: job.total ? `Re-running the model without each event: ${job.done} of ${job.total}.` : 'Starting…' }),
      el('div', { class: 'progress' }, [bar]),
    ]);
  }

  async function poll(jobId, label) {
    if (destroyed) return;
    try {
      const job = await api.job(jobId);
      if (destroyed) return;
      if (job.status === 'done') {
        out.replaceChildren(doneView(job, label));
        runBtn.disabled = false;
      } else if (job.status === 'error') {
        out.replaceChildren(errorBlock(job.error || 'The explanation failed.'));
        runBtn.disabled = false;
      } else {
        out.replaceChildren(progress(job));
        timer = setTimeout(() => poll(jobId, label), POLL_MS);
      }
    } catch (err) {
      if (!destroyed) {
        out.replaceChildren(errorBlock(err));
        runBtn.disabled = false;
      }
    }
  }

  runBtn.addEventListener('click', async () => {
    const [kind, name] = select.value.split(':');
    const label = select.selectedOptions[0]?.textContent ?? name;
    const target = kind === 'event' ? { kind, name, horizon_hours: 24 } : { kind, name };
    const t = getT();
    if (kind === 'event' && onsets[name] != null && onsets[name] <= t) {
      clearTimeout(timer);
      out.replaceChildren(el('div', {
        class: 'note',
        text: `${label} had already happened at ${clock(onsets[name])}. Move to an earlier moment to see what the forecast leaned on.`,
      }));
      return;
    }
    runBtn.disabled = true;
    clearTimeout(timer);
    out.replaceChildren(progress({ done: 0, total: 0 }));
    try {
      const job = await api.evidence(sid, vid, { t_hours: t, target, lookback_hours: LOOKBACK_HOURS });
      if (destroyed) return;
      out.prepend(el('div', { class: 'muted', text: `Explaining ${label} at ${clock(t)}.` }));
      poll(job.job_id, label);
    } catch (err) {
      if (!destroyed) {
        out.replaceChildren(errorBlock(err));
        runBtn.disabled = false;
      }
    }
  });

  return {
    destroy() {
      destroyed = true;
      clearTimeout(timer);
      panel.remove();
    },
  };
}
