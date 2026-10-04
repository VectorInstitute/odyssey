/**
 * Gallery: curated admissions grouped by what the model did (warned in
 * time, stayed quiet, missed, or raised a false alarm), then the full
 * patient list behind a disclosure. Every section says how often its
 * pattern happens, so a hand-picked case is never mistaken for the norm.
 */

import { api } from '../api.js';
import { el, loadingBlock, errorBlock, emptyBlock } from '../dom.js';
import { duration } from '../format.js';
import { eventInfo } from '../meta.js';
import { eventColor } from '../theme.js';

const INTRO = {
  early_warning: 'The event happened, and the alert had been on for hours before it began.',
  quiet: 'Nothing happened, and the risk stayed low the whole stay.',
  miss: 'The event happened while no alert was on.',
  false_alarm: 'The alert came on, but the event never happened.',
};

/**
 * Badge saying whether the model saw this patient in training.
 * @param {boolean} seen
 * @returns {HTMLElement}
 */
export function trainingBadge(seen) {
  return seen
    ? el('span', {
        class: 'pill pill--training',
        title: 'This patient was in the model’s training data, so its forecasts here are not a fair test.',
        text: 'In training data',
      })
    : el('span', { class: 'pill pill--heldout', title: 'The model never saw this patient.', text: 'Unseen patient' });
}

function caseHref(c) {
  return `#/p/${c.subject_id}/v/${c.visit_id}`;
}

function keyLine(c) {
  const lead = c.lead_hours != null ? `${c.lead_approximate ? 'About ' : ''}${Math.round(c.lead_hours)} h of warning` : '';
  switch (c.kind) {
    case 'early_warning': return lead || 'Alert on before it began';
    case 'quiet': return `Risk stayed low for ${duration(c.los_hours)}`;
    case 'miss': return 'Began with no alert on';
    case 'false_alarm': return 'Alert came on; it never happened';
    default: return c.headline;
  }
}

function caseCard(meta, c) {
  const info = c.event ? eventInfo(meta, c.event) : null;
  return el('a', { class: 'case', href: caseHref(c), title: c.headline }, [
    el('div', { class: 'case__event' }, [
      info ? el('span', { class: 'dot', style: { background: eventColor(c.event) } }) : null,
      info ? info.display : 'No event',
    ]),
    el('div', { class: 'case__key', text: keyLine(c) }),
    el('div', { class: 'case__meta' }, [
      `Patient ${c.subject_id} · ${duration(c.los_hours)} stay`,
      c.seen_in_training ? null : el('span', { class: 'case__unseen', text: ' · unseen patient' }),
    ]),
  ]);
}

function section(meta, s) {
  const body = !s.cases.length
    ? el('p', { class: 'muted', text: 'No admission in this data matched.' })
    : el('div', { class: 'case-grid' }, s.cases.map((c) => caseCard(meta, c)));
  return el('section', { class: 'section', 'aria-labelledby': `sec-${s.kind}` }, [
    el('div', { class: 'section__head' }, [
      el('h2', { id: `sec-${s.kind}`, text: s.title }),
      el('span', { class: 'section__intro', text: INTRO[s.kind] ?? '' }),
    ]),
    body,
    el('p', { class: 'section__rate', text: `How often: ${s.summary}.` }),
  ]);
}

function allPatients(s) {
  const byPatient = new Map();
  for (const c of s.cases) {
    const entry = byPatient.get(c.subject_id) ?? { sid: c.subject_id, n: 0, seen: c.seen_in_training };
    entry.n += 1;
    byPatient.set(c.subject_id, entry);
  }
  const rows = [...byPatient.values()].sort((a, b) => a.sid - b.sid);
  return el('details', { class: 'section all-patients' }, [
    el('summary', {}, [
      el('span', { class: 'all-patients__title', text: `All ${rows.length} patients` }),
      el('span', { class: 'muted', text: ` · ${s.cases.length} admissions` }),
    ]),
    el('ul', { class: 'patient-list' }, rows.map((r) => el('li', {}, [
      el('a', { href: `#/p/${r.sid}`, text: `Patient ${r.sid}` }),
      el('span', { class: 'muted', text: ` · ${r.n} admission${r.n === 1 ? '' : 's'}` }),
      r.seen ? null : el('span', { class: 'case__unseen', text: ' · unseen' }),
    ]))),
  ]);
}

function trainingNote(other) {
  if (!other) return null;
  const subjects = new Map(other.cases.map((c) => [c.subject_id, c.seen_in_training]));
  const seen = [...subjects.values()].filter(Boolean).length;
  if (!seen) return null;
  return el('p', { class: 'note', text:
    `${seen} of the ${subjects.size} patients here were in the model’s training data. ` +
    'Only the ones marked "unseen" are a fair test of the model.' });
}

/**
 * Render the gallery page.
 * @param {HTMLElement} root
 * @param {{meta: object}} ctx
 * @returns {Promise<() => void>} cleanup
 */
export async function renderGallery(root, { meta }) {
  root.replaceChildren(loadingBlock('Choosing patients…'));
  let gallery;
  try {
    gallery = await api.gallery();
  } catch (err) {
    root.replaceChildren(errorBlock(err));
    return () => {};
  }
  const other = gallery.sections.find((s) => s.kind === 'other');
  const curated = gallery.sections.filter((s) => s.kind !== 'other');
  const head = el('div', { class: 'page-head' }, [
    el('h1', { text: 'Pick an admission to replay' }),
    el('p', {
      text:
        'The model reads the chart one entry at a time, as it was written, and forecasts what happens next. ' +
        'It never sees the future. Hits and misses are both shown.',
    }),
    trainingNote(other),
  ]);
  root.replaceChildren(
    head,
    ...(curated.length ? curated.map((s) => section(meta, s)) : [emptyBlock('No patients are available.')]),
    other ? allPatients(other) : null,
  );
  return () => {};
}
