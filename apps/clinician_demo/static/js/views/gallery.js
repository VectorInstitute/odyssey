/**
 * Gallery: curated visits, grouped into early warnings, quiet stays,
 * misses and false alarms. Every section shows its honest summary line.
 */

import { api } from '../api.js';
import { el, loadingBlock, errorBlock, emptyBlock } from '../dom.js';
import { duration } from '../format.js';
import { eventInfo } from '../meta.js';
import { eventColor } from '../theme.js';

const INTRO = {
  early_warning: 'The event happened, and the alert had already been on for hours when it began.',
  quiet: 'No event happened, and the risk stayed low. The model does not flag everyone.',
  miss: 'The event happened with no alert on at that moment.',
  false_alarm: 'The alert came on, but the event never happened during the stay.',
  other: 'Every patient in this dataset.',
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
        title: 'The model saw this patient during training. Its forecasts here are not a fair test.',
        text: 'Seen in training',
      })
    : el('span', { class: 'pill pill--heldout', title: 'The model never saw this patient.', text: 'Unseen patient' });
}

function caseHref(c) {
  return `#/p/${c.subject_id}/v/${c.visit_id}`;
}

function caseCard(meta, c) {
  const color = c.event ? eventColor(c.event) : null;
  const chips = [
    c.event
      ? el('span', { class: 'pill pill--event', style: { background: color }, text: eventInfo(meta, c.event).short })
      : null,
    c.lead_hours != null ? el('span', { class: 'pill', text: `${c.lead_approximate ? '≈' : ''}${Math.round(c.lead_hours)} h of warning` }) : null,
    el('span', { text: `Stay ${duration(c.los_hours)}` }),
    trainingBadge(c.seen_in_training),
  ];
  return el(
    'a',
    { class: 'card case-card', href: caseHref(c), style: color ? { '--case-color': color } : {} },
    [
      el('div', { class: 'case-card__headline', text: c.headline }),
      el('div', { class: 'case-card__meta' }, chips),
      el('div', { class: 'case-card__id', text: `Patient ${c.subject_id} · visit ${c.visit_id}` }),
    ],
  );
}

function compactList(c) {
  return el('li', {}, [
    el('a', { href: caseHref(c), text: `Patient ${c.subject_id}` }),
    ' ',
    el('span', { class: 'muted', text: `· ${c.headline} · ${duration(c.los_hours)}` }),
    c.seen_in_training ? el('span', { class: 'faint', text: ' · seen in training' }) : null,
  ]);
}

function section(meta, s) {
  const body = !s.cases.length
    ? emptyBlock('No visit matched this rule.')
    : s.kind === 'other'
      ? el('ul', { class: 'compact-list' }, s.cases.map(compactList))
      : el('div', { class: 'case-grid' }, s.cases.map((c) => caseCard(meta, c)));
  return el('section', { class: 'section', 'aria-labelledby': `sec-${s.kind}` }, [
    el('div', { class: 'section__head' }, [
      el('h2', { id: `sec-${s.kind}`, text: s.title }),
      el('span', { class: 'section__kind', text: `${s.cases.length} shown` }),
    ]),
    el('p', { class: 'section__summary' }, [INTRO[s.kind] ? `${INTRO[s.kind]} ` : '', el('strong', { text: s.summary })]),
    body,
  ]);
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
  const head = el('div', { class: 'page-head' }, [
    el('div', {}, [
      el('h1', { text: 'Replay a real admission' }),
      el('p', {
        text:
          'Pick a stay. The model reads the chart one event at a time, as it was written, and forecasts ' +
          'what happens next. It never sees the future. We show the hits and the misses.',
      }),
    ]),
  ]);
  const sections = gallery.sections.length
    ? gallery.sections.map((s) => section(meta, s))
    : [emptyBlock('No patients are available.')];
  root.replaceChildren(head, ...sections);
  return () => {};
}
