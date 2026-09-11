/**
 * Patient page: header facts and the list of admissions to replay.
 */

import { api } from '../api.js';
import { el, loadingBlock, errorBlock, emptyBlock } from '../dom.js';
import { duration } from '../format.js';
import { trainingBadge } from './gallery.js';

/**
 * Render one patient's visit list.
 * @param {HTMLElement} root
 * @param {{sid: string}} ctx
 * @returns {Promise<() => void>} cleanup
 */
export async function renderPatient(root, { sid }) {
  root.replaceChildren(loadingBlock('Finding the patient…'));
  let patient;
  try {
    patient = await api.patient(sid);
  } catch (err) {
    root.replaceChildren(errorBlock(err));
    return () => {};
  }
  const facts = [
    patient.age_years != null ? `${Math.round(patient.age_years)} years` : null,
    patient.sex ? `Sex ${patient.sex}` : null,
    `${patient.visits.length} admission${patient.visits.length === 1 ? '' : 's'}`,
  ].filter(Boolean);
  const rows = patient.visits.map((v) =>
    el('a', { class: 'card visit-row', href: `#/p/${patient.subject_id}/v/${v.visit_id}` }, [
      el('div', { class: 'grow' }, [
        el('div', { class: 'case-card__headline', text: v.admission }),
        el('div', { class: 'muted', text: `Stay ${duration(v.end_hours - v.start_hours)} · ${v.n_events} recorded events` }),
      ]),
      el('span', { class: 'btn btn--small btn--ghost', text: 'Replay →' }),
    ]),
  );
  root.replaceChildren(
    el('div', { class: 'page-head' }, [
      el('div', {}, [
        el('h1', { text: `Patient ${patient.subject_id}` }),
        el('p', { text: facts.join(' · ') }),
      ]),
      trainingBadge(patient.seen_in_training),
    ]),
    rows.length ? el('div', { class: 'visit-list' }, rows) : emptyBlock('This patient has no admissions.'),
  );
  return () => {};
}
