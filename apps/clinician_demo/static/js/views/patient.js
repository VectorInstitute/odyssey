/**
 * Patient page: who they are and the list of admissions to replay.
 */

import { api } from '../api.js';
import { el, loadingBlock, errorBlock, emptyBlock } from '../dom.js';
import { ageSex, duration } from '../format.js';
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
  const n = patient.visits.length;
  const rows = patient.visits.map((v) =>
    el('a', { class: 'visit-row', href: `#/p/${patient.subject_id}/v/${v.visit_id}` }, [
      el('div', { class: 'grow' }, [
        el('div', { class: 'visit-row__title', text: v.admission }),
        el('div', { class: 'muted', text: `${duration(v.end_hours - v.start_hours)} stay · ${v.n_events.toLocaleString()} chart entries` }),
      ]),
      el('span', { class: 'btn btn--small btn--ghost', text: 'Replay' }),
    ]),
  );
  root.replaceChildren(
    el('a', { href: '#/', class: 'backlink', text: '← All patients' }),
    el('div', { class: 'page-head' }, [
      el('h1', {}, [`Patient ${patient.subject_id}`, ' ', trainingBadge(patient.seen_in_training)]),
      el('p', { text: `${ageSex(patient.age_years, patient.sex)} · ${n} admission${n === 1 ? '' : 's'}` }),
    ]),
    rows.length ? el('div', { class: 'visit-list' }, rows) : emptyBlock('This patient has no admissions.'),
  );
  return () => {};
}
