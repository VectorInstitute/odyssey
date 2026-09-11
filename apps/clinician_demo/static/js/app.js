/**
 * Entry point: load model metadata, fill the page chrome, and route
 * between views by URL hash.
 *
 *   #/                    gallery
 *   #/scorecard           report card
 *   #/p/{sid}             one patient's admissions
 *   #/p/{sid}/v/{vid}?t=  replay one admission (optionally at hour t)
 */

import { api } from './api.js';
import { el, errorBlock, loadingBlock } from './dom.js';
import { store } from './state.js';
import { renderGallery } from './views/gallery.js';
import { renderPatient } from './views/patient.js';
import { renderReplay } from './views/replay.js';
import { renderScorecard } from './views/scorecard.js';

const main = document.getElementById('app');
let cleanup = () => {};
let renderToken = 0;

/**
 * Parse a location hash into a route.
 * @param {string} hash
 * @returns {{name: string, sid?: string, vid?: string, t?: number|null}}
 */
export function parseRoute(hash) {
  const raw = hash.replace(/^#/, '') || '/';
  const [path, query = ''] = raw.split('?');
  const params = new URLSearchParams(query);
  const tRaw = params.get('t');
  const t = tRaw != null && tRaw !== '' && !Number.isNaN(Number(tRaw)) ? Number(tRaw) : null;
  let m = path.match(/^\/p\/(\d+)\/v\/(\d+)\/?$/);
  if (m) return { name: 'replay', sid: m[1], vid: m[2], t };
  m = path.match(/^\/p\/(\d+)\/?$/);
  if (m) return { name: 'patient', sid: m[1] };
  if (path === '/scorecard') return { name: 'scorecard' };
  if (path === '/' || path === '') return { name: 'gallery' };
  return { name: 'notfound' };
}

function markNav(route) {
  const active = route.name === 'scorecard' ? 'scorecard' : 'gallery';
  document.querySelectorAll('[data-nav]').forEach((a) => {
    a.classList.toggle('is-active', a.dataset.nav === active);
    if (a.dataset.nav === active) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });
}

async function render() {
  const token = ++renderToken;
  cleanup();
  cleanup = () => {};
  const meta = store.get().meta;
  const route = parseRoute(window.location.hash);
  store.set({ route });
  markNav(route);
  window.scrollTo({ top: 0 });
  const views = {
    gallery: () => renderGallery(main, { meta }),
    scorecard: () => renderScorecard(main, { meta }),
    patient: () => renderPatient(main, route),
    replay: () => renderReplay(main, { meta, ...route }),
  };
  const view = views[route.name];
  if (!view) {
    main.replaceChildren(errorBlock('That page does not exist.'), el('p', {}, el('a', { href: '#/', text: 'Back to patients' })));
    return;
  }
  const done = await view();
  if (token === renderToken) cleanup = done;
  else done();
  main.focus({ preventScroll: true });
}

function fillChrome(meta) {
  document.getElementById('provenance').textContent =
    `Model ${meta.run_name} · ${meta.checkpoint} · forecasts ${meta.events.length} events at ${meta.horizons.join(' / ')} h`;
  const badge = document.getElementById('mode-badge');
  badge.hidden = false;
  badge.textContent = meta.data_mode === 'open' ? 'Open demo data' : 'Credentialed data · PhysioNet DUA';
  badge.className = `mode-badge mode-badge--${meta.data_mode}`;
  const banner = document.getElementById('banner');
  banner.textContent = [meta.disclaimers.banner, meta.disclaimers[meta.data_mode]].filter(Boolean).join(' ');
  document.getElementById('footer').textContent =
    'Times are hours since the start of each admission. Dates are never shown.';
  const search = document.getElementById('search');
  search.hidden = !meta.searchable;
  search.addEventListener('submit', (event) => {
    event.preventDefault();
    const id = new FormData(search).get('patient')?.toString().trim() ?? '';
    if (/^\d+$/.test(id)) window.location.hash = `#/p/${id}`;
  });
}

async function start() {
  main.replaceChildren(loadingBlock('Loading the model…'));
  try {
    const meta = await api.meta();
    store.set({ meta });
    fillChrome(meta);
  } catch (err) {
    main.replaceChildren(errorBlock(err));
    document.getElementById('provenance').textContent = 'Model unavailable';
    return;
  }
  window.addEventListener('hashchange', render);
  render();
}

start();
