/**
 * Replay: the hero screen. Scrub (or play) through one admission and see,
 * at every moment, the model's risks, its alert crossings, what it thinks
 * is going on, what it expects next, and what was just recorded.
 */

import { api } from '../api.js';
import { el, card, loadingBlock, errorBlock, emptyBlock } from '../dom.js';
import { clamp, clock, duration, indexAtOrBefore, pct, timesTypical, trend } from '../format.js';
import { eventInfo, horizonKey, operatingPoint } from '../meta.js';
import { eventColor, onThemeChange } from '../theme.js';
import { createRiskChart } from '../charts/risk_chart.js';
import { createConceptStrip } from '../charts/concept_strip.js';
import { trainingBadge } from './gallery.js';
import { createWhatIfPanel } from './whatif.js';
import { createEvidencePanel } from './evidence.js';

const TREND_HOURS = 6;
const RECENT_HOURS = 6;
const RECENT_MAX = 40;
const PLAY_MS = 300;
const PLAY_TICKS = 150; // a whole visit plays in about 45 s
const URL_THROTTLE_MS = 400;

function level(p, threshold) {
  if (p == null || threshold == null) return 'none';
  const r = p / threshold;
  if (r >= 1) return 'alert';
  return r >= 0.5 ? 'watch' : 'low';
}

function buildCard(meta, event) {
  const nodes = {
    value: el('div', { class: 'risk-card__value' }),
    context: el('div', { class: 'risk-card__context' }),
    minor: el('div', { class: 'risk-card__minor' }),
    line: el('div', { class: 'risk-card__line' }),
  };
  const info = eventInfo(meta, event);
  const root = el('article', {
    class: 'card risk-card',
    style: { '--card-color': eventColor(event) },
    'aria-live': 'polite',
    title: info.definition,
  }, [
    el('div', { class: 'risk-card__name' }, [el('span', { class: 'dot', style: { background: eventColor(event) } }), info.display]),
    nodes.value, nodes.context, nodes.minor, nodes.line,
  ]);
  return { root, nodes, event };
}

function recentEntries(timeline, t) {
  const out = [];
  for (let i = timeline.length - 1; i >= 0 && out.length < RECENT_MAX; i -= 1) {
    const e = timeline[i];
    if (e.t > t) continue;
    if (e.t < t - RECENT_HOURS) break;
    out.push(e);
  }
  return out;
}

/**
 * Render the replay view.
 * @param {HTMLElement} root
 * @param {{meta: object, sid: string, vid: string, t: number|null}} ctx
 * @returns {Promise<() => void>} cleanup
 */
export async function renderReplay(root, { meta, sid, vid, t }) {
  root.replaceChildren(loadingBlock('Reading the chart and running the model. This can take a few seconds…'));
  let patient;
  let trace;
  try {
    [patient, trace] = await Promise.all([api.patient(sid), api.trace(sid, vid)]);
  } catch (err) {
    root.replaceChildren(errorBlock(err));
    return () => {};
  }
  const { times } = trace;
  if (!times.length) {
    root.replaceChildren(emptyBlock('This admission has no moments the model could score.'));
    return () => {};
  }

  const cleanups = [];
  const events = meta.events.filter((e) => trace.risk[e.name]);
  const timeline = [...trace.timeline].sort((a, b) => a.t - b.t);
  let index = t != null ? indexAtOrBefore(times, t) : 0;
  let horizon = 24;
  let hidden = new Set();
  let overlayEvent = '';
  let playTimer = null;
  let urlTimer = null;

  // ---- header
  const visit = trace.visit;
  const facts = [
    patient.age_years != null ? ['Age', `${Math.round(patient.age_years)}`] : null,
    patient.sex ? ['Sex', patient.sex] : null,
    ['Admission', visit.admission],
    ['Stay', duration(visit.end_hours - visit.start_hours)],
    ['Recorded events', String(visit.n_events)],
  ].filter(Boolean);
  const header = el('div', { class: 'patient-head' }, [
    el('a', { href: '#/', class: 'btn btn--ghost btn--small', text: '← Patients' }),
    el('h1', { text: `Patient ${trace.subject_id}` }),
    el('div', { class: 'patient-head__facts' }, facts.map(([k, v]) => el('span', {}, [`${k} `, el('strong', { text: v })]))),
    trainingBadge(trace.seen_in_training),
  ]);

  // ---- moment + cards
  const momentTime = el('span', { class: 'moment__time' });
  const momentNote = el('span', { class: 'muted' });
  const cards = events.map((e) => buildCard(meta, e.name));
  const cardsRow = el('div', { class: 'risk-cards' }, cards.map((c) => c.root));

  // ---- callouts
  const alerts = [...trace.alerts].sort((a, b) => (b.lead_hours ?? -1) - (a.lead_hours ?? -1));
  const calloutItems = alerts.map((a) =>
    el('li', {
      class: `callout ${a.lead_hours != null ? 'callout--lead' : ''}`,
      style: { '--callout-color': eventColor(a.event) },
      dataset: { event: a.event },
    }, [
      el('span', { class: 'callout__tag', text: eventInfo(meta, a.event).short }),
      el('span', { class: 'callout__body' }, [
        el('span', { text: a.callout }),
        a.detail ? el('span', { class: 'callout__detail', text: a.detail }) : null,
      ]),
    ]),
  );
  const calloutCard = card('Alerts in this stay', calloutItems.length
    ? el('ul', { class: 'callouts' }, calloutItems)
    : el('div', { class: 'muted', text: 'No alert line was crossed during this admission.' }), {
    sub: 'Outlined red while that alert is on at the moment shown',
  });

  // ---- chart
  const chartHost = el('div');
  const legend = el('div', { class: 'legend', role: 'group', 'aria-label': 'Show or hide events' });
  const segmented = el('div', { class: 'segmented', role: 'group', 'aria-label': 'Forecast horizon' });
  const overlaySelect = trace.banked?.length
    ? el('select', { 'aria-label': 'Compare with the tuned GBM' }, [
        el('option', { value: '', text: 'No GBM overlay' }),
        ...events.map((e) => el('option', { value: e.name, text: `GBM: ${e.display}` })),
      ])
    : null;
  const chartTitle = el('h2');
  const chartCard = card(chartTitle, [el('div', { class: 'card__head' }, [legend, el('span', { class: 'spacer' }), overlaySelect]), chartHost], {
    actions: segmented,
  });

  // ---- scrubber
  const range = el('input', {
    type: 'range', min: 0, max: times.length - 1, step: 1, value: index,
    'aria-label': 'Time in the admission. Use the arrow keys to step; hold Shift for bigger steps.',
  });
  const playBtn = el('button', { class: 'btn play-btn', type: 'button', text: '▶ Play' });
  const scrubLabel = el('span', { class: 'scrubber__label', text: `${clock(times[0])} → ${clock(times[times.length - 1])}` });
  const scrubber = el('div', { class: 'card scrubber' }, [playBtn, range, scrubLabel]);

  // ---- concepts
  const conceptHost = el('div');
  const conceptChips = el('div', { class: 'concept-chips', 'aria-live': 'polite' });
  const conceptCard = card('What the model thinks is going on', [
    el('div', { class: 'faint', text: meta.disclaimers.concepts }),
    conceptChips,
    conceptHost,
  ], { sub: 'Darker = more likely during this admission' });

  // ---- tabs
  const tabBody = el('div');
  const tabWhatIf = el('button', { type: 'button', role: 'tab', 'aria-selected': 'true', text: 'What if…' });
  const tabWhy = el('button', { type: 'button', role: 'tab', 'aria-selected': 'false', text: 'Why?' });
  const tabsCard = el('section', { class: 'card' }, [el('div', { class: 'tabs', role: 'tablist' }, [tabWhatIf, tabWhy]), tabBody]);

  // ---- sidebar
  const nextList = el('ul', { class: 'side-list' });
  const recentList = el('ul', { class: 'side-list side-scroll' });
  const side = el('aside', { class: 'replay__side' }, [
    card('What the model expects next', nextList, { sub: 'Most likely next events' }),
    card('Recently recorded', recentList, { sub: `Last ${RECENT_HOURS} h` }),
  ]);

  const main = el('div', { class: 'replay__main' }, [
    el('section', { class: 'card' }, [
      el('div', { class: 'moment' }, [momentTime, momentNote]),
      el('div', { class: 'faint', text: meta.disclaimers.risk }),
    ]),
    cardsRow, calloutCard, chartCard, scrubber, conceptCard, tabsCard,
  ]);
  root.replaceChildren(header, el('div', { class: 'replay' }, [main, side]));

  // ---- charts
  const chart = createRiskChart(chartHost, { onScrub: (i) => setIndex(i) });
  const strip = createConceptStrip(conceptHost, {
    concepts: meta.concepts,
    onScrub: (i) => setIndex(i),
  });
  cleanups.push(() => chart.destroy(), () => strip.destroy());
  strip.update({ times, values: trace.concepts });

  function chartData() {
    const key = horizonKey(horizon);
    return {
      times,
      series: events
        .filter((e) => !hidden.has(e.name))
        .map((e) => ({
          event: e.name,
          label: e.short,
          color: eventColor(e.name),
          values: trace.risk[e.name][key] ?? [],
          threshold: operatingPoint(meta, e.name, horizon)?.threshold ?? null,
        })),
      onsets: events
        .filter((e) => trace.onsets[e.name] != null)
        .map((e) => ({ t: trace.onsets[e.name], label: e.short, color: eventColor(e.name) })),
      markers: trace.markers,
      overlay: overlayEvent && horizon === 24
        ? {
            color: eventColor(overlayEvent),
            label: 'Tuned GBM, 24 h risk, at 4-hourly landmarks',
            points: trace.banked
              .filter((b) => b.event === overlayEvent)
              .map((b) => ({ t: b.t, v: b.gbm_24h })),
          }
        : null,
    };
  }

  function renderLegend() {
    legend.replaceChildren(...events.map((e) => {
      const on = !hidden.has(e.name);
      return el('button', {
        type: 'button', 'aria-pressed': String(on), title: e.definition,
        onClick: () => {
          hidden = new Set(hidden);
          if (on) hidden.add(e.name);
          else hidden.delete(e.name);
          renderLegend();
          chart.update(chartData());
        },
      }, [el('span', { class: 'dot', style: { background: eventColor(e.name) } }), e.short]);
    }));
  }

  function renderSegmented() {
    segmented.replaceChildren(...meta.horizons.map((h) => el('button', {
      type: 'button', 'aria-pressed': String(h === horizon), text: `${h} h`,
      onClick: () => {
        horizon = h;
        renderSegmented();
        chart.update(chartData());
      },
    })));
    chartTitle.textContent = `Chance of each event within ${horizon} hours`;
  }

  function updateCards() {
    const now = times[index];
    const before = indexAtOrBefore(times, now - TREND_HOURS);
    for (const c of cards) {
      const series = trace.risk[c.event];
      const onset = trace.onsets[c.event];
      const op = operatingPoint(meta, c.event, 24);
      const p24 = series['24h']?.[index];
      if (onset != null && now >= onset) {
        c.root.dataset.level = 'done';
        c.nodes.value.replaceChildren('Happened');
        c.nodes.context.textContent = `at ${clock(onset)}`;
        c.nodes.minor.textContent = '';
        c.nodes.line.textContent = 'Forecast stops at the event';
        continue;
      }
      const lvl = level(p24, op?.threshold);
      c.root.dataset.level = lvl;
      c.nodes.value.replaceChildren(pct(p24), el('small', { text: 'in 24 h' }));
      const tr = trend(p24, series['24h']?.[before], TREND_HOURS);
      c.nodes.context.replaceChildren(
        timesTypical(p24, op?.base_rate) ?? '',
        tr.arrow ? el('span', { class: `trend ${tr.dir > 0 ? 'trend--up' : tr.dir < 0 ? 'trend--down' : ''}`, text: ` ${tr.arrow} `, title: tr.label }) : '',
      );
      c.nodes.minor.textContent = `8 h ${pct(series['8h']?.[index])} · 72 h ${pct(series['72h']?.[index])}`;
      const lineText = {
        alert: 'Alert on: above the line',
        watch: `Near the alert line (${pct(op?.threshold)})`,
      };
      c.nodes.line.textContent = op ? lineText[lvl] ?? `Alert line ${pct(op.threshold)}` : '';
    }
  }

  // An alert is "on" exactly when the 24 h risk at this moment is at or
  // above its line (risk is null once the event has happened).
  function updateCallouts() {
    calloutItems.forEach((item, i) => {
      const a = alerts[i];
      const p = trace.risk[a.event]?.['24h']?.[index];
      item.classList.toggle('is-live', p != null && a.threshold != null && p >= a.threshold);
    });
  }

  function updateSide() {
    const next = trace.top_next[index] ?? [];
    nextList.replaceChildren(...(next.length ? next.map((n) => {
      const fill = el('div', { class: 'bar__fill' });
      fill.style.setProperty('width', `${Math.round(n.probability * 100)}%`);
      return el('li', { class: 'next-row' }, [
        el('span', { class: 'next-row__label', title: n.label, text: n.label }),
        el('span', { class: 'next-row__p', text: pct(n.probability) }),
        el('div', { class: 'bar' }, [fill]),
      ]);
    }) : [el('li', { class: 'faint', text: 'Nothing to forecast here.' })]));
    const recent = recentEntries(timeline, times[index]);
    recentList.replaceChildren(...(recent.length ? recent.map((e) => el('li', { class: `event-row event-row--${e.category}` }, [
      el('span', { class: 'event-row__t', text: clock(e.t).replace('Day ', 'D') }),
      el('span', { class: 'event-row__body' }, [
        el('span', { class: `cat-dot cat-dot--${e.category}`, title: e.category }),
        el('span', { class: 'event-row__label', text: e.label }),
        e.value ? el('span', { class: 'event-row__value', text: e.value }) : null,
        e.flag ? el('span', { class: `pill pill--${e.flag}`, text: e.flag.toLowerCase() }) : null,
      ]),
    ])) : [el('li', { class: 'faint', text: 'Nothing recorded in this window.' })]));
  }

  function updateConcepts() {
    const row = trace.concepts[index] ?? [];
    const top = meta.concepts
      .map((c, i) => ({ c, v: row[i] ?? 0 }))
      .filter((x) => x.v >= 0.5)
      .sort((a, b) => b.v - a.v)
      .slice(0, 6);
    conceptChips.replaceChildren(...(top.length
      ? top.map((x) => el('span', { class: 'concept-chip', title: x.c.description }, [x.c.display, el('span', { class: 'num', text: pct(x.v) })]))
      : [el('span', { class: 'faint', text: 'Nothing stands out yet.' })]));
  }

  function syncUrl() {
    clearTimeout(urlTimer);
    urlTimer = setTimeout(() => {
      history.replaceState(null, '', `#/p/${sid}/v/${vid}?t=${times[index].toFixed(2)}`);
    }, URL_THROTTLE_MS);
  }

  function setIndex(i) {
    index = clamp(Math.round(i), 0, times.length - 1);
    range.value = String(index);
    momentTime.textContent = clock(times[index]);
    momentNote.textContent = `${Math.round(times[index])} h after admission · moment ${index + 1} of ${times.length}`;
    chart.setCursor(index);
    strip.setCursor(index);
    updateCards();
    updateCallouts();
    updateSide();
    updateConcepts();
    syncUrl();
  }

  function stopPlay() {
    clearInterval(playTimer);
    playTimer = null;
    playBtn.textContent = '▶ Play';
  }

  function togglePlay() {
    if (playTimer) {
      stopPlay();
      return;
    }
    if (index >= times.length - 1) setIndex(0);
    const step = Math.max(1, Math.round(times.length / PLAY_TICKS));
    playBtn.textContent = '❚❚ Pause';
    playTimer = setInterval(() => {
      if (index >= times.length - 1) stopPlay();
      else setIndex(index + step);
    }, PLAY_MS);
  }

  // ---- wiring
  range.addEventListener('input', () => setIndex(Number(range.value)));
  playBtn.addEventListener('click', togglePlay);
  if (overlaySelect) {
    overlaySelect.addEventListener('change', () => {
      overlayEvent = overlaySelect.value;
      chart.update(chartData());
    });
  }
  const onKey = (event) => {
    const tag = event.target?.tagName;
    const typing = tag === 'SELECT' || tag === 'TEXTAREA' || (tag === 'INPUT' && event.target.type !== 'range');
    if (typing || event.metaKey || event.ctrlKey || event.altKey) return;
    const stride = event.shiftKey ? 10 : 1;
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      if (event.target === range) return; // the slider handles its own arrows
      event.preventDefault();
      setIndex(index + (event.key === 'ArrowRight' ? stride : -stride));
    } else if (event.key === ' ' && event.target === document.body) {
      event.preventDefault();
      togglePlay();
    } else if (event.key === 'Home') {
      setIndex(0);
    } else if (event.key === 'End') {
      setIndex(times.length - 1);
    }
  };
  document.addEventListener('keydown', onKey);
  cleanups.push(() => document.removeEventListener('keydown', onKey));
  range.addEventListener('keydown', (event) => {
    if (event.shiftKey && (event.key === 'ArrowRight' || event.key === 'ArrowLeft')) {
      event.preventDefault();
      setIndex(index + (event.key === 'ArrowRight' ? 10 : -10));
    }
  });
  cleanups.push(onThemeChange(() => {
    for (const c of cards) c.root.style.setProperty('--card-color', eventColor(c.event));
    renderLegend();
    chart.update(chartData());
    strip.redraw();
  }));

  let panel = null;
  const panelCtx = { meta, sid, vid, events, onsets: trace.onsets, getT: () => times[index] };
  function showTab(which) {
    panel?.destroy();
    tabWhatIf.setAttribute('aria-selected', String(which === 'whatif'));
    tabWhy.setAttribute('aria-selected', String(which === 'why'));
    panel = which === 'whatif' ? createWhatIfPanel(tabBody, panelCtx) : createEvidencePanel(tabBody, panelCtx);
  }
  tabWhatIf.addEventListener('click', () => showTab('whatif'));
  tabWhy.addEventListener('click', () => showTab('why'));
  showTab('whatif');
  cleanups.push(() => panel?.destroy());

  renderLegend();
  renderSegmented();
  chart.update(chartData());
  setIndex(index);

  return () => {
    stopPlay();
    clearTimeout(urlTimer);
    for (const fn of cleanups) fn();
  };
}
