/**
 * Replay: the hero screen. Scrub (or play) through one admission and see,
 * at every moment, the risk of each event, whether an alert is on, what
 * the model thinks is going on, and what was just charted. Answers a
 * clinician's questions in order: how is this patient right now, how did
 * we get here, what happened in the end, and why does the model say so.
 */

import { api } from '../api.js';
import { el, card, loadingBlock, errorBlock, emptyBlock } from '../dom.js';
import { ageSex, clamp, clock, duration, indexAtOrBefore, pct, timesTypical, trend } from '../format.js';
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
const ALERT_HORIZON = 24;
const CONCEPT_SHOW = 0.5;
const CONCEPT_MAX = 8;

const STATUS = {
  alert: 'Alert on',
  watch: 'Watch',
  low: 'Low',
  none: '',
};

function level(p, threshold) {
  if (p == null || threshold == null) return 'none';
  const r = p / threshold;
  if (r >= 1) return 'alert';
  return r >= 0.5 ? 'watch' : 'low';
}

function buildTile(meta, event) {
  const info = eventInfo(meta, event);
  const nodes = {
    value: el('div', { class: 'tile__value' }),
    status: el('div', { class: 'tile__status' }),
    context: el('div', { class: 'tile__context' }),
  };
  const root = el('article', {
    class: 'tile',
    style: { '--tile-color': eventColor(event) },
    'aria-live': 'polite',
    title: info.definition,
  }, [
    el('div', { class: 'tile__name' }, [el('span', { class: 'dot', style: { background: eventColor(event) } }), info.display]),
    nodes.value, nodes.status, nodes.context,
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

/** Plain words for what the alert line did on this visit, in visit clock time. */
function alertStory(a) {
  const onset = a.onset_hours;
  const start = a.alert_start_hours;
  const cross = a.first_cross_hours;
  if (onset != null) {
    const began = `Began ${clock(onset)}.`;
    if (start != null) {
      const lead = onset - start;
      if (lead < 1) return { text: `${began} The alert came on just before it.`, kind: 'warned' };
      return { text: `${began} The alert had been on since ${clock(start)}: ${Math.round(lead)} h of warning.`, kind: 'warned' };
    }
    if (cross != null) return { text: `${began} The alert came on at ${clock(cross)} but was off again by then.`, kind: 'miss' };
    return { text: `${began} The risk never reached the alert line: a miss.`, kind: 'miss' };
  }
  if (cross != null) return { text: `Did not happen. The alert came on at ${clock(cross)}: a false alarm.`, kind: 'false' };
  return { text: 'Did not happen, and the risk stayed below the alert line.', kind: 'quiet' };
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
  const last = times[times.length - 1];
  let index = t != null ? indexAtOrBefore(times, t) : 0;
  let horizon = ALERT_HORIZON;
  let hidden = new Set();
  let overlayEvent = '';
  let playTimer = null;
  let urlTimer = null;

  // ---- header
  const visit = trace.visit;
  const header = el('div', { class: 'patient-head' }, [
    el('a', { href: '#/', class: 'backlink', text: '← All patients' }),
    el('h1', {}, [`Patient ${trace.subject_id}`, ' ', trainingBadge(trace.seen_in_training)]),
    el('p', { class: 'patient-head__facts', text:
      `${ageSex(patient.age_years, patient.sex)} · ${visit.admission} · ` +
      `${duration(visit.end_hours - visit.start_hours)} stay · ${visit.n_events.toLocaleString()} chart entries` }),
  ]);

  // ---- "now" bar
  const nowTime = el('span', { class: 'now__time' });
  const nowSub = el('span', { class: 'now__sub' });
  const range = el('input', {
    type: 'range', min: 0, max: times.length - 1, step: 1, value: index,
    'aria-label': 'Time in the admission. Use the arrow keys to step; hold Shift for bigger steps.',
  });
  const playBtn = el('button', { class: 'btn play-btn', type: 'button', text: '▶ Play' });
  const nowBar = el('div', { class: 'now' }, [
    playBtn,
    el('div', { class: 'now__label' }, [el('span', { class: 'now__caption', text: 'Now' }), nowTime, nowSub]),
    range,
    el('span', { class: 'now__end', text: `Discharge ${clock(last)}` }),
  ]);

  // ---- risk tiles
  const tiles = events.map((e) => buildTile(meta, e.name));
  const segmented = el('div', { class: 'segmented', role: 'group', 'aria-label': 'Forecast horizon' });
  const tilesSection = el('section', { class: 'tiles' }, [
    el('div', { class: 'tiles__head' }, [
      el('h2', { text: 'Risk in the next' }),
      segmented,
      el('span', { class: 'info', title: meta.disclaimers.risk, text: 'ⓘ' }),
    ]),
    el('div', { class: 'tile-row' }, tiles.map((c) => c.root)),
  ]);

  // ---- chart
  const chartHost = el('div');
  const legend = el('div', { class: 'legend', role: 'group', 'aria-label': 'Show or hide events' });
  const overlaySelect = trace.banked?.length
    ? el('select', { class: 'select--small', 'aria-label': 'Compare with the tuned GBM' }, [
        el('option', { value: '', text: 'Compare with the tuned GBM…' }),
        ...events.map((e) => el('option', { value: e.name, text: `GBM: ${e.display}` })),
      ])
    : null;
  const chartTitle = el('h2');
  const chartCard = card(chartTitle, chartHost, {
    sub: 'Dashed lines are the alert lines. What comes after "now" is faded.',
    actions: el('span', { class: 'legend-wrap' }, [legend, overlaySelect]),
  });

  // ---- what happened
  const alerts = [...trace.alerts].sort((a, b) => {
    const ao = a.onset_hours ?? Infinity;
    const bo = b.onset_hours ?? Infinity;
    return ao - bo || (b.lead_hours ?? -1) - (a.lead_hours ?? -1);
  });
  const liveTags = new Map();
  const storyRows = alerts.map((a) => {
    const story = alertStory(a);
    const live = el('span', { class: 'pill pill--alert', text: 'Alert on now', hidden: true });
    liveTags.set(a.event, live);
    return el('li', { class: `story story--${story.kind}` }, [
      el('span', { class: 'story__event' }, [
        el('span', { class: 'dot', style: { background: eventColor(a.event) } }),
        eventInfo(meta, a.event).display,
      ]),
      el('span', { class: 'story__text', text: story.text }),
      live,
    ]);
  });
  const reliability = alerts.filter((a) => a.detail).map((a) => el('li', {}, [
    el('strong', { text: `${eventInfo(meta, a.event).display}. ` }), a.detail,
  ]));
  const storyCard = card('What happened in this stay', [
    storyRows.length
      ? el('ul', { class: 'stories' }, storyRows)
      : el('p', { class: 'muted', text: 'No alert line was crossed during this admission.' }),
    reliability.length
      ? el('details', { class: 'more' }, [
          el('summary', { text: 'How reliable are these alert lines?' }),
          el('ul', { class: 'notes' }, reliability),
        ])
      : null,
  ], { sub: `Alert lines are set on the ${ALERT_HORIZON}-hour risk` });

  // ---- what the model thinks is going on
  const conceptList = el('ul', { class: 'belief-list', 'aria-live': 'polite' });
  const conceptHost = el('div');
  const conceptCard = card('What the model thinks is going on', [
    conceptList,
    el('details', { class: 'more' }, [
      el('summary', { text: `All ${meta.concepts.length} conditions over the stay` }),
      el('p', { class: 'muted', text: 'One row per condition, time left to right. Darker means the model believes it more.' }),
      conceptHost,
    ]),
  ], { sub: meta.disclaimers.concepts });

  // ---- ask the model
  const tabBody = el('div');
  const tabWhatIf = el('button', { type: 'button', role: 'tab', 'aria-selected': 'true', text: 'What if a value were different?' });
  const tabWhy = el('button', { type: 'button', role: 'tab', 'aria-selected': 'false', text: 'What is driving this forecast?' });
  const askCard = el('section', { class: 'card' }, [
    el('h2', { class: 'card__title', text: 'Ask the model' }),
    el('div', { class: 'tabs', role: 'tablist' }, [tabWhatIf, tabWhy]),
    tabBody,
  ]);

  // ---- sidebar
  const recentList = el('ul', { class: 'chart-list' });
  const nextList = el('ul', { class: 'side-list' });
  const side = el('aside', { class: 'replay__side' }, [
    card('Recently charted', recentList, { sub: `Last ${RECENT_HOURS} h` }),
    el('details', { class: 'card more' }, [
      el('summary', { text: 'What the model expects to be charted next' }),
      nextList,
    ]),
  ]);

  const main = el('div', { class: 'replay__main' }, [chartCard, storyCard, conceptCard, askCard]);
  root.replaceChildren(header, nowBar, tilesSection, el('div', { class: 'replay' }, [main, side]));

  // ---- charts
  const chart = createRiskChart(chartHost, { onScrub: (i) => setIndex(i) });
  const strip = createConceptStrip(conceptHost, {
    concepts: meta.concepts,
    onScrub: (i) => setIndex(i),
  });
  cleanups.push(() => chart.destroy(), () => strip.destroy());
  strip.update({ times, values: trace.concepts });

  function opFor(event) {
    return operatingPoint(meta, event, horizon) ?? operatingPoint(meta, event, ALERT_HORIZON);
  }

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
      overlay: overlayEvent && horizon === ALERT_HORIZON
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
      type: 'button', 'aria-pressed': String(h === horizon), text: `${h} hours`,
      onClick: () => {
        horizon = h;
        renderSegmented();
        chart.update(chartData());
        updateTiles();
      },
    })));
    chartTitle.textContent = `How the ${horizon}-hour risk moved over the stay`;
  }

  function updateTiles() {
    const now = times[index];
    const before = indexAtOrBefore(times, now - TREND_HOURS);
    const key = horizonKey(horizon);
    for (const c of tiles) {
      const series = trace.risk[c.event];
      const onset = trace.onsets[c.event];
      const op = opFor(c.event);
      const p = series[key]?.[index];
      if (onset != null && now >= onset) {
        c.root.dataset.level = 'done';
        c.nodes.value.textContent = 'Happened';
        c.nodes.status.textContent = clock(onset);
        c.nodes.context.textContent = '';
        continue;
      }
      const lvl = level(p, op?.threshold);
      c.root.dataset.level = lvl;
      c.nodes.value.textContent = pct(p);
      const tr = trend(p, series[key]?.[before], TREND_HOURS);
      c.nodes.status.replaceChildren(
        el('span', { class: `status status--${lvl}`, text: STATUS[lvl] }),
        tr.arrow ? el('span', { class: `trend trend--${tr.dir > 0 ? 'up' : tr.dir < 0 ? 'down' : 'flat'}`, text: tr.arrow, title: tr.label, 'aria-label': tr.label }) : null,
      );
      c.nodes.context.textContent = timesTypical(p, op?.base_rate) ?? '';
    }
  }

  // An alert is "on" exactly when the 24 h risk at this moment is at or
  // above its line (risk is null once the event has happened).
  function updateStories() {
    for (const a of alerts) {
      const p = trace.risk[a.event]?.[horizonKey(ALERT_HORIZON)]?.[index];
      liveTags.get(a.event).hidden = !(p != null && a.threshold != null && p >= a.threshold);
    }
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
    }) : [el('li', { class: 'muted', text: 'Nothing to forecast here.' })]));
    const recent = recentEntries(timeline, times[index]);
    recentList.replaceChildren(...(recent.length ? recent.map((e) => el('li', { class: `entry entry--${e.category}` }, [
      el('span', { class: 'entry__t', text: clock(e.t) }),
      el('span', { class: 'entry__body' }, [
        el('span', { class: `cat-dot cat-dot--${e.category}`, title: e.category }),
        el('span', { class: 'entry__label', text: e.label }),
        e.value ? el('span', { class: 'entry__value', text: e.value }) : null,
        e.flag ? el('span', { class: `pill pill--${e.flag}`, text: e.flag.toLowerCase() }) : null,
      ]),
    ])) : [el('li', { class: 'muted', text: 'Nothing charted in this window.' })]));
  }

  function updateConcepts() {
    const row = trace.concepts[index] ?? [];
    const top = meta.concepts
      .map((c, i) => ({ c, v: row[i] ?? 0 }))
      .filter((x) => x.v >= CONCEPT_SHOW)
      .sort((a, b) => b.v - a.v)
      .slice(0, CONCEPT_MAX);
    conceptList.replaceChildren(...(top.length
      ? top.map((x) => {
          const fill = el('div', { class: 'bar__fill' });
          fill.style.setProperty('width', `${Math.round(x.v * 100)}%`);
          return el('li', { class: 'belief', title: x.c.description }, [
            el('span', { class: 'belief__label', text: x.c.display }),
            el('div', { class: 'bar' }, [fill]),
            el('span', { class: 'belief__p', text: pct(x.v) }),
          ]);
        })
      : [el('li', { class: 'muted', text: 'Nothing stands out at this moment.' })]));
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
    nowTime.textContent = clock(times[index]);
    nowSub.textContent = `hour ${Math.round(times[index])} of ${Math.round(last)}`;
    chart.setCursor(index);
    strip.setCursor(index);
    updateTiles();
    updateStories();
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
    for (const c of tiles) c.root.style.setProperty('--tile-color', eventColor(c.event));
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
