/* app.js — UI logic for AI NIDS (XGBoost only)
   - Realtime start/stop
   - PCAP analysis
   - Status polling (no progress bars)
   - Charts render only after a session completes, using the latest alerts file
*/

/* ===================== Helpers ===================== */
const $ = (id) => document.getElementById(id);
const fmtPct = (v, digits = 2) => `${Number(v).toFixed(digits)}%`; // backend alert_rate is already percent
const fmtSec = (s) => `${Number(s).toFixed(1)}s`;
const fmtTime = (ts) => {
  try {
    const d = new Date(Number(ts) * 1000);
    return isNaN(d.getTime()) ? '-' : d.toLocaleString();
  } catch { return '-'; }
};
async function jsonGet(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} -> ${res.status}`);
  return await res.json();
}

/* ===================== State ===================== */
let statusTimer = null;
let activeTab = 'realtime';      // 'realtime' | 'pcap'
let lastStatus = null;           // stores the last polled status to detect transitions
let latestChartsFromFile = null; // remember which file charts were built from
let freezeAfterCompletion = false; // freeze tiles once we finalize a session

/* ===================== Status ===================== */
async function getStatus() {
  try {
    const res = await fetch('/api/status');
    return await res.json();
  } catch (e) {
    console.error('status error', e);
    return null;
  }
}

function setStatusPill(status) {
  const pill = $('statusPill');
  if (!pill) return;
  pill.classList.remove('status-idle','status-running','status-completed','status-error');

  const st = status?.status || 'idle';
  if (st === 'running' || st === 'analyzing_pcap') {
    pill.textContent = st === 'running' ? 'Running' : 'Analyzing';
    pill.classList.add('status-running');
  } else if (st === 'completed' || st === 'stopped') {
    pill.textContent = 'Completed';
    pill.classList.add('status-completed');
  } else if (st === 'error') {
    pill.textContent = 'Error';
    pill.classList.add('status-error');
  } else {
    pill.textContent = 'Idle';
    pill.classList.add('status-idle');
  }
}

// prefer finalized counts from root status if present
function preferFinalizedCounts(st, perf) {
  if (!st || !perf) return perf;
  const out = { ...perf };
  if (typeof st.alert_count === 'number' && st.alert_count !== perf.total_alerts) {
    out.total_alerts = st.alert_count;
  }
  if (typeof st.total_packets === 'number' && st.total_packets !== perf.total_packets_analyzed) {
    out.total_packets_analyzed = st.total_packets;
  }
  return out;
}

function updateRealtimeTiles(perf) {
  const packetsTotal   = perf?.total_packets_analyzed ?? 0;
  const tcpUdpTotal    = perf?.total_tcp_udp_packets ?? 0;
  const alertsTotal    = perf?.total_alerts ?? 0;
  const alertRatePct   = perf?.alert_rate != null ? perf.alert_rate : 0; // already percent
  const procTime       = perf?.processing_time != null ? perf.processing_time : 0;

  $('rtPacketsTotal') && ($('rtPacketsTotal').textContent  = packetsTotal);
  $('rtPacketsTcpUdp') && ($('rtPacketsTcpUdp').textContent = tcpUdpTotal);
  $('rtAlertsTotal') && ($('rtAlertsTotal').textContent   = alertsTotal);
  $('rtAlertRate') && ($('rtAlertRate').textContent     = fmtPct(alertRatePct, 2));
  $('rtProcessingTime') && ($('rtProcessingTime').textContent= fmtSec(procTime));

  ensureRateCaption('rtAlertRate', alertsTotal, tcpUdpTotal);
}

function updatePcapTiles(perf) {
  const packetsTotal   = perf?.total_packets_analyzed ?? 0;
  const tcpUdpTotal    = perf?.total_tcp_udp_packets ?? 0;
  const alertsTotal    = perf?.total_alerts ?? 0;
  const alertRatePct   = perf?.alert_rate != null ? perf.alert_rate : 0; // already percent
  const procTime       = perf?.processing_time != null ? perf.processing_time : 0;

  $('pcPacketsTotal') && ($('pcPacketsTotal').textContent  = packetsTotal);
  $('pcPacketsTcpUdp') && ($('pcPacketsTcpUdp').textContent = tcpUdpTotal);
  $('pcAlertsTotal') && ($('pcAlertsTotal').textContent   = alertsTotal);
  $('pcAlertRate') && ($('pcAlertRate').textContent     = fmtPct(alertRatePct, 2));
  $('pcProcessingTime') && ($('pcProcessingTime').textContent= fmtSec(procTime));

  ensureRateCaption('pcAlertRate', alertsTotal, tcpUdpTotal);
}

// Caption under alert rate tile
function ensureRateCaption(rateId, alerts, tcpUdp) {
  const rateEl = $(rateId);
  if (!rateEl) return;
  const parent = rateEl.parentElement;
  if (!parent) return;

  let cap = parent.querySelector('.rate-caption');
  const denom = (typeof tcpUdp === 'number' && tcpUdp >= 0) ? tcpUdp : 0;
  const text = `(${alerts} / ${denom})`;
  if (!cap) {
    cap = document.createElement('div');
    cap.className = 'rate-caption small text-muted';
    cap.style.marginTop = '2px';
    cap.textContent = text;
    parent.appendChild(cap);
  } else {
    cap.textContent = text;
  }
}

/* ===================== Charts (after completion only) ===================== */
function barsHtmlFromCounts(obj, topN = 6) {
  const entries = Object.entries(obj || {});
  if (entries.length === 0) return `<div class="small-muted">No data</div>`;
  // FIX: sort by value (index 1)
  const sorted = entries.sort((a, b) => b[14] - a[14]).slice(0, topN);
  const max = Math.max(...sorted.map(x => x[14])) || 1;
  return sorted.map(([label, val], idx) => {
    const pct = Math.max(5, (val / max) * 100); // min visible width
    const colorClass = `c${(idx % 6) + 1}`;
    return `
      <div class="bar-row">
        <div class="bar-label">${label}</div>
        <div class="bar-track"><div class="bar-fill ${colorClass}" style="width:${pct}%"></div></div>
        <div class="bar-value">${val}</div>
      </div>
    `;
  }).join('');
}

function computeCountsFromRows(rows) {
  const ports = {};
  const protos = {};
  const confBuckets = {
    '0–10%': 0, '10–20%': 0, '20–30%': 0, '30–40%': 0, '40–50%': 0, '50%+': 0
  };
  for (const r of rows || []) {
    const p = String(r.destination_port ?? 'Unknown');
    ports[p] = (ports[p] || 0) + 1;

    const proto = r.protocol || 'Unknown';
    protos[proto] = (protos[proto] || 0) + 1;

    const c = Number(r.confidence || 0);
    const pc = c * 100;
    if (pc < 10) confBuckets['0–10%']++;
    else if (pc < 20) confBuckets['10–20%']++;
    else if (pc < 30) confBuckets['20–30%']++;
    else if (pc < 40) confBuckets['30–40%']++;
    else if (pc < 50) confBuckets['40–50%']++;
    else confBuckets['50%+']++;
  }
  return { ports, protos, confBuckets };
}

function renderChartsForRows(rows) {
  const { ports, protos, confBuckets } = computeCountsFromRows(rows);
  if (activeTab === 'realtime') {
    $('rtTopPorts') && ($('rtTopPorts').innerHTML = barsHtmlFromCounts(ports));
    $('rtProtocols') && ($('rtProtocols').innerHTML = barsHtmlFromCounts(protos));
    $('rtConfidence') && ($('rtConfidence').innerHTML = barsHtmlFromCounts(confBuckets));
  } else {
    $('pcTopPorts') && ($('pcTopPorts').innerHTML = barsHtmlFromCounts(ports));
    $('pcProtocols') && ($('pcProtocols').innerHTML = barsHtmlFromCounts(protos));
    $('pcConfidence') && ($('pcConfidence').innerHTML = barsHtmlFromCounts(confBuckets));
  }
}

async function loadLatestFileRowsForCharts(maxRows = 200000) {
  // Prefer name from status if available
  let newest = null;
  try {
    const st = lastStatus || await getStatus();
    newest = st?.latest_alerts_file || null;
  } catch {}

  if (!newest) {
    // Fallback to files endpoint: pick newest file at index 0
    let filesResp;
    try {
      filesResp = await jsonGet('/api/alerts/files');
    } catch (e) {
      console.error('files list error', e);
      renderChartsForRows([]);
      return;
    }
    const files = Array.isArray(filesResp?.files) ? filesResp.files : [];
    if (!files.length || !files?.name) {
      renderChartsForRows([]);
      return;
    }
    newest = files.name;
  }

  // Avoid redundant rebuild if already from this file
  if (latestChartsFromFile === newest) return;
  latestChartsFromFile = newest;

  try {
    const page = await jsonGet(`/api/alerts/file?name=${encodeURIComponent(newest)}&offset=0&limit=${maxRows}`);
    const rows = Array.isArray(page?.rows) ? page.rows : [];
    renderChartsForRows(rows);
  } catch (e) {
    console.error('charts load error', e);
    latestChartsFromFile = null; // allow retry
  }
}

function showCompletionToast(text = 'Session complete — charts updated from latest run.') {
  let host = document.getElementById('toastHost');
  if (!host) {
    host = document.createElement('div');
    host.id = 'toastHost';
    host.style.position = 'fixed';
    host.style.bottom = '16px';
    host.style.right = '16px';
    host.style.zIndex = '2000';
    document.body.appendChild(host);
  }
  const el = document.createElement('div');
  el.className = 'alert alert-primary shadow-sm mb-2';
  el.textContent = text;
  host.appendChild(el);
  setTimeout(() => { el.remove(); }, 3000);
}

/* ===================== Status polling with completion trigger ===================== */
async function refreshStatus() {
  if (freezeAfterCompletion) return;

  const st = await getStatus();
  if (!st) return;

  if (!lastStatus) {
    lastStatus = st;
    setStatusPill(st);
    const basePerf = st.current_performance || st.performance_metrics || {};
    const perfInit = preferFinalizedCounts(st, basePerf);
    if (activeTab === 'realtime') updateRealtimeTiles(perfInit);
    else updatePcapTiles(perfInit);

    // NEW: when landing on a completed/stopped/idle state, load newest file on first paint
    const doneLike = (st.status === 'completed' || st.status === 'stopped' || st.status === 'idle');
    if (doneLike) {
      try {
        await loadLatestFileRowsForCharts(200000);
      } catch (e) {
        console.error('initial charts load error', e);
      }
      // small retry to survive file listing/writes
      setTimeout(() => {
        loadLatestFileRowsForCharts(200000).catch(()=>{});
      }, 600);
    }
    return;
  }

  setStatusPill(st);

  $('startBtn') && ($('startBtn').disabled   = !!st.is_monitoring);
  $('stopBtn') && ($('stopBtn').disabled    = !st.is_monitoring);
  $('analyzeBtn') && ($('analyzeBtn').disabled = !!st.is_monitoring);

  const basePerf = st.current_performance || st.performance_metrics || {};
  const perf = preferFinalizedCounts(st, basePerf);

  if (activeTab === 'realtime') updateRealtimeTiles(perf);
  else updatePcapTiles(perf);

  const prev = lastStatus.status;
  const now = st.status;
  const finished = (prev === 'running' || prev === 'analyzing_pcap') && (now === 'completed' || now === 'stopped' || now === 'error');

  if (finished) {
    try {
      await loadLatestFileRowsForCharts(200000);
      showCompletionToast();
    } catch (e) {
      console.error('charts load error', e);
    }
    setTimeout(() => {
      loadLatestFileRowsForCharts(200000).catch(()=>{});
    }, 600);

    freezeAfterCompletion = true;
    setTimeout(() => {
      stopStatusTimer();
    }, 1500);
  }

  lastStatus = st;
}

/* ===================== Actions ===================== */
async function startRealtime() {
  const iface = $('ifaceInput').value.trim() || 'wlan0';
  const duration = Number($('durationInput').value || 60);
  const threshold = $('thresholdInput').value;
  const body = { interface: iface, duration, model_type: 'xgboost' };
  if (threshold !== '') body.threshold = Number(threshold);

  try {
    const res = await fetch('/api/start_detection', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    const out = await res.json();
    if (out.status !== 'success') console.warn('start failed', out);

    latestChartsFromFile = null;
    lastStatus = null;
    freezeAfterCompletion = false;
    startStatusTimer();
  } catch (e) { console.error('start', e); }
}

async function stopRealtime() {
  try {
    await fetch('/api/stop_detection', { method: 'POST' });
  } catch(e){ console.error('stop', e); }
  await refreshStatus();
}

async function analyzePcap() {
  const path = $('pcapSelect').value;
  const maxPackets = Number($('maxPacketsInput').value || 5000);
  const threshold = $('thresholdInput').value;
  if (!path) return;

  const body = { pcap_path: path, max_packets: maxPackets, model_type: 'xgboost' };
  if (threshold !== '') body.threshold = Number(threshold);

  try {
    const res = await fetch('/api/analyze_pcap', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    const out = await res.json();
    if (out.status !== 'success') console.warn('pcap analyze failed', out);

    latestChartsFromFile = null;
    lastStatus = null;
    freezeAfterCompletion = false;
    startStatusTimer();
  } catch (e) { console.error('pcap', e); }
}

/* ===================== Lists & Health ===================== */
async function loadPcapList() {
  try {
    const res = await fetch('/api/pcap_list');
    const files = await res.json();
    const sel = $('pcapSelect');
    sel.innerHTML = '';

    const working = files.filter(f => /WorkingHours\.pcap$/i.test(f.name));
    const items = working.length ? working : files;

    if (!items.length) {
      sel.innerHTML = `<option value="">No PCAP files</option>`;
      return;
    }
    items.forEach(f => {
      const opt = document.createElement('option');
      opt.value = f.path;
      opt.textContent = `${f.name} (${(f.size_mb || 0)} MB)`;
      sel.appendChild(opt);
    });
  } catch (e) {
    console.error('pcap list', e);
    $('pcapSelect').innerHTML = `<option value="">Error loading list</option>`;
  }
}

async function openHealthModal() {
  try {
    const res = await fetch('/api/system_health');
    const health = await res.json();
    $('healthBody').textContent = JSON.stringify(health, null, 2);
  } catch (e) {
    $('healthBody').textContent = 'Error loading health';
  }
  const modal = new bootstrap.Modal($('healthModal'));
  modal.show();
}

/* ===================== Timers ===================== */
function startStatusTimer() {
  if (statusTimer) clearInterval(statusTimer);
  statusTimer = setInterval(async () => {
    await refreshStatus();
  }, 1500);
}
function stopStatusTimer() {
  if (statusTimer) clearInterval(statusTimer);
  statusTimer = null;
}

/* ===================== Events & Init ===================== */
function bindEvents() {
  // Tabs
  const rtTab = document.querySelector('#realtime-tab');
  const pcTab = document.querySelector('#pcap-tab');
  rtTab && rtTab.addEventListener('click', () => { activeTab = 'realtime'; });
  pcTab && pcTab.addEventListener('click', () => { activeTab = 'pcap'; });

  // Buttons
  $('startBtn') && $('startBtn').addEventListener('click', startRealtime);
  $('stopBtn') && $('stopBtn').addEventListener('click', stopRealtime);
  $('analyzeBtn') && $('analyzeBtn').addEventListener('click', analyzePcap);

  // Health
  $('healthBtn') && $('healthBtn').addEventListener('click', openHealthModal);
  $('refreshHealthBtn') && $('refreshHealthBtn').addEventListener('click', openHealthModal);
}

async function init() {
  bindEvents();
  await loadPcapList();
  await refreshStatus();
  startStatusTimer();
}

document.addEventListener('DOMContentLoaded', init);
