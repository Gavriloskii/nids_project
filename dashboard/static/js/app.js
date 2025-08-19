/* app.js — Minimal UI logic for AI NIDS (XGBoost only)
   - Realtime start/stop
   - PCAP analysis
   - Status/progress polling
   - Alerts list with filters + clickable rows
   - Alert details drawer
   - Simple CSS bar “charts” for ports/protocols/confidence
*/

/* ===================== Helpers ===================== */
const $ = (id) => document.getElementById(id);
const fmtPct = (v, digits = 2) => `${Number(v).toFixed(digits)}%`; // backend already provides percent
const fmtSec = (s) => `${Number(s).toFixed(1)}s`;
const fmtTime = (ts) => {
  try {
    const d = new Date(Number(ts) * 1000);
    return isNaN(d.getTime()) ? '-' : d.toLocaleString();
  } catch { return '-'; }
};

/* ===================== State ===================== */
let statusTimer = null;
let alertsTimer = null;
let activeTab = 'realtime';      // 'realtime' | 'pcap'
let selectedAlert = null;

/* ===================== Status/Progress ===================== */
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

function updateRealtimeTiles(perf, progress) {
  const packetsTotal   = perf?.total_packets_analyzed ?? 0;
  const tcpUdpTotal    = perf?.total_tcp_udp_packets ?? 0;
  const alertsTotal    = perf?.total_alerts ?? 0;
  const alertRatePct   = perf?.alert_rate != null ? perf.alert_rate : 0; // already in percent
  const procTime       = perf?.processing_time != null ? perf.processing_time : 0;

  $('rtPacketsTotal').textContent  = packetsTotal;
  $('rtPacketsTcpUdp').textContent = tcpUdpTotal;
  $('rtAlertsTotal').textContent   = alertsTotal;
  $('rtAlertRate').textContent     = fmtPct(alertRatePct, 2);
  $('rtProcessingTime').textContent= fmtSec(procTime);

  // Inline caption: (alerts / tcp_udp)
  ensureRateCaption('rtAlertRate', alertsTotal, tcpUdpTotal);

  $('rtProgress').style.width      = `${Math.min(100, Number(progress || 0))}%`;
}

function updatePcapTiles(perf, progress) {
  const packetsTotal   = perf?.total_packets_analyzed ?? 0;
  const tcpUdpTotal    = perf?.total_tcp_udp_packets ?? 0;
  const alertsTotal    = perf?.total_alerts ?? 0;
  const alertRatePct   = perf?.alert_rate != null ? perf.alert_rate : 0; // already in percent
  const procTime       = perf?.processing_time != null ? perf.processing_time : 0;

  $('pcPacketsTotal').textContent  = packetsTotal;
  $('pcPacketsTcpUdp').textContent = tcpUdpTotal;
  $('pcAlertsTotal').textContent   = alertsTotal;
  $('pcAlertRate').textContent     = fmtPct(alertRatePct, 2);
  $('pcProcessingTime').textContent= fmtSec(procTime);

  ensureRateCaption('pcAlertRate', alertsTotal, tcpUdpTotal);

  $('pcProgress').style.width      = `${Math.min(100, Number(progress || 0))}%`;
}

// Helper to add a "(alerts / tcp_udp)" caption under the rate value
function ensureRateCaption(rateId, alerts, tcpUdp) {
  const rateEl = $(rateId);
  if (!rateEl) return;
  const parent = rateEl.parentElement;
  if (!parent) return;

  let cap = parent.querySelector('.rate-caption');
  const text = `(${alerts} / ${tcpUdp || 0})`;
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

async function refreshStatus() {
  const st = await getStatus();
  if (!st) return;

  setStatusPill(st);

  // Buttons
  $('startBtn').disabled   = !!st.is_monitoring;
  $('stopBtn').disabled    = !st.is_monitoring;
  $('analyzeBtn').disabled = !!st.is_monitoring;

  // Prefer current_performance, fallback to performance_metrics
  const perf = st.current_performance || st.performance_metrics || {};
  const progress = st.progress ?? 0;

  if (activeTab === 'realtime') updateRealtimeTiles(perf, progress);
  else updatePcapTiles(perf, progress);

  // If not running and auto-refresh is off, stop alerts timer
  if (!st.is_monitoring && $('autoRefreshToggle').checked === false) {
    stopAlertsTimer();
  }
}

/* ===================== Alerts ===================== */
function alertsRowHtml(a) {
  const conf = Number(a.confidence || 0);
  const ts = a.timestamp ?? 0;
  return `
    <tr data-alert='${JSON.stringify(a).replace(/'/g, '&apos;')}'>
      <td>${fmtTime(ts)}</td>
      <td>${a.protocol || '-'}</td>
      <td>${a.destination_port ?? '-'}</td>
      <td>${fmtPct(conf * 100, 1)}</td>
    </tr>
  `;
}

async function getAlerts() {
  const limit = Number($('limitSelect').value || 100);
  const minC  = Number($('minConfidence').value || 0) / 100; // slider is 0–10 => %
  try {
    const res = await fetch(`/api/alerts?limit=${limit}&min_confidence=${minC}`);
    return await res.json();
  } catch (e) {
    console.error('alerts error', e);
    return [];
  }
}

function attachAlertRowClicks() {
  const tbody = $('alertsBody');
  tbody.querySelectorAll('tr').forEach(tr => {
    tr.addEventListener('click', () => {
      const payload = tr.getAttribute('data-alert');
      if (!payload) return;
      try {
        const alert = JSON.parse(payload.replace(/&apos;/g, "'"));
        openDrawerWithAlert(alert);
      } catch(e) { console.error('row parse', e); }
    });
  });
}

async function refreshAlerts() {
  const alerts = await getAlerts();
  if (!alerts || alerts.length === 0) {
    $('alertsBody').innerHTML = `<tr><td colspan="4" class="text-center py-3">No alerts</td></tr>`;
    return;
  }
  $('alertsBody').innerHTML = alerts.map(alertsRowHtml).join('');
  attachAlertRowClicks();
}

/* ===================== Drawer ===================== */
function openDrawerWithAlert(a) {
  selectedAlert = a || null;
  $('dTime').textContent       = fmtTime(a?.timestamp);
  $('dProtocol').textContent   = a?.protocol ?? '-';
  $('dPort').textContent       = a?.destination_port ?? '-';
  $('dConfidence').textContent = a?.confidence != null ? fmtPct(Number(a.confidence) * 100, 1) : '-';
  $('dThreshold').textContent  = a?.threshold_used != null ? a.threshold_used : '-';
  $('dReason').textContent     = a?.alert_reason ?? '-';
  $('dPacketLen').textContent  = a?.packet_info?.length ?? '-';
  $('dPacketTime').textContent = a?.packet_info?.time ?? '-';
  $('dJson').textContent       = JSON.stringify(a ?? {}, null, 2);

  $('alertDrawer').classList.add('open');
}
function closeDrawer() {
  $('alertDrawer').classList.remove('open');
}

/* ===================== CSS Bars (Charts) ===================== */
function barsHtmlFromCounts(obj, topN = 6) {
  const entries = Object.entries(obj || {});
  if (entries.length === 0) return `<div class="small-muted">No data</div>`;
  const sorted = entries.sort((a,b) => b[1]-a[1]).slice(0, topN);
  const max = Math.max(...sorted.map(x=>x[1])) || 1;
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

async function getStats() {
  try {
    const res = await fetch('/api/stats');
    return await res.json();
  } catch (e) {
    console.error('stats error', e);
    return {};
  }
}

async function refreshCharts() {
  const s = await getStats();
  const ports = s.destination_ports || {};
  const protos = s.alert_types || {};
  const confBuckets = bucketizeConfidence(s);

  if (activeTab === 'realtime') {
    $('rtTopPorts').innerHTML = barsHtmlFromCounts(ports);
    $('rtProtocols').innerHTML = barsHtmlFromCounts(protos);
    $('rtConfidence').innerHTML = barsHtmlFromCounts(confBuckets);
  } else {
    $('pcTopPorts').innerHTML = barsHtmlFromCounts(ports);
    $('pcProtocols').innerHTML = barsHtmlFromCounts(protos);
    $('pcConfidence').innerHTML = barsHtmlFromCounts(confBuckets);
  }
}

function bucketizeConfidence(stats) {
  // Approximate buckets from summary stats; replace with actual histogram if available later.
  const cs = stats?.confidence_stats;
  if (!cs) return {};
  const max = Number(cs.max || 0);
  const mean = Number(cs.mean || 0);
  return {
    '0–10%':  Math.round((0.10 >= mean ? 3 : 1)),
    '10–20%': Math.round((0.20 >= mean ? 2 : 1)),
    '20–30%': Math.round((0.30 >= mean ? 2 : 1)),
    '30–40%': Math.round((0.40 >= mean ? 2 : 1)),
    '40–50%': Math.round((0.50 >= mean ? 2 : 1)),
    '50%+':   Math.max(1, Math.round(max * 10))
  };
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

    startStatusTimer();
    if ($('autoRefreshToggle').checked) startAlertsTimer();
  } catch (e) { console.error('start', e); }
}

async function stopRealtime() {
  try {
    await fetch('/api/stop_detection', { method: 'POST' });
  } catch(e){ console.error('stop', e); }
  await refreshStatus();
  await refreshCharts();
  stopAlertsTimer();
}

async function analyzePcap() {
  const path = $('pcapSelect').value;
  const maxPackets = Number($('maxPacketsInput').value || 5000);
  if (!path) return;

  try {
    const res = await fetch('/api/analyze_pcap', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ pcap_path: path, max_packets: maxPackets, model_type: 'xgboost' })
    });
    const out = await res.json();
    if (out.status !== 'success') console.warn('pcap analyze failed', out);

    startStatusTimer();
    if ($('autoRefreshToggle').checked) startAlertsTimer();
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
  const modal = new bootstrap.Modal($('#healthModal'));
  modal.show();
}

/* ===================== Timers ===================== */
function startStatusTimer() {
  if (statusTimer) clearInterval(statusTimer);
  statusTimer = setInterval(async () => {
    await refreshStatus();
    // Optionally refresh charts here less frequently if desired
  }, 1500);
}
function stopStatusTimer() {
  if (statusTimer) clearInterval(statusTimer);
  statusTimer = null;
}

function startAlertsTimer() {
  if (alertsTimer) clearInterval(alertsTimer);
  alertsTimer = setInterval(refreshAlerts, 2000);
}
function stopAlertsTimer() {
  if (alertsTimer) clearInterval(alertsTimer);
  alertsTimer = null;
}

/* ===================== Events & Init ===================== */
function bindEvents() {
  // Tabs
  const rtTab = document.querySelector('#realtime-tab');
  const pcTab = document.querySelector('#pcap-tab');
  rtTab.addEventListener('click', () => { activeTab = 'realtime'; refreshCharts(); });
  pcTab.addEventListener('click', () => { activeTab = 'pcap'; refreshCharts(); });

  // Buttons
  $('startBtn').addEventListener('click', startRealtime);
  $('stopBtn').addEventListener('click', stopRealtime);
  $('analyzeBtn').addEventListener('click', analyzePcap);

  // Alerts
  $('refreshAlertsBtn').addEventListener('click', refreshAlerts);
  $('autoRefreshToggle').addEventListener('change', (e) => {
    if (e.target.checked) startAlertsTimer();
    else stopAlertsTimer();
  });
  $('minConfidence').addEventListener('input', () => {
    const v = Number($('minConfidence').value || 0);
    $('minConfidenceLabel').textContent = `${v.toFixed(1)}%`;
    refreshAlerts();
  });
  $('limitSelect').addEventListener('change', refreshAlerts);

  // Drawer
  $('drawerClose').addEventListener('click', closeDrawer);
  $('copyJsonBtn').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText($('dJson').textContent || '');
      $('copyJsonBtn').classList.remove('btn-outline-primary');
      $('copyJsonBtn').classList.add('btn-success');
      $('copyJsonBtn').innerHTML = `<i class="fa-solid fa-check me-1"></i> Copied`;
      setTimeout(() => {
        $('copyJsonBtn').classList.remove('btn-success');
        $('copyJsonBtn').classList.add('btn-outline-primary');
        $('copyJsonBtn').innerHTML = `<i class="fa-regular fa-copy me-1"></i> Copy JSON`;
      }, 1000);
    } catch {}
  });

  // Health
  $('healthBtn').addEventListener('click', openHealthModal);
  $('refreshHealthBtn').addEventListener('click', openHealthModal);
}

async function init() {
  bindEvents();
  await loadPcapList();
  await refreshStatus();
  await refreshAlerts();
  await refreshCharts();

  // baseline timers
  startStatusTimer();
  if ($('autoRefreshToggle').checked) startAlertsTimer();
}

document.addEventListener('DOMContentLoaded', init);
