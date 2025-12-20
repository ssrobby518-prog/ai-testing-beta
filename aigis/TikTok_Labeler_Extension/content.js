// Aigis TikTok Labeler - 第一性原理重寫版
// 核心：最小複雜度 + 最大可見性

console.log('[Aigis] ✅ Extension loaded');

// === 配置 ===
const API_URL = 'http://127.0.0.1:5000/api/label';
const COMMAND_URL = 'http://127.0.0.1:5000/api/command';

const KEY_MAP = {
  'ArrowLeft': {label: 'real', text: 'REAL', color: '#00ff00'},
  'ArrowRight': {label: 'ai', text: 'AI', color: '#ff0000'},
  'ArrowUp': {label: 'uncertain', text: 'UNCERTAIN', color: '#ffaa00'},
  'ArrowDown': {label: 'exclude', text: 'MOVIE/ANIME', color: '#888888'},
  'KeyQ': {label: 'ai', reason: 'motion_jitter', text: 'AI: MOTION', color: '#ff0000'},
  'KeyW': {label: 'ai', reason: 'lighting_error', text: 'AI: LIGHT', color: '#ff0000'},
  'KeyE': {label: 'ai', reason: 'artifacts', text: 'AI: PIXEL', color: '#ff0000'},
  'KeyR': {label: 'ai', reason: 'lipsync_fail', text: 'AI: LIPSYNC', color: '#ff0000'},
  'KeyS': {label: null, text: 'SKIP', color: '#cccccc'}
};

let sessionStats = {
  total: 0,
  real: 0,
  ai: 0,
  uncertain: 0,
  exclude: 0,
  skip: 0
};

let statsPanel = null;
let commandOverlay = null;

// === 第一性原理：全局監聽，無過濾 ===
document.addEventListener('keydown', handleKeyPress, true);

function handleKeyPress(e) {
  console.log('[Aigis] Key pressed:', e.code, e.key);

  if (commandOverlay) {
    if (e.code === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      closeCommandModal();
    }
    return;
  }

  if (e.code === 'Escape') {
    e.preventDefault();
    e.stopPropagation();
    openCommandModal();
    return;
  }

  const mapping = KEY_MAP[e.code];
  if (!mapping) {
    console.log('[Aigis] Key not mapped, ignoring');
    return;
  }

  // 阻止默認行為
  e.preventDefault();
  e.stopPropagation();

  console.log('[Aigis] Handling:', mapping.text);

  // === 沙皇炸彈：立即反饋（不等API）===
  showFeedback(mapping.text, mapping.color);

  // === 數據傳輸（如果不是SKIP）===
  if (mapping.label !== null) {
    sendLabel(mapping.label, mapping.reason);
  } else {
    console.log('[Aigis] SKIP - 不發送數據');
  }

  updateSessionStats(mapping);

  // === 自動滾動 ===
  setTimeout(() => {
    console.log('[Aigis] Auto scrolling...');
    window.scrollBy(0, window.innerHeight);
  }, 100);
}

function showFeedback(text, color) {
  console.log('[Aigis] Showing feedback:', text);

  // 移除舊的overlay
  const old = document.querySelector('.aigis-feedback');
  if (old) old.remove();

  // 創建新overlay
  const overlay = document.createElement('div');
  overlay.className = 'aigis-feedback';
  overlay.textContent = text;
  overlay.style.cssText = `
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 999999;
    font-size: 6rem;
    font-weight: bold;
    color: ${color};
    text-shadow: 0 0 20px #000, 0 0 40px #000;
    pointer-events: none;
    animation: aigis-flash 0.8s ease-out;
  `;

  document.body.appendChild(overlay);

  setTimeout(() => overlay.remove(), 800);
}

async function sendLabel(label, reason = null) {
  const url = normalizeTikTokUrl();
  const videoId = extractVideoId(url);

  const payload = {
    timestamp: new Date().toISOString(),
    video_url: url,
    author_id: extractAuthorId(),
    video_id: videoId,
    label: label,
    reason: reason,
    source_version: 'aigis_v1'
  };

  console.log('[Aigis] Sending to API:', payload);

  try {
    // === 🔥 新增：同時捕獲視頻blob ===
    const videoBlob = await captureVideoBlob();

    if (videoBlob) {
      console.log('[Aigis] 📹 Video captured:', (videoBlob.size / 1024 / 1024).toFixed(2), 'MB');

      // 發送帶視頻的表單數據
      const formData = new FormData();
      formData.append('data', JSON.stringify(payload));
      formData.append('video', videoBlob, `${videoId}.mp4`);

      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
        mode: 'cors'
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const result = await response.json();
      console.log('[Aigis] ✅ API Response (with video):', result);

      if (result.total_count) {
        showMiniNotification(`✅ 已標註+下載: ${result.total_count}`);
      }
    } else {
      // fallback：沒有視頻時只發送標籤
      console.log('[Aigis] ⚠️ No video found, sending label only');

      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
        mode: 'cors'
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const result = await response.json();
      console.log('[Aigis] ✅ API Response:', result);

      if (result.total_count) {
        showMiniNotification(`已標註: ${result.total_count}`);
      }
    }

  } catch (err) {
    console.error('[Aigis] ❌ API Error:', err);
    showMiniNotification('⚠️ 伺服器離線');
  }
}

async function captureVideoBlob() {
  try {
    // 查找頁面上的video元素
    const videoElement = document.querySelector('video');
    if (!videoElement) {
      console.log('[Aigis] No video element found');
      return null;
    }

    const videoSrc = videoElement.currentSrc || videoElement.src;
    if (!videoSrc) {
      console.log('[Aigis] No video source found');
      return null;
    }

    console.log('[Aigis] Downloading video from:', videoSrc.substring(0, 100) + '...');

    // 直接從視頻URL下載blob
    const response = await fetch(videoSrc);
    if (!response.ok) {
      console.error('[Aigis] Video fetch failed:', response.status);
      return null;
    }

    const blob = await response.blob();
    console.log('[Aigis] Video blob created:', blob.size, 'bytes');

    return blob;

  } catch (err) {
    console.error('[Aigis] ❌ Video capture error:', err);
    return null;
  }
}

function showMiniNotification(text) {
  const notif = document.createElement('div');
  notif.textContent = text;
  notif.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: rgba(0,0,0,0.8);
    color: white;
    padding: 10px 20px;
    border-radius: 5px;
    z-index: 999999;
    font-size: 14px;
  `;

  document.body.appendChild(notif);
  setTimeout(() => notif.remove(), 2000);
}

function normalizeTikTokUrl() {
  let url = window.location.href;

  const canonical = document.querySelector('link[rel="canonical"]');
  if (canonical && canonical.href) {
    url = canonical.href;
  } else {
    const og = document.querySelector('meta[property="og:url"]');
    if (og && og.content) {
      url = og.content;
    }
  }

  try {
    const u = new URL(url);
    u.hash = '';
    u.search = '';
    url = u.toString();
  } catch (_) {
  }

  return url;
}

function extractVideoId(url) {
  try {
    const match = String(url).match(/\/video\/(\d+)/);
    return match ? match[1] : '';
  } catch {
    return '';
  }
}

function extractAuthorId() {
  try {
    const match = window.location.pathname.match(/@([^/]+)/);
    return match ? match[1] : 'unknown';
  } catch {
    return 'unknown';
  }
}

function ensureStatsPanel() {
  if (statsPanel) return;

  statsPanel = document.createElement('div');
  statsPanel.className = 'aigis-stats-panel';
  statsPanel.innerHTML = `
    <div class="aigis-stats-title">Aigis</div>
    <div class="aigis-stats-row">Total: <span data-k="total">0</span></div>
    <div class="aigis-stats-row">Real: <span data-k="real">0</span></div>
    <div class="aigis-stats-row">AI: <span data-k="ai">0</span></div>
    <div class="aigis-stats-row">Uncertain: <span data-k="uncertain">0</span></div>
    <div class="aigis-stats-row">Exclude: <span data-k="exclude">0</span></div>
    <div class="aigis-stats-row">Skip: <span data-k="skip">0</span></div>
    <div class="aigis-stats-hint">Esc: 指令｜S: Skip</div>
  `;
  document.body.appendChild(statsPanel);
}

function updateStatsPanel() {
  ensureStatsPanel();
  for (const k of Object.keys(sessionStats)) {
    const el = statsPanel.querySelector(`[data-k="${k}"]`);
    if (el) el.textContent = String(sessionStats[k]);
  }
}

function updateSessionStats(mapping) {
  sessionStats.total += 1;
  if (mapping.label === 'real') sessionStats.real += 1;
  else if (mapping.label === 'ai') sessionStats.ai += 1;
  else if (mapping.label === 'uncertain') sessionStats.uncertain += 1;
  else if (mapping.label === 'exclude') sessionStats.exclude += 1;
  else sessionStats.skip += 1;
  updateStatsPanel();
}

function openCommandModal() {
  if (commandOverlay) return;

  const overlay = document.createElement('div');
  overlay.className = 'aigis-command-overlay';
  overlay.innerHTML = `
    <div class="aigis-command-modal" role="dialog" aria-modal="true">
      <div class="aigis-command-title">Aigis 指令</div>
      <input class="aigis-command-input" type="text" placeholder="輸入：第一層下載" />
      <div class="aigis-command-actions">
        <button class="aigis-command-cancel" type="button">取消</button>
        <button class="aigis-command-submit" type="button">送出</button>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);
  commandOverlay = overlay;

  const input = overlay.querySelector('.aigis-command-input');
  const cancelBtn = overlay.querySelector('.aigis-command-cancel');
  const submitBtn = overlay.querySelector('.aigis-command-submit');

  cancelBtn.addEventListener('click', closeCommandModal);
  overlay.addEventListener('click', (ev) => {
    if (ev.target === overlay) closeCommandModal();
  });

  input.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') {
      ev.preventDefault();
      submitCommand(String(input.value || '').trim());
    } else if (ev.key === 'Escape') {
      ev.preventDefault();
      closeCommandModal();
    }
  });

  submitBtn.addEventListener('click', () => submitCommand(String(input.value || '').trim()));

  input.focus();
}

function closeCommandModal() {
  if (!commandOverlay) return;
  commandOverlay.remove();
  commandOverlay = null;
}

async function submitCommand(command) {
  if (!command) {
    closeCommandModal();
    return;
  }

  try {
    const response = await fetch(COMMAND_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command })
    });

    if (!response.ok) {
      const text = await response.text().catch(() => '');
      showMiniNotification(`⚠️ 指令失敗 (${response.status})`);
      console.log('[Aigis] Command error:', response.status, text);
      closeCommandModal();
      return;
    }

    const result = await response.json().catch(() => ({}));
    if (result && result.status === 'started') {
      showMiniNotification('✅ 已開始執行：第一層下載');
    } else if (result && result.status === 'ignored') {
      showMiniNotification('⚠️ 未識別的指令');
    } else {
      showMiniNotification('⚠️ 指令已送出');
    }
  } catch (err) {
    console.error('[Aigis] ❌ Command API Error:', err);
    showMiniNotification('⚠️ 指令送出失敗');
  } finally {
    closeCommandModal();
  }
}

// === 添加CSS動畫 ===
const style = document.createElement('style');
style.textContent = `
  @keyframes aigis-flash {
    0% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
    100% { opacity: 0; transform: translate(-50%, -50%) scale(1.5); }
  }
`;
document.head.appendChild(style);

setTimeout(() => {
  updateStatsPanel();
}, 0);

console.log('[Aigis] 🎯 Ready! Press ← or → to label');
