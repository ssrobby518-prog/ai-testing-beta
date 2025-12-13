// Aigis TikTok Labeler - 第一性原理重寫版
// 核心：最小複雜度 + 最大可見性

console.log('[Aigis] ✅ Extension loaded');

// === 配置 ===
const API_URL = 'http://127.0.0.1:5000/api/label';

const KEY_MAP = {
  'ArrowLeft': {label: 0, text: 'REAL', color: '#00ff00'},
  'ArrowRight': {label: 1, text: 'AI', color: '#ff0000'},
  'KeyQ': {label: 1, reason: 'motion_jitter', text: 'AI: MOTION', color: '#ff0000'},
  'KeyW': {label: 1, reason: 'lighting_error', text: 'AI: LIGHT', color: '#ff0000'},
  'KeyE': {label: 1, reason: 'artifacts', text: 'AI: PIXEL', color: '#ff0000'},
  'KeyR': {label: 1, reason: 'lipsync_fail', text: 'AI: LIPSYNC', color: '#ff0000'},
  'ArrowDown': {label: null, text: 'SKIP', color: '#cccccc'}
};

// === 第一性原理：全局監聽，無過濾 ===
document.addEventListener('keydown', handleKeyPress, true);

function handleKeyPress(e) {
  console.log('[Aigis] Key pressed:', e.code, e.key);

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
  const payload = {
    timestamp: new Date().toISOString(),
    video_url: window.location.href,
    author_id: extractAuthorId(),
    label: label,
    reason: reason,
    source_version: 'aigis_v1'
  };

  console.log('[Aigis] Sending to API:', payload);

  try {
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

    // 顯示總數
    if (result.total_count) {
      showMiniNotification(`已標註: ${result.total_count}`);
    }

  } catch (err) {
    console.error('[Aigis] ❌ API Error:', err);
    showMiniNotification('⚠️ 伺服器離線');
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

function extractAuthorId() {
  try {
    const match = window.location.pathname.match(/@([^/]+)/);
    return match ? match[1] : 'unknown';
  } catch {
    return 'unknown';
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

console.log('[Aigis] 🎯 Ready! Press ← or → to label');
