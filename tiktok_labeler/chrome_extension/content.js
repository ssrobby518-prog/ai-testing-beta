/**
 * TSAR-RAPTOR TikTok Labeler - Chrome Extension
 * Tinder式快速標註系統
 *
 * 設計原則:
 * - 第一性原理: 最小摩擦力，直覺判斷
 * - 沙皇炸彈: 高頻標註，數據爆炸性增長
 * - 猛禽3: 簡約界面，極速操作
 *
 * 按鍵映射:
 * ← ArrowLeft:  Real（真實）
 * → ArrowRight: AI（生成）
 * ↑ ArrowUp:    Uncertain（不確定）
 * ↓ ArrowDown:  Movie/Anime（電影/動畫，排除）
 */

// ========== 配置 ==========
const CONFIG = {
  API_ENDPOINT: 'http://127.0.0.1:5000/api/label',
  KEY_MAPPING: {
    'ArrowLeft': {
      label: 'real',
      emoji: '✅',
      color: '#00ff00',
      text: 'REAL'
    },
    'ArrowRight': {
      label: 'ai',
      emoji: '🤖',
      color: '#ff0000',
      text: 'AI'
    },
    'ArrowUp': {
      label: 'uncertain',
      emoji: '❓',
      color: '#ffaa00',
      text: 'UNCERTAIN'
    },
    'ArrowDown': {
      label: 'exclude',  // 電影/動畫
      emoji: '🎬',
      color: '#888888',
      text: 'MOVIE/ANIME'
    }
  }
};

// ========== 狀態管理 ==========
let sessionStats = {
  total: 0,
  real: 0,
  ai: 0,
  uncertain: 0,
  exclude: 0,
  lastLabel: null
};

// ========== 初始化 ==========
function initialize() {
  console.log('[TSAR-RAPTOR] ✅ TikTok Labeler 已激活');
  console.log('[TSAR-RAPTOR] 按鍵: ← Real | → AI | ↑ Uncertain | ↓ Movie/Anime');

  // 顯示歡迎提示
  showWelcomeBanner();

  // 加載會話統計
  loadSessionStats();

  // 創建統計面板
  createStatsPanel();
}

// ========== 歡迎橫幅 ==========
function showWelcomeBanner() {
  const banner = document.createElement('div');
  banner.className = 'tsar-welcome-banner';
  banner.innerHTML = `
    <div class="tsar-welcome-content">
      <h2>🚀 TSAR-RAPTOR TikTok Labeler</h2>
      <p>Tinder式快速標註已啟動</p>
      <div class="tsar-key-guide">
        <span>← Real</span>
        <span>→ AI</span>
        <span>↑ Uncertain</span>
        <span>↓ Movie/Anime</span>
      </div>
    </div>
  `;
  document.body.appendChild(banner);

  // 3秒後自動消失
  setTimeout(() => {
    banner.style.opacity = '0';
    setTimeout(() => banner.remove(), 500);
  }, 3000);
}

// ========== 統計面板 ==========
function createStatsPanel() {
  const panel = document.createElement('div');
  panel.id = 'tsar-stats-panel';
  panel.className = 'tsar-stats-panel';
  panel.innerHTML = `
    <div class="tsar-stats-header">TSAR-RAPTOR</div>
    <div class="tsar-stats-content">
      <div class="tsar-stat-item">
        <span class="tsar-stat-label">Total:</span>
        <span class="tsar-stat-value" id="tsar-total">0</span>
      </div>
      <div class="tsar-stat-item">
        <span class="tsar-stat-label">✅ Real:</span>
        <span class="tsar-stat-value" id="tsar-real">0</span>
      </div>
      <div class="tsar-stat-item">
        <span class="tsar-stat-label">🤖 AI:</span>
        <span class="tsar-stat-value" id="tsar-ai">0</span>
      </div>
      <div class="tsar-stat-item">
        <span class="tsar-stat-label">❓ Uncertain:</span>
        <span class="tsar-stat-value" id="tsar-uncertain">0</span>
      </div>
      <div class="tsar-stat-item">
        <span class="tsar-stat-label">🎬 Exclude:</span>
        <span class="tsar-stat-value" id="tsar-exclude">0</span>
      </div>
    </div>
  `;
  document.body.appendChild(panel);

  // 更新顯示
  updateStatsPanel();
}

function updateStatsPanel() {
  document.getElementById('tsar-total').textContent = sessionStats.total;
  document.getElementById('tsar-real').textContent = sessionStats.real;
  document.getElementById('tsar-ai').textContent = sessionStats.ai;
  document.getElementById('tsar-uncertain').textContent = sessionStats.uncertain;
  document.getElementById('tsar-exclude').textContent = sessionStats.exclude;
}

// ========== 主要邏輯：按鍵監聽 ==========
document.addEventListener('keydown', async (e) => {
  // 過濾：忽略輸入框
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
    return;
  }

  // 檢查是否為標註按鍵
  const mapping = CONFIG.KEY_MAPPING[e.code];
  if (!mapping) {
    return;
  }

  e.preventDefault();
  e.stopPropagation();

  // === 沙皇炸彈：原子化事務 ===
  // Step 1: 提取當前視頻信息
  const videoData = extractVideoData();
  if (!videoData) {
    console.error('[TSAR-RAPTOR] ❌ 無法提取視頻信息');
    return;
  }

  // Step 2: 視覺反饋（即時）
  showLabelFeedback(mapping);

  // Step 3: 發送標註到後端（非阻塞）
  await sendLabelToBackend(videoData, mapping.label);

  // Step 4: 更新統計
  updateSessionStats(mapping.label);

  // Step 5: 自動滾動到下一個視頻（猛禽3：零延遲）
  setTimeout(() => {
    scrollToNextVideo();
  }, 300);
});

// ========== 提取視頻數據 ==========
function extractVideoData() {
  try {
    // 獲取當前URL
    const url = window.location.href;

    // 提取視頻ID
    const videoIdMatch = url.match(/\/video\/(\d+)/);
    const videoId = videoIdMatch ? videoIdMatch[1] : 'unknown';

    // 提取作者
    const authorMatch = url.match(/@([^/]+)/);
    const author = authorMatch ? authorMatch[1] : 'unknown';

    // 嘗試提取視頻標題（從DOM）
    let title = 'N/A';
    const titleElement = document.querySelector('[data-e2e="browse-video-desc"]');
    if (titleElement) {
      title = titleElement.textContent.trim();
    }

    // 嘗試提取點贊數、評論數等
    const likeElement = document.querySelector('[data-e2e="browse-like-count"]');
    const likes = likeElement ? likeElement.textContent.trim() : '0';

    return {
      url: url,
      video_id: videoId,
      author: author,
      title: title,
      likes: likes,
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    console.error('[TSAR-RAPTOR] 提取視頻數據失敗:', error);
    return null;
  }
}

// ========== 視覺反饋 ==========
function showLabelFeedback(mapping) {
  const feedback = document.createElement('div');
  feedback.className = 'tsar-label-feedback';
  feedback.innerHTML = `
    <div class="tsar-feedback-emoji">${mapping.emoji}</div>
    <div class="tsar-feedback-text">${mapping.text}</div>
  `;
  feedback.style.color = mapping.color;
  feedback.style.textShadow = `0 0 20px ${mapping.color}`;

  document.body.appendChild(feedback);

  // 動畫後移除
  setTimeout(() => {
    feedback.style.opacity = '0';
    setTimeout(() => feedback.remove(), 500);
  }, 800);
}

// ========== 發送標註到後端 ==========
async function sendLabelToBackend(videoData, label) {
  const payload = {
    ...videoData,
    label: label,
    source: 'tiktok_chrome_extension',
    version: '2.0.0'
  };

  try {
    const response = await fetch(CONFIG.API_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const result = await response.json();
    console.log('[TSAR-RAPTOR] ✅ 標註已保存:', result);

    // 顯示成功提示
    if (result.total_count && result.total_count % 10 === 0) {
      showToast(`已標註 ${result.total_count} 個視頻！`);
    }

  } catch (error) {
    console.error('[TSAR-RAPTOR] ❌ 發送失敗:', error);
    showToast('⚠️ 連接後端失敗，請檢查服務器', 'error');
  }
}

// ========== 更新統計 ==========
function updateSessionStats(label) {
  sessionStats.total++;

  switch (label) {
    case 'real':
      sessionStats.real++;
      break;
    case 'ai':
      sessionStats.ai++;
      break;
    case 'uncertain':
      sessionStats.uncertain++;
      break;
    case 'exclude':
      sessionStats.exclude++;
      break;
  }

  sessionStats.lastLabel = label;

  // 保存到localStorage
  saveSessionStats();

  // 更新面板
  updateStatsPanel();
}

// ========== 本地存儲 ==========
function saveSessionStats() {
  chrome.storage.local.set({ sessionStats: sessionStats });
}

function loadSessionStats() {
  chrome.storage.local.get(['sessionStats'], (result) => {
    if (result.sessionStats) {
      sessionStats = result.sessionStats;
      updateStatsPanel();
    }
  });
}

// ========== 滾動到下一個視頻 ==========
function scrollToNextVideo() {
  // 模擬按下向下鍵（TikTok原生滾動）
  const scrollEvent = new KeyboardEvent('keydown', {
    key: 'ArrowDown',
    code: 'ArrowDown',
    keyCode: 40,
    bubbles: true,
    cancelable: true
  });

  document.dispatchEvent(scrollEvent);
}

// ========== Toast 提示 ==========
function showToast(message, type = 'success') {
  const toast = document.createElement('div');
  toast.className = `tsar-toast tsar-toast-${type}`;
  toast.textContent = message;

  document.body.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 2000);
}

// ========== 初始化執行 ==========
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initialize);
} else {
  initialize();
}
