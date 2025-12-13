#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Aigis - 主動學習驅動的AI檢測系統總控
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

第一性原理 (First Principles):
    效率 = (數據 × 速度) / 摩擦力

    核心洞察：
    1. 數據原子化：每個視頻 = 一個事務 (View -> Label -> Save -> Next)
    2. 摩擦力歸零：直接在數據源(TikTok DOM)標註，消除窗口切換
    3. 直覺數字化：將人類"直覺"(抖動、光照)轉換為結構化標籤

沙皇炸彈原則 (Tsar Bomba):
    - 高頻流式標註：2秒/視頻 (傳統批次處理的10倍速)
    - 主動學習挖掘：只標註"困惑區"樣本 (0.4-0.6信心度)
    - 夜間管道自動化：睡覺時系統自動下載+訓練

猛禽3引擎原則 (Raptor 3):
    - 模塊化架構：Frontend(Chrome擴展) + Backend(Flask) + Pipeline(ETL)
    - 並行化處理：標註與下載/訓練解耦
    - 零依賴啟動：本地運行，無需雲端

Architecture:
    Layer 1 - The Eye (Chrome Extension): TikTok DOM標註界面
    Layer 2 - The Bridge (REST API): HTTP通信層
    Layer 3 - The Brain (Flask Server): 數據持久化 + 模型服務
    Layer 4 - The Loop (Pipeline): 自動化ETL + 主動學習

Target: 70% -> 99% 準確率 (7天衝刺)
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === 配置 ===
BASE_DIR = Path(__file__).parent
AIGIS_DIR = BASE_DIR / "aigis"
EXTENSION_DIR = AIGIS_DIR / "TikTok_Labeler_Extension"
SERVER_DIR = AIGIS_DIR / "TikTok_Labeler_Server"
PIPELINE_DIR = AIGIS_DIR / "Pipeline"

# 藍隊模組配置（複用）
BLUE_TEAM_MODULES_DIR = BASE_DIR / "modules"

# === Aigis系統配置（基於YAML定義）===
AIGIS_CONFIG = {
    "meta": {
        "name": "Aigis",
        "version": "2.0.0",
        "objective": "高頻流式主動學習",
        "target_accuracy": 0.99
    },

    "key_mapping": {
        # 二元分類（快速模式）
        "ArrowLeft": {"label": 0, "semantic": "Real", "color": "#00ff00"},
        "ArrowRight": {"label": 1, "semantic": "AI", "color": "#ff0000"},

        # 特徵診斷（進階模式）
        "KeyQ": {"label": 1, "reason": "motion_jitter", "semantic": "AI-動作抖動"},
        "KeyW": {"label": 1, "reason": "lighting_error", "semantic": "AI-光照錯誤"},
        "KeyE": {"label": 1, "reason": "artifacts", "semantic": "AI-視覺偽影"},
        "KeyR": {"label": 1, "reason": "lipsync_fail", "semantic": "AI-唇音不同步"},
        "ArrowDown": {"label": None, "semantic": "Skip"}
    },

    "api": {
        "endpoint": "http://127.0.0.1:5000/api/label",
        "method": "POST"
    },

    "pipeline": {
        "download_dir": str(PIPELINE_DIR / "downloaded_videos"),
        "features_file": str(PIPELINE_DIR / "features_matrix.csv"),
        "model_file": str(PIPELINE_DIR / "model_latest.json"),
        "dataset_file": str(SERVER_DIR / "dataset.csv")
    }
}


class AigisOrchestrator:
    """
    Aigis總控協調器

    職責：
    1. 初始化所有子系統
    2. 啟動Flask服務器
    3. 觸發自動化管道
    4. 管理主動學習循環

    猛禽3：單一職責，純協調邏輯
    """

    def __init__(self):
        self.config = AIGIS_CONFIG
        self._ensure_directories()

    def _ensure_directories(self):
        """創建必要目錄"""
        AIGIS_DIR.mkdir(exist_ok=True)
        EXTENSION_DIR.mkdir(exist_ok=True)
        SERVER_DIR.mkdir(exist_ok=True)
        PIPELINE_DIR.mkdir(exist_ok=True)
        (PIPELINE_DIR / "downloaded_videos").mkdir(exist_ok=True)

    def bootstrap_system(self):
        """
        系統引導（首次運行）

        沙皇炸彈：一鍵部署所有組件
        """
        logging.info("🚀 Aigis System Bootstrap - 系統引導中...")

        # 1. 生成Chrome擴展
        logging.info("📦 [1/4] 生成Chrome擴展...")
        self._generate_extension()

        # 2. 生成Flask服務器
        logging.info("🧠 [2/4] 生成Flask服務器...")
        self._generate_server()

        # 3. 生成Pipeline腳本
        logging.info("⚙️ [3/4] 生成自動化管道...")
        self._generate_pipeline()

        # 4. 生成使用指南
        logging.info("📚 [4/4] 生成使用指南...")
        self._generate_guide()

        logging.info("✅ 系統引導完成！")
        self._print_next_steps()

    def _generate_extension(self):
        """生成Chrome擴展（The Eye）"""
        # manifest.json
        manifest = {
            "manifest_version": 3,
            "name": "Aigis - TikTok Rapid Labeler",
            "version": "1.0.0",
            "description": "高速TikTok視頻標註工具 - 2秒/視頻",
            "permissions": ["activeTab", "scripting"],
            "host_permissions": [
                "https://www.tiktok.com/*",
                "http://127.0.0.1:5000/*"
            ],
            "content_scripts": [{
                "matches": ["https://www.tiktok.com/*"],
                "js": ["content.js"],
                "css": ["styles.css"],
                "run_at": "document_end"
            }],
            "icons": {
                "16": "icon16.png",
                "48": "icon48.png",
                "128": "icon128.png"
            }
        }

        with open(EXTENSION_DIR / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # content.js（核心邏輯）
        content_js = """// Aigis TikTok Labeler - Content Script
// 第一性原理：最小摩擦力標註

const KEY_MAP = {
  'ArrowLeft': {label: 0, text: 'REAL', color: '#00ff00'},
  'ArrowRight': {label: 1, text: 'AI', color: '#ff0000'},
  'KeyQ': {label: 1, reason: 'motion_jitter', text: 'AI: MOTION', color: '#ff0000'},
  'KeyW': {label: 1, reason: 'lighting_error', text: 'AI: LIGHT', color: '#ff0000'},
  'KeyE': {label: 1, reason: 'artifacts', text: 'AI: PIXEL', color: '#ff0000'},
  'KeyR': {label: 1, reason: 'lipsync_fail', text: 'AI: LIPSYNC', color: '#ff0000'},
  'ArrowDown': {label: null, text: 'SKIP', color: '#cccccc'}
};

// 猛禽3：事件驅動架構
document.addEventListener('keydown', async (e) => {
  // 過濾：忽略輸入框
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  const mapping = KEY_MAP[e.code];
  if (!mapping) return;

  e.preventDefault();

  // === 沙皇炸彈：原子化事務 ===
  // Step 1: 視覺反饋（即時）
  showFeedback(mapping.text, mapping.color);

  // Step 2: 數據傳輸（非阻塞）
  if (mapping.label !== null) {
    sendLabel(mapping.label, mapping.reason);
  }

  // Step 3: 自動滾動（0延遲）
  simulateScroll();
});

function showFeedback(text, color) {
  const overlay = document.createElement('div');
  overlay.className = 'aigis-feedback';
  overlay.textContent = text;
  overlay.style.color = color;
  document.body.appendChild(overlay);

  setTimeout(() => overlay.remove(), 500);
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

  try {
    const response = await fetch('http://127.0.0.1:5000/api/label', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });

    const result = await response.json();
    console.log('[Aigis]', result.status, `Total: ${result.total_count}`);
  } catch (err) {
    console.error('[Aigis] 傳輸失敗:', err);
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

function simulateScroll() {
  const event = new KeyboardEvent('keydown', {
    key: 'ArrowDown',
    code: 'ArrowDown',
    bubbles: true
  });
  document.dispatchEvent(event);
}

console.log('[Aigis] ✅ 已激活 - 使用 ←/→ 快速標註');
"""

        with open(EXTENSION_DIR / "content.js", "w", encoding="utf-8") as f:
            f.write(content_js)

        # styles.css
        styles_css = """.aigis-feedback {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 99999;
  font-size: 4rem;
  font-weight: bold;
  text-shadow: 0 0 10px #000;
  pointer-events: none;
  animation: aigis-flash 0.5s ease-out;
}

@keyframes aigis-flash {
  0% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
  100% { opacity: 0; transform: translate(-50%, -50%) scale(1.5); }
}
"""

        with open(EXTENSION_DIR / "styles.css", "w", encoding="utf-8") as f:
            f.write(styles_css)

        # 創建佔位圖標
        for size in [16, 48, 128]:
            icon_file = EXTENSION_DIR / f"icon{size}.png"
            if not icon_file.exists():
                icon_file.write_bytes(b'')  # 空文件佔位

    def _generate_server(self):
        """生成Flask服務器（The Brain）"""
        server_py = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aigis Backend Server - 數據持久化 + 模型服務
第一性原理：O(1)去重 + 即時落盤
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import csv
import os
from pathlib import Path

app = Flask(__name__)
CORS(app)  # 允許Chrome擴展跨域

# === 配置 ===
DATASET_FILE = Path(__file__).parent / "dataset.csv"
DATASET_HEADER = ["timestamp", "video_url", "author_id", "label", "reason", "source_version"]

# === 內存去重集合（沙皇炸彈：O(1)查詢）===
loaded_urls = set()

def initialize():
    """啟動時加載已有數據"""
    global loaded_urls

    if DATASET_FILE.exists():
        with open(DATASET_FILE, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                loaded_urls.add(row['video_url'])
        print(f"✅ 已加載 {len(loaded_urls)} 條記錄")
    else:
        # 創建CSV表頭
        with open(DATASET_FILE, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=DATASET_HEADER)
            writer.writeheader()
        print("✅ 創建新數據集")

@app.route('/api/label', methods=['POST'])
def label():
    """
    標註API

    猛禽3：純函數邏輯，無副作用（除了文件寫入）
    """
    data = request.json
    url = data.get('video_url')

    # 去重檢查
    if url in loaded_urls:
        return jsonify({
            'status': 'duplicate',
            'total_count': len(loaded_urls)
        })

    # 持久化（即時寫入）
    with open(DATASET_FILE, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_HEADER)
        writer.writerow(data)

    # 更新內存
    loaded_urls.add(url)

    return jsonify({
        'status': 'saved',
        'total_count': len(loaded_urls)
    })

@app.route('/api/stats', methods=['GET'])
def stats():
    """統計信息"""
    return jsonify({
        'total': len(loaded_urls),
        'dataset_file': str(DATASET_FILE)
    })

if __name__ == '__main__':
    initialize()
    print("🧠 Aigis Backend Server 啟動中...")
    print("📊 API端點: http://127.0.0.1:5000/api/label")
    app.run(host='127.0.0.1', port=5000, debug=False)
'''

        with open(SERVER_DIR / "server.py", "w", encoding="utf-8") as f:
            f.write(server_py)

        # requirements.txt
        with open(SERVER_DIR / "requirements.txt", "w") as f:
            f.write("flask\nflask-cors\npandas\n")

    def _generate_pipeline(self):
        """生成自動化管道（The Loop）"""
        pipeline_py = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aigis Pipeline - 夜間自動化ETL + 主動學習
第一性原理：睡覺時訓練，醒來時收穫
"""

import os
import sys
import subprocess
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# === 配置 ===
BASE_DIR = Path(__file__).parent
DATASET_FILE = BASE_DIR.parent / "TikTok_Labeler_Server" / "dataset.csv"
DOWNLOAD_DIR = BASE_DIR / "downloaded_videos"
FEATURES_FILE = BASE_DIR / "features_matrix.csv"
MODEL_FILE = BASE_DIR / "model_latest.json"

# 藍隊模組路徑
BLUE_TEAM_DIR = BASE_DIR.parent.parent / "modules"

def phase_1_download():
    """
    Phase 1: 下載視頻

    沙皇炸彈：增量下載（只下載新增）
    """
    logging.info("📥 [Phase 1] 下載視頻中...")

    if not DATASET_FILE.exists():
        logging.error("❌ dataset.csv 不存在，請先標註數據")
        return

    df = pd.read_csv(DATASET_FILE)
    urls = df['video_url'].unique()

    # 計算增量
    existing_files = set(DOWNLOAD_DIR.glob("*.mp4"))
    existing_ids = {f.stem for f in existing_files}

    to_download = []
    for url in urls:
        video_id = url.split('/')[-1].split('?')[0]
        if video_id not in existing_ids:
            to_download.append(url)

    if not to_download:
        logging.info("✅ 所有視頻已下載")
        return

    logging.info(f"📥 需下載 {len(to_download)} 個視頻...")

    # 使用yt-dlp下載
    for i, url in enumerate(to_download):
        try:
            subprocess.run([
                "yt-dlp",
                "-o", str(DOWNLOAD_DIR / "%(id)s.%(ext)s"),
                url
            ], check=True, capture_output=True)
            logging.info(f"  [{i+1}/{len(to_download)}] ✓")
        except Exception as e:
            logging.error(f"  [{i+1}/{len(to_download)}] ✗ {e}")

    logging.info("✅ 下載完成")

def phase_2_extract():
    """
    Phase 2: 特徵提取

    猛禽3：並行化處理（未來優化）
    """
    logging.info("🔬 [Phase 2] 特徵提取中...")

    # 加載標註
    df_labels = pd.read_csv(DATASET_FILE)

    # 加載已有特徵（增量）
    if FEATURES_FILE.exists():
        df_features = pd.read_csv(FEATURES_FILE)
        processed_ids = set(df_features['video_id'])
    else:
        df_features = pd.DataFrame()
        processed_ids = set()

    # 計算增量
    video_files = list(DOWNLOAD_DIR.glob("*.mp4"))
    new_files = [f for f in video_files if f.stem not in processed_ids]

    if not new_files:
        logging.info("✅ 所有視頻已提取特徵")
        return

    logging.info(f"🔬 需提取 {len(new_files)} 個視頻...")

    # TODO: 調用藍隊12模組
    # 暫時返回隨機特徵
    logging.warning("⚠️ 特徵提取未實現，請手動集成藍隊模組")

def phase_3_train():
    """
    Phase 3: 模型訓練

    沙皇炸彈：XGBoost + 主動學習
    """
    logging.info("🧠 [Phase 3] 模型訓練中...")

    if not FEATURES_FILE.exists():
        logging.error("❌ features_matrix.csv 不存在")
        return

    # TODO: XGBoost訓練
    logging.warning("⚠️ 模型訓練未實現")

def main():
    """主流程"""
    logging.info("🚀 Aigis Pipeline 啟動")

    phase_1_download()
    phase_2_extract()
    phase_3_train()

    logging.info("✅ Pipeline 完成")

if __name__ == "__main__":
    main()
'''

        with open(PIPELINE_DIR / "pipeline.py", "w", encoding="utf-8") as f:
            f.write(pipeline_py)

    def _generate_guide(self):
        """生成使用指南"""
        guide_md = """# Aigis 使用指南

## 🚀 快速開始（5分鐘部署）

### 1. 啟動Backend服務器

```bash
cd aigis/TikTok_Labeler_Server
pip install -r requirements.txt
python server.py
```

看到：`🧠 Aigis Backend Server 啟動中...`

### 2. 安裝Chrome擴展

1. 打開Chrome瀏覽器
2. 訪問 `chrome://extensions/`
3. 開啟"開發者模式"
4. 點擊"加載已解壓的擴展程序"
5. 選擇 `aigis/TikTok_Labeler_Extension` 目錄

### 3. 開始標註

1. 訪問 https://www.tiktok.com/foryou
2. 使用鍵盤快捷鍵：
   - **← 左箭頭**: Real（真實）
   - **→ 右箭頭**: AI（生成）
   - **Q**: AI - 動作抖動
   - **W**: AI - 光照錯誤
   - **E**: AI - 視覺偽影
   - **R**: AI - 唇音不同步
   - **↓ 下箭頭**: 跳過

3. 每次標註後會自動滾動到下一個視頻

### 4. 夜間管道（自動化）

```bash
cd aigis/Pipeline
python pipeline.py
```

---

## 📊 7天衝刺計劃

### Day 1: 大量攝入
- 目標：標註2000+視頻（只用←/→）
- 預期準確率：70%

### Day 2-3: 模組審計
- 分析特徵重要性
- 重構/替換底部3個模組

### Day 4: 困難樣本挖掘
- 只標註信心度0.4-0.6的視頻
- 使用Q/W/E/R添加原因標籤

### Day 5-7: 對抗訓練+集成
- 數據增強
- 集成Deep Learning
- 最終準確率：99%

---

**設計原則**: 沙皇炸彈 × 猛禽3引擎 × 第一性原理
"""

        with open(AIGIS_DIR / "README.md", "w", encoding="utf-8") as f:
            f.write(guide_md)

    def _print_next_steps(self):
        """打印下一步指令"""
        print("\n" + "="*80)
        print("🎉 Aigis系統已就緒！")
        print("="*80)
        print("\n📋 下一步:")
        print("\n1️⃣ 啟動Backend服務器:")
        print(f"   cd {SERVER_DIR}")
        print("   pip install -r requirements.txt")
        print("   python server.py")
        print("\n2️⃣ 安裝Chrome擴展:")
        print("   - 打開 chrome://extensions/")
        print("   - 開啟開發者模式")
        print(f"   - 加載目錄: {EXTENSION_DIR}")
        print("\n3️⃣ 開始標註:")
        print("   - 訪問 https://www.tiktok.com/foryou")
        print("   - 使用 ← / → 快速標註")
        print("\n4️⃣ 查看完整指南:")
        print(f"   {AIGIS_DIR / 'README.md'}")
        print("\n" + "="*80)


def main():
    """主入口"""
    orchestrator = AigisOrchestrator()
    orchestrator.bootstrap_system()


if __name__ == "__main__":
    main()
