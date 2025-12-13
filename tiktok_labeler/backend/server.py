#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSAR-RAPTOR TikTok Labeler Backend Server
後端服務器：接收Chrome擴展標註，存儲到Excel A

設計原則:
- 第一性原理: O(1)去重 + 即時落盤
- 沙皇炸彈: 高頻寫入，數據爆炸性增長
- 猛禽3: 純函數邏輯，無副作用

功能:
1. 接收標註API (/api/label)
2. 實時存儲到Excel A (labels_raw.xlsx)
3. 去重機制（避免重複標註）
4. 統計信息API (/api/stats)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 允許Chrome擴展跨域

# ========== 配置 ==========
BASE_DIR = Path(__file__).parent

# 使用新的路徑配置
import sys
sys.path.insert(0, str(BASE_DIR.parent))
from config import EXCEL_A_PATH, LAYER1_DATA_DIR

# 確保目錄存在
LAYER1_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ========== 內存去重集合 ==========
loaded_urls = set()
label_counts = {
    'total': 0,
    'real': 0,
    'ai': 0,
    'uncertain': 0,
    'exclude': 0
}


def initialize():
    """服務器啟動時初始化"""
    global loaded_urls, label_counts

    if EXCEL_A_PATH.exists():
        # 加載已有數據
        df = pd.read_excel(EXCEL_A_PATH)

        # 兼容處理：支持舊格式(url)和新格式(影片網址)
        url_col = '影片網址' if '影片網址' in df.columns else 'url'
        label_col = '判定結果' if '判定結果' in df.columns else 'label'

        loaded_urls = set(df[url_col].values)

        # 統計各類別數量（不區分大小寫）
        label_counts['total'] = len(df)
        for label in ['real', 'ai', 'uncertain', 'exclude']:
            label_counts[label] = len(df[df[label_col].str.lower() == label])

        logger.info(f"✅ 已加載 {len(loaded_urls)} 條標註記錄")
        logger.info(f"   統計: Real={label_counts['real']}, "
                   f"AI={label_counts['ai']}, "
                   f"Uncertain={label_counts['uncertain']}, "
                   f"Exclude={label_counts['exclude']}")
    else:
        # 創建新的Excel A（調整列順序：重要信息前置）
        df = pd.DataFrame(columns=[
            '序號', '影片網址', '判定結果', '標註時間',
            '視頻ID', '作者', '標題', '點贊數', '來源', '版本'
        ])
        df.to_excel(EXCEL_A_PATH, index=False)
        logger.info("✅ 創建新的Excel A")


@app.route('/api/label', methods=['POST'])
def label():
    """
    標註API

    接收格式:
    {
        "url": "https://www.tiktok.com/@user/video/123",
        "video_id": "123",
        "author": "user",
        "title": "...",
        "likes": "1.2M",
        "label": "real|ai|uncertain|exclude",
        "timestamp": "2025-12-12T10:30:00Z",
        "source": "tiktok_chrome_extension",
        "version": "2.0.0"
    }

    返回格式:
    {
        "status": "saved|duplicate|error",
        "total_count": 123,
        "message": "..."
    }
    """
    try:
        data = request.json
        url = data.get('url')
        label = data.get('label')

        # 驗證必填字段
        if not url or not label:
            return jsonify({
                'status': 'error',
                'message': '缺少必填字段 (url, label)'
            }), 400

        # 去重檢查
        if url in loaded_urls:
            logger.info(f"⚠️  重複標註: {url}")
            return jsonify({
                'status': 'duplicate',
                'total_count': label_counts['total'],
                'message': '該視頻已標註過'
            })

        # 讀取現有數據
        df_existing = pd.read_excel(EXCEL_A_PATH)

        # 準備數據行（使用中文列名，重要信息前置）
        new_row = {
            '序號': len(df_existing) + 1,  # 自動編號
            '影片網址': url,
            '判定結果': label.upper(),  # 大寫顯示 (REAL/AI/UNCERTAIN/EXCLUDE)
            '標註時間': data.get('timestamp', datetime.now().isoformat()),
            '視頻ID': data.get('video_id', 'unknown'),
            '作者': data.get('author', 'unknown'),
            '標題': data.get('title', 'N/A'),
            '點贊數': data.get('likes', '0'),
            '來源': data.get('source', 'unknown'),
            '版本': data.get('version', '2.0.0')
        }

        # 即時寫入Excel A（追加模式）
        df_new = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
        df_new.to_excel(EXCEL_A_PATH, index=False)

        # 更新內存
        loaded_urls.add(url)
        label_counts['total'] += 1
        label_counts[label] += 1

        logger.info(f"✅ 標註已保存: {label.upper()} | 總計={label_counts['total']}")

        return jsonify({
            'status': 'saved',
            'total_count': label_counts['total'],
            'label_counts': label_counts,
            'message': f'標註成功 ({label})'
        })

    except Exception as e:
        logger.error(f"❌ 處理標註失敗: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """
    統計信息API

    返回格式:
    {
        "total": 123,
        "real": 50,
        "ai": 40,
        "uncertain": 20,
        "exclude": 13,
        "excel_a_path": "...",
        "last_updated": "2025-12-12T10:30:00Z"
    }
    """
    try:
        # 獲取文件最後修改時間
        last_updated = "N/A"
        if EXCEL_A_PATH.exists():
            timestamp = os.path.getmtime(EXCEL_A_PATH)
            last_updated = datetime.fromtimestamp(timestamp).isoformat()

        return jsonify({
            'total': label_counts['total'],
            'real': label_counts['real'],
            'ai': label_counts['ai'],
            'uncertain': label_counts['uncertain'],
            'exclude': label_counts['exclude'],
            'excel_a_path': str(EXCEL_A_PATH),
            'last_updated': last_updated
        })

    except Exception as e:
        logger.error(f"❌ 獲取統計失敗: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/export', methods=['GET'])
def export_labels():
    """
    導出標註列表API（用於批量下載）

    查詢參數:
    - label: 過濾特定標籤 (real|ai|uncertain|exclude)
    - limit: 限制返回數量

    返回格式:
    {
        "labels": [
            {"url": "...", "label": "real", "video_id": "123", ...},
            ...
        ],
        "count": 50
    }
    """
    try:
        # 讀取Excel A
        df = pd.read_excel(EXCEL_A_PATH)

        # 過濾標籤
        label_filter = request.args.get('label')
        if label_filter:
            df = df[df['label'] == label_filter]

        # 限制數量
        limit = request.args.get('limit', type=int)
        if limit:
            df = df.head(limit)

        # 轉換為JSON
        labels = df.to_dict('records')

        return jsonify({
            'labels': labels,
            'count': len(labels)
        })

    except Exception as e:
        logger.error(f"❌ 導出標註失敗: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/reset', methods=['POST'])
def reset():
    """
    重置數據庫（謹慎使用！）

    需要確認參數: {"confirm": "yes"}
    """
    data = request.json
    if data and data.get('confirm') == 'yes':
        global loaded_urls, label_counts

        # 備份舊數據
        if EXCEL_A_PATH.exists():
            backup_path = DATA_DIR / f"excel_a_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            import shutil
            shutil.copy(EXCEL_A_PATH, backup_path)
            logger.info(f"✅ 已備份至: {backup_path}")

        # 重置
        df = pd.DataFrame(columns=[
            'timestamp', 'url', 'video_id', 'author', 'title',
            'likes', 'label', 'source', 'version'
        ])
        df.to_excel(EXCEL_A_PATH, index=False)

        loaded_urls.clear()
        label_counts = {
            'total': 0,
            'real': 0,
            'ai': 0,
            'uncertain': 0,
            'exclude': 0
        }

        logger.info("✅ 數據庫已重置")
        return jsonify({
            'status': 'reset',
            'message': '數據庫已重置'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': '需要確認參數 {"confirm": "yes"}'
        }), 400


if __name__ == '__main__':
    print("="*80)
    print("🚀 TSAR-RAPTOR TikTok Labeler Backend Server")
    print("="*80)
    print(f"📊 Excel A 路徑: {EXCEL_A_PATH}")
    print(f"🌐 API 端點: http://127.0.0.1:5000/api/label")
    print(f"📈 統計信息: http://127.0.0.1:5000/api/stats")
    print("="*80)
    print()

    initialize()

    print("\n🟢 服務器啟動中...")
    app.run(host='127.0.0.1', port=5000, debug=False)
