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
import re
import subprocess
import sys

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


def resolve_tiktok_url(url: str) -> str:
    """
    解析TikTok短網址為真實網址

    短網址格式: vm.tiktok.com/xxx, vt.tiktok.com/xxx
    真實網址格式: https://www.tiktok.com/@user/video/123456

    Args:
        url: 輸入URL（可能是短網址）

    Returns:
        真實完整URL
    """
    url = str(url).strip()

    # 如果已經是完整URL（包含/video/），直接返回
    if '/video/' in url and 'www.tiktok.com' in url:
        return url

    # 檢測短網址
    if 'vm.tiktok.com' in url or 'vt.tiktok.com' in url or 'm.tiktok.com' in url:
        try:
            logger.info(f"🔗 解析短網址: {url}")

            # 使用 yt-dlp 解析真實URL
            cmd = [
                sys.executable, "-m", "yt_dlp",
                "--get-url",
                "--no-warnings",
                "--skip-download",
                url
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                # yt-dlp 返回真實視頻URL
                real_url = result.stdout.strip()

                # 從返回的URL中提取 TikTok 網頁URL
                # yt-dlp 可能返回直接下載URL，我們需要網頁URL
                # 嘗試從stderr或其他方式獲取

                # 使用 --dump-json 獲取完整信息
                cmd2 = [
                    sys.executable, "-m", "yt_dlp",
                    "--dump-json",
                    "--no-warnings",
                    "--skip-download",
                    url
                ]

                result2 = subprocess.run(
                    cmd2,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result2.returncode == 0:
                    import json
                    info = json.loads(result2.stdout)
                    webpage_url = info.get('webpage_url', '')
                    if webpage_url:
                        logger.info(f"✅ 解析成功: {webpage_url}")
                        return webpage_url

                # 如果 JSON 解析失敗，返回原始URL
                logger.warning(f"⚠️  無法解析短網址，使用原始URL: {url}")
                return url
            else:
                logger.warning(f"⚠️  解析短網址失敗: {result.stderr[:200]}")
                return url

        except Exception as e:
            logger.error(f"❌ 解析短網址異常: {e}")
            return url

    # 不是短網址，直接返回
    return url


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
        # 創建新的Excel A（調整列順序：重要信息前置 + 下載狀態）
        df = pd.DataFrame(columns=[
            '序號', '影片網址', '判定結果', '下載狀態', '標註時間',
            '視頻ID', '作者', '標題', '點贊數', '來源', '版本'
        ])
        df.to_excel(EXCEL_A_PATH, index=False)
        logger.info("✅ 創建新的Excel A")

    # 確保所有舊數據都有"下載狀態"列
    if EXCEL_A_PATH.exists():
        df = pd.read_excel(EXCEL_A_PATH)
        if '下載狀態' not in df.columns:
            df['下載狀態'] = '未下載'
            # 調整列順序
            cols = list(df.columns)
            if '判定結果' in cols and '下載狀態' in cols:
                # 將下載狀態移到判定結果後面
                cols.remove('下載狀態')
                idx = cols.index('判定結果') + 1
                cols.insert(idx, '下載狀態')
                df = df[cols]
            df.to_excel(EXCEL_A_PATH, index=False)
            logger.info("✅ 已添加「下載狀態」列到 Excel A")


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
        url_raw = data.get('url')
        label = data.get('label')

        # 驗證必填字段
        if not url_raw or not label:
            return jsonify({
                'status': 'error',
                'message': '缺少必填字段 (url, label)'
            }), 400

        # 解析短網址為真實網址
        url = resolve_tiktok_url(url_raw)

        # 去重檢查（使用真實URL）
        if url in loaded_urls:
            logger.info(f"⚠️  重複標註: {url}")
            return jsonify({
                'status': 'duplicate',
                'total_count': label_counts['total'],
                'message': '該視頻已標註過'
            })

        # 讀取現有數據
        df_existing = pd.read_excel(EXCEL_A_PATH)

        # 準備數據行（使用中文列名，重要信息前置 + 下載狀態）
        new_row = {
            '序號': len(df_existing) + 1,  # 自動編號
            '影片網址': url,  # 真實網址（已解析）
            '判定結果': label.upper(),  # 大寫顯示 (REAL/AI/UNCERTAIN/EXCLUDE)
            '下載狀態': '未下載',  # 初始狀態
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

        url_col = '影片網址' if '影片網址' in df.columns else 'url'
        label_col = '判定結果' if '判定結果' in df.columns else 'label'
        video_id_col = '視頻ID' if '視頻ID' in df.columns else 'video_id'
        author_col = '作者' if '作者' in df.columns else 'author'
        title_col = '標題' if '標題' in df.columns else 'title'
        likes_col = '點贊數' if '點贊數' in df.columns else 'likes'
        ts_col = '標註時間' if '標註時間' in df.columns else 'timestamp'

        # 過濾標籤
        label_filter = request.args.get('label')
        if label_filter:
            label_filter_lower = str(label_filter).strip().lower()
            df = df[df[label_col].astype(str).str.lower() == label_filter_lower]

        # 限制數量
        limit = request.args.get('limit', type=int)
        if limit:
            df = df.head(limit)

        labels = []
        for _, row in df.iterrows():
            labels.append({
                'url': row.get(url_col, ''),
                'label': str(row.get(label_col, '')).lower(),
                'video_id': str(row.get(video_id_col, 'unknown')),
                'author': row.get(author_col, 'unknown'),
                'title': row.get(title_col, 'N/A'),
                'likes': row.get(likes_col, '0'),
                'timestamp': row.get(ts_col, datetime.now().isoformat())
            })

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
            backup_path = LAYER1_DATA_DIR / f"excel_a_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            import shutil
            shutil.copy(EXCEL_A_PATH, backup_path)
            logger.info(f"✅ 已備份至: {backup_path}")

        # 重置
        df = pd.DataFrame(columns=[
            '序號', '影片網址', '判定結果', '標註時間',
            '視頻ID', '作者', '標題', '點贊數', '來源', '版本'
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


@app.route('/api/command', methods=['POST'])
def command():
    try:
        data = request.json or {}
        cmd = str(data.get('command', '')).strip()

        if cmd != '第一層下載':
            return jsonify({
                'status': 'ignored',
                'message': 'unknown_command'
            })

        import subprocess
        import sys

        pipeline_path = (BASE_DIR.parent / 'pipeline' / 'layer1_pipeline.py').resolve()
        result = subprocess.Popen(
            [sys.executable, str(pipeline_path), '--download-detect-report'],
            cwd=str(BASE_DIR.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
        )

        return jsonify({
            'status': 'started',
            'pid': result.pid
        })

    except Exception as e:
        logger.error(f"❌ 執行命令失敗: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    import sys
    import io

    # 設置輸出編碼為 UTF-8
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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
