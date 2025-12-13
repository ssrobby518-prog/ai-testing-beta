#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aigis Backend Server - 第一性原理重新設計
架構：三階段解耦

Phase 1（實時）: 快速標註 - 只記錄URL+標註
Phase 2（批量）: 用戶觸發 - 下載+分析
Phase 3（自動）: AI優化 - 訓練+調整模組
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import csv
import os
from pathlib import Path
from datetime import datetime
import logging
import threading
import time
import base64
import io
import subprocess
import pandas as pd

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# === 路徑配置 ===
BASE_DIR = Path(__file__).parent
DATASET_FILE = BASE_DIR / "dataset.csv"
TRAINING_FILE = BASE_DIR / "training_data.csv"
DOWNLOADS_DIR = BASE_DIR / "downloads"
TRAINING_HEADER = [
    "video_url","label","timestamp",
    "model_fingerprint","frequency_analysis","sensor_noise","physics_violation",
    "texture_noise","text_fingerprint","metadata_score","heartbeat","blink_dynamics",
    "lighting_geometry","av_sync","semantic_stylometry","bitrate","fps","duration",
    "resolution","author_id","reason"
]

# === 高速緩衝寫入（TINDER滑動保證不掉事件） ===
BUFFER_INTERVAL = 0.3  # 秒，批次落盤間隔
_buffer_labels = []
_buffer_features = []
_buffer_lock = threading.Lock()
_flusher_started = False
_shutdown_evt = threading.Event()
_hydrate_queue = []
_hydrate_pending = set()
_yt_dlp_checked = False
_yt_dlp_available = False
_yt_dlp_cmd = ["yt-dlp"]

DATASET_HEADER = ["timestamp", "video_url", "author_id", "label", "reason", "source_version"]

# === 內存去重集合 ===
loaded_urls = set()


def initialize():
    """啟動時加載已有數據"""
    global loaded_urls

    if DATASET_FILE.exists():
        with open(DATASET_FILE, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                loaded_urls.add(row['video_url'])
        logging.info(f"Loaded {len(loaded_urls)} labeled records")
    else:
        with open(DATASET_FILE, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=DATASET_HEADER)
            writer.writeheader()
        logging.info("Created new dataset.csv")

    if not TRAINING_FILE.exists():
        with open(TRAINING_FILE, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=TRAINING_HEADER)
            writer.writeheader()
        logging.info("Created new training_data.csv")

    # 下載資料夾
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _flush_once():
    """將緩衝區的事件批量寫入CSV（降低I/O成本，跟上高速滑動）"""
    global _buffer_labels, _buffer_features
    labels_to_write = []
    features_to_write = []
    with _buffer_lock:
        if _buffer_labels:
            labels_to_write = _buffer_labels
            _buffer_labels = []
        if _buffer_features:
            features_to_write = _buffer_features
            _buffer_features = []

    if labels_to_write:
        try:
            with open(DATASET_FILE, 'a', encoding='utf-8-sig', newline='') as f_ds, open(TRAINING_FILE, 'a', encoding='utf-8-sig', newline='') as f_tr:
                ds_writer = csv.DictWriter(f_ds, fieldnames=DATASET_HEADER)
                tr_writer = csv.DictWriter(f_tr, fieldnames=TRAINING_HEADER)
                for data in labels_to_write:
                    ds_writer.writerow(data)
                    minimal_row = {k: '' for k in TRAINING_HEADER}
                    minimal_row.update({
                        'video_url': data.get('video_url', ''),
                        'label': data.get('label', ''),
                        'timestamp': data.get('timestamp', datetime.utcnow().isoformat()),
                        'author_id': data.get('author_id', ''),
                        'reason': data.get('reason', '')
                    })
                    tr_writer.writerow(minimal_row)
        except Exception as e:
            logging.error(f"[FLUSH] Failed writing labels: {e}")

    if features_to_write:
        try:
            with open(TRAINING_FILE, 'a', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=TRAINING_HEADER)
                for row in features_to_write:
                    writer.writerow(row)
        except Exception as e:
            logging.error(f"[FLUSH] Failed writing features: {e}")


def _flusher_loop():
    logging.info(f"[FLUSHER] Started with interval={BUFFER_INTERVAL}s")
    while not _shutdown_evt.is_set():
        _flush_once()
        time.sleep(BUFFER_INTERVAL)
    logging.info("[FLUSHER] Stopped")


def _check_yt_dlp():
    global _yt_dlp_checked, _yt_dlp_available
    if _yt_dlp_checked:
        return _yt_dlp_available
    try:
        subprocess.run(["yt-dlp", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        _yt_dlp_available = True
        _yt_dlp_cmd = ["yt-dlp"]
        logging.info("[HYDRATE] yt-dlp available (exe)")
    except Exception:
        # fallback: python -m yt_dlp
        try:
            subprocess.run(["python", "-m", "yt_dlp", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            _yt_dlp_available = True
            _yt_dlp_cmd = ["python", "-m", "yt_dlp"]
            logging.info("[HYDRATE] yt-dlp available (python -m)")
        except Exception:
            _yt_dlp_available = False
            logging.warning("[HYDRATE] yt-dlp not found; hydration disabled")
    _yt_dlp_checked = True
    return _yt_dlp_available


def _hydrate_loop():
    logging.info("[HYDRATE] Loop started")
    while not _shutdown_evt.is_set():
        url = None
        with _buffer_lock:
            if _hydrate_queue:
                url = _hydrate_queue.pop(0)
                _hydrate_pending.discard(url)
        if url is not None and _check_yt_dlp():
            try:
                # 最快路徑：交給yt-dlp，下載到downloads目錄
                cmd = _yt_dlp_cmd + ["-o", str(DOWNLOADS_DIR / "%(id)s.%(ext)s"), url]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            except Exception as e:
                logging.error(f"[HYDRATE] Failed for {url}: {e}")
        time.sleep(1.0)
    logging.info("[HYDRATE] Loop stopped")


@app.route('/api/label', methods=['POST'])
def label():
    """
    Phase 1: 快速標註 - 沙皇炸彈設計

    只做一件事：記錄URL+標註到CSV
    不下載、不分析，零延遲響應
    """
    data = request.json
    url = data.get('video_url')

    # 去重檢查
    if url in loaded_urls:
        return jsonify({
            'status': 'duplicate',
            'total_count': len(loaded_urls)
        })

    # 緩衝入列（零阻塞返回）
    with _buffer_lock:
        _buffer_labels.append(data)
        if url and url not in _hydrate_pending:
            _hydrate_queue.append(url)
            _hydrate_pending.add(url)
    loaded_urls.add(url)

    # 立即返回（零延遲）
    return jsonify({
        'status': 'queued',
        'total_count': len(loaded_urls),
        'message': f'Queued {len(loaded_urls)} labels (flush {int(BUFFER_INTERVAL*1000)}ms)'
    })


@app.route('/api/feature', methods=['POST'])
def feature():
    """
    Phase 1+: 極速特徵落盤
    目標：擴展標註請求，若擁有即時計算的輕量特徵，直接寫入 training_data.csv
    約束：不觸發任何下載或重計算，零阻塞
    """
    data = request.json or {}
    url = data.get('video_url', '')

    # 允許空URL（擴展端可能僅提供特徵），但建議傳入URL用於去重
    ts = datetime.utcnow().isoformat()
    data.setdefault('timestamp', ts)

    # 正規化最小欄位
    data.setdefault('label', '')
    data.setdefault('author_id', '')
    data.setdefault('reason', '')

    # 確保所有欄位存在
    row = {k: data.get(k, '') for k in TRAINING_HEADER}
    with _buffer_lock:
        _buffer_features.append(row)

    return jsonify({
        'status': 'queued',
        'file': str(TRAINING_FILE),
        'buffer_features': len(_buffer_features)
    })


@app.route('/api/frames', methods=['POST'])
def frames():
    data = request.json or {}
    url = data.get('video_url', '')
    ts = data.get('timestamp', datetime.utcnow().isoformat())
    label = data.get('label', '')
    fps = data.get('fps', '')
    duration = data.get('duration', '')
    resolution = data.get('resolution', '')
    frames_b64 = data.get('frames', [])

    def _decode(b64str):
        try:
            if b64str.startswith('data:'):
                b64str = b64str.split(',')[1]
            raw = base64.b64decode(b64str)
            import numpy as np, cv2
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    imgs = []
    for s in frames_b64[:10]:
        im = _decode(s)
        if im is not None:
            imgs.append(im)

    mfp = 50.0
    fa = 50.0
    sna = 50.0
    pvd = 50.0
    tn = 50.0
    tf = 50.0
    try:
        import numpy as np, cv2
        if imgs:
            vars_ = []
            for im in imgs:
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                lap = cv2.Laplacian(gray, cv2.CV_64F)
                vars_.append(float(lap.var()))
            avg_var = float(np.mean(vars_)) if vars_ else 0.0
            tn = 15.0 if avg_var > 200 else (75.0 if avg_var < 50 else 50.0)

            mags = []
            for im in imgs[:5]:
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                f = np.fft.fft2(gray)
                mag = np.log(np.abs(f) + 1e-6)
                mags.append(float(np.mean(mag)))
            avg_mag = float(np.mean(mags)) if mags else 0.0
            fa = 80.0 if avg_mag > 8.5 else 50.0

            discons = []
            prev = None
            for im in imgs:
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                if prev is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
                    discons.append(float(np.percentile(mag, 95)))
                prev = gray
            p95 = float(np.mean(discons)) if discons else 0.0
            pvd = 70.0 if p95 > 2.0 else 50.0
    except Exception as e:
        logging.warning(f"[FRAMES] Fallback features due to error: {e}")

    row = {k: '' for k in TRAINING_HEADER}
    row.update({
        'video_url': url,
        'label': label,
        'timestamp': ts,
        'model_fingerprint': mfp,
        'frequency_analysis': fa,
        'sensor_noise': sna,
        'physics_violation': pvd,
        'texture_noise': tn,
        'text_fingerprint': tf,
        'bitrate': '',
        'fps': fps,
        'duration': duration,
        'resolution': resolution,
        'author_id': '',
        'reason': 'frames'
    })

    with _buffer_lock:
        _buffer_features.append(row)
        if url and url not in _hydrate_pending:
            _hydrate_queue.append(url)
            _hydrate_pending.add(url)

    return jsonify({'status': 'queued', 'queued_frames': len(frames_b64)})


@app.route('/api/stats', methods=['GET'])
def stats():
    """統計信息"""
    # 統計真實/AI數量
    real_count = 0
    ai_count = 0

    if DATASET_FILE.exists():
        with open(DATASET_FILE, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                if row['label'] == '0':
                    real_count += 1
                elif row['label'] == '1':
                    ai_count += 1

    training_count = 0
    if TRAINING_FILE.exists():
        with open(TRAINING_FILE, 'r', encoding='utf-8-sig') as f:
            training_count = max(0, sum(1 for _ in f) - 1)

    return jsonify({
        'total': len(loaded_urls),
        'real': real_count,
        'ai': ai_count,
        'dataset_file': str(DATASET_FILE),
        'training_file': str(TRAINING_FILE),
        'training_count': training_count,
        'buffer_labels': len(_buffer_labels),
        'buffer_features': len(_buffer_features),
        'hydrate_pending': len(_hydrate_pending),
        'download_dir': str(DOWNLOADS_DIR),
        'ready_for_processing': len(loaded_urls) >= 10
    })


@app.route('/api/commit', methods=['POST'])
def commit():
    """強制立即落盤（用於刷完一段後立刻看到更新）。"""
    _flush_once()
    # 返回最新統計
    training_count = 0
    if TRAINING_FILE.exists():
        with open(TRAINING_FILE, 'r', encoding='utf-8-sig') as f:
            training_count = max(0, sum(1 for _ in f) - 1)
    xlsx_path = BASE_DIR / 'training_data.xlsx'
    xlsx_rows = None
    try:
        df = pd.read_csv(TRAINING_FILE, encoding='utf-8-sig')
        df.to_excel(xlsx_path, index=False)
        xlsx_rows = int(len(df))
    except Exception as e:
        return jsonify({
            'status': 'flushed',
            'training_count': training_count,
            'buffer_labels': len(_buffer_labels),
            'buffer_features': len(_buffer_features),
            'xlsx_file': str(xlsx_path),
            'error': str(e)
        })
    return jsonify({
        'status': 'flushed',
        'training_count': training_count,
        'buffer_labels': len(_buffer_labels),
        'buffer_features': len(_buffer_features),
        'xlsx_file': str(xlsx_path),
        'rows': xlsx_rows
    })


@app.route('/api/export', methods=['POST'])
def export():
    """把CSV導出為Excel（.xlsx），方便你在WPS/Excel中查看最新內容）。"""
    try:
        df = pd.read_csv(TRAINING_FILE, encoding='utf-8-sig')
        xlsx_path = BASE_DIR / 'training_data.xlsx'
        df.to_excel(xlsx_path, index=False)
        return jsonify({'status': 'ok', 'xlsx_file': str(xlsx_path), 'rows': int(len(df))})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})


if __name__ == '__main__':
    initialize()
    print("[Aigis] Fast Labeling Server (Phase 1 Only)")
    print("[API] http://127.0.0.1:5000/api/label")
    print("[API] http://127.0.0.1:5000/api/feature")
    print("[Mode] Record-only (no download/analysis)")
    print("")
    print("Usage:")
    print("  1. Label videos on TikTok (press left/right)")
    print("  2. When done, type '執行' in chat to process all")
    print("")
    # 啟動批次落盤與下載背景執行緒
    if not _flusher_started:
        threading.Thread(target=_flusher_loop, daemon=True).start()
        threading.Thread(target=_hydrate_loop, daemon=True).start()
        _flusher_started = True

    app.run(host='127.0.0.1', port=5000, debug=False)


