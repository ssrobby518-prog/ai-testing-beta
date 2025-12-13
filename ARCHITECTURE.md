# TSAR-RAPTOR AI Detection System - Architecture Design
# 沙皇-猛禽 AI檢測系統 - 架構設計文檔

## 設計原則 (Design Principles)

### 1. 第一性原理 (First Principles)
- **物理隨機性不可偽造**: AI無法完美複製傳感器噪聲、光學缺陷
- **生物週期性唯一**: 心跳變異、呼吸節律具有個體特徵
- **統計指紋不可消除**: 模型殘差、頻域異常總會留下痕跡

### 2. 沙皇炸彈原則 (Tsar Bomba - 97% Thermonuclear Purity)
```
三階段輻射內爆設計:
┌─────────────────────────────────────────────────────────┐
│ Primary Fission (初級裂變)                              │
│ → 物理不可偽造層                                        │
│    - 傳感器噪聲 (Sensor Noise)                          │
│    - 物理違規 (Physics Violation)                       │
│    - 頻域異常 (Frequency Anomaly)                       │
│    - 紋理噪聲 (Texture Noise)                           │
│ 能量輸出: 40% → 點燃 Secondary                          │
└─────────────────────────────────────────────────────────┘
              ↓ Radiation Implosion (輻射內爆)
┌─────────────────────────────────────────────────────────┐
│ Secondary Fusion (次級聚變)                             │
│ → 生物力學層                                            │
│    - 心跳模式 (Heartbeat Pattern)                       │
│    - 眨眼動力學 (Blink Dynamics)                        │
│    - 光照幾何 (Lighting Geometry)                       │
│ 能量輸出: 30% → 點燃 Tertiary                           │
└─────────────────────────────────────────────────────────┘
              ↓ Radiation Implosion (輻射內爆)
┌─────────────────────────────────────────────────────────┐
│ Tertiary Fusion (三級聚變)                              │
│ → 數學結構層                                            │
│    - 模型指紋 (Model Fingerprint)                       │
│    - 文本指紋 (Text Pattern)                            │
│    - 語義風格 (Semantic Style)                          │
│    - 音視頻同步 (AV Sync)                               │
│    - 元數據 (Metadata)                                  │
│ 能量輸出: 30%                                           │
└─────────────────────────────────────────────────────────┘
              ↓ Thermonuclear Yield (熱核當量)
        97% Fusion Purity (97% 聚變純度)
```

### 3. 猛禽3原則 (Raptor 3 - "No Part is the Best Part")
- **極致推重比**: 280tf推力 / 1525kg = 0.183 (檢測精度/執行時間最大化)
- **激進簡化**: 800+行if-else → 50行核心邏輯
- **功能整合**: 12模組 → 3 Stage統一接口
- **350bar壓力**: < 5秒/視頻執行時間
- **測試驅動**: 10+ tests/week 快速迭代

## 機台性能規格 (Machine Specs)

```yaml
CPU: AMD Ryzen 9 9950X (16C/32T)
  - 最大加速: 5.7GHz
  - 可並行處理 32 個檢測任務
  - 策略: 批量處理多視頻，充分利用多核

RAM: 128GB DDR5-5600
  - 頻寬: 89.6 GB/s
  - 可同時載入 64+ 個視頻到記憶體
  - 策略: 預載入視頻，減少I/O等待

SSD #1: Samsung 990 PRO 2TB (Gen4)
  - 讀取: 7.45 GB/s
  - 用途: 系統盤 + 模組代碼

SSD #2: Crucial T705 4TB (Gen5)
  - 讀取: 12.6 GB/s
  - 用途: 視頻緩存 + 檢測結果
  - 策略: 批量I/O，減少隨機訪問

GPU: ASUS RTX 5090 TUF Gaming OC 32GB
  - CUDA Cores: 21760
  - Tensor Cores: 680 (Gen5)
  - 記憶體頻寬: 1792 GB/s
  - 用途:
    - CNN頻域分類器 (ResNet-18)
    - MediaPipe 468點骨骼追蹤
    - XGBoost GPU加速推理
  - 策略: 批量推理，最大化GPU利用率

PSU: Corsair 1500W SHIFT AXi ATX 3.1
  - 充足供電，支持GPU滿載
```

## Tier 0-7 流水線架構 (基於 Part 10-12)

### Tier 0: 超快速預篩選 (Ultra-Fast Pre-screening)
**目標**: < 0.5秒/視頻，過濾99%明顯真實視頻
**方法**:
- Bitrate 快速檢查（社交媒體 vs 專業製作）
- 文件元數據指紋（是否包含AI工具簽名）
- 快速幀採樣（10幀）+ 極簡特徵

**決策**:
- **絕對真實** → PASS，跳過後續Tier
- **可疑** → 進入 Tier 1

**資源**: 1 CPU核心，無GPU

---

### Tier 1: 快速物理檢測 (Fast Physical Check)
**目標**: < 1秒/視頻，檢測物理層異常
**方法**:
- Sensor Noise (輕量版，100幀採樣)
- Frequency Analysis (快速DCT)

**決策**:
- **絕對AI** (分數 >= 90) → 直接 BLOCK，跳過後續Tier
- **絕對真實** (分數 <= 15) → PASS，跳過後續Tier
- **灰色地帶** → 進入 Tier 2

**資源**: 2 CPU核心，無GPU

---

### Tier 2: 完整物理層檢測 (Full Physical Layer)
**目標**: < 2秒/視頻
**方法**:
- **Stage 1 (Primary) 完整執行**:
  - Sensor Noise Authenticator (完整版)
  - Physics Violation Detector
  - Frequency Analyzer (完整頻譜)
  - Texture Noise Detector

**決策**:
- **Stage 1 分數 >= 85** → 高度可疑，進入 Tier 5 (跳過3-4)
- **Stage 1 分數 <= 20** → 高度真實，PASS
- **中等分數** → 進入 Tier 3

**資源**: 4 CPU核心，無GPU

---

### Tier 3: 生物層檢測 (Biological Layer)
**目標**: < 1秒/視頻
**方法**:
- **Stage 2 (Secondary) 執行**:
  - Heartbeat Detector
  - Blink Dynamics Analyzer
  - Lighting Geometry Checker

**級聯放大**:
- 如果 Tier 2 (Stage 1) 分數高，放大 Stage 2 權重 × 1.2
- 如果 Tier 2 (Stage 1) 分數低，抑制 Stage 2 權重 × 0.8

**決策**:
- **Stage 1+2 平均 >= 80** → 進入 Tier 5 (跳過 Tier 4)
- **Stage 1+2 平均 <= 25** → PASS
- **中等** → 進入 Tier 4

**資源**: 3 CPU核心，無GPU

---

### Tier 4: 數學層檢測 (Mathematical Layer)
**目標**: < 1.5秒/視頻
**方法**:
- **Stage 3 (Tertiary) 執行**:
  - Model Fingerprint Detector
  - Text Fingerprinting
  - Semantic Stylometry
  - AV Sync Verifier
  - Metadata Extractor

**級聯放大**:
- 如果 Stage 1+2 平均 >= 70，放大 Stage 3 權重 × 1.15
- 如果 Stage 1+2 平均 <= 30，抑制 Stage 3 權重 × 0.85

**決策**:
- **三階段平均 >= 75** → 進入 Tier 5 (XGBoost終裁)
- **三階段平均 <= 30** → PASS
- **灰色地帶 (30-75)** → 進入 Tier 5

**資源**: 5 CPU核心，無GPU

---

### Tier 5: XGBoost Ensemble 終裁 (Final Adjudication)
**目標**: < 0.5秒/視頻
**方法**:
- XGBoost Ensemble Brain
- 輸入: 12模組分數 + 視頻元數據 (25維特徵)
- 輸出: AI Probability [0, 100]
- SHAP Values (可解釋性)

**決策**:
- **AI_P >= 60** → KILL_ZONE → BLOCK
- **20 < AI_P < 60** → GRAY_ZONE → FLAG
- **AI_P <= 20** → SAFE_ZONE → PASS

**資源**: 2 CPU核心，可選GPU加速

---

### Tier 6: 深度學習終極驗證 (Deep Learning Verification) - 可選
**目標**: < 3秒/視頻
**方法**:
- ResNet-18 頻域分類器 (GPU)
- MediaPipe 468點面部骨骼追蹤 (GPU)
- CNN紋理分析 (GPU)

**何時觸發**:
- Tier 5 結果為 GRAY_ZONE
- 且置信度 < 0.7

**決策**:
- 最終判定

**資源**: 1 CPU核心 + GPU滿載

---

### Tier 7: 人工復審佇列 (Human Review Queue)
**目標**: 異步處理
**方法**:
- 將 GRAY_ZONE 視頻加入人工復審佇列
- 收集人類標註結果
- 用於模型持續訓練

**資源**: 無計算資源

---

## Smart Downgrade (智能降級) - Part 10

**核心思想**: 早停機制（Early Stopping）

```python
if tier_0_result == "ABSOLUTE_REAL":
    return PASS  # 節省 99% 計算資源

if tier_1_result == "ABSOLUTE_AI":
    return BLOCK  # 節省 80% 計算資源

if tier_2_score >= 85:
    skip_to_tier_5()  # 跳過 Tier 3-4

# 只有灰色地帶才走完全部流程
```

**預期收益**:
- 60% 視頻在 Tier 0-1 就完成判定
- 平均執行時間從 5秒 → 2秒

---

## Dual-Engine Integration (雙引擎整合) - Part 11

### Engine 1: Rule-Based (規則引擎)
- 基於第一性原理的硬規則
- 絕對判定（MFP >= 88 → 98% AI）
- 零誤判（保護真實視頻）

### Engine 2: ML-Based (機器學習引擎)
- XGBoost Ensemble
- 學習複雜模式
- 處理灰色地帶

### 整合策略:
```python
if rule_engine.is_absolute():
    return rule_engine.result  # 信任規則
else:
    return ml_engine.result  # 使用機器學習
```

---

## 並行處理策略 (Parallelization Strategy)

### Level 1: 視頻級並行 (Video-level Parallelism)
- 16C/32T → 同時處理 16 個視頻
- 使用 Python multiprocessing
- 每個進程獨立處理一個視頻

### Level 2: 模組級並行 (Module-level Parallelism)
- 同一 Stage 內的模組並行執行
- 使用 ThreadPoolExecutor
- Stage 1: 4個模組 → 4線程並行

### Level 3: 幀級並行 (Frame-level Parallelism)
- 單個模組內部，多幀並行處理
- 使用 NumPy vectorization
- GPU批量推理

### Level 4: GPU批量推理 (GPU Batch Inference)
- 累積 32 個視頻
- 批量送入 ResNet-18
- 最大化 GPU 利用率（21760 CUDA Cores）

---

## I/O 優化策略 (I/O Optimization)

### 策略 1: 預載入 (Preloading)
```python
# 利用 128GB RAM
video_cache = {}
for video_file in all_videos:
    video_cache[video_file] = load_to_memory(video_file)
```

### 策略 2: 批量讀取 (Batch Reading)
```python
# 利用 12.6GB/s SSD
videos = batch_load_from_ssd(video_list, batch_size=32)
```

### 策略 3: 異步I/O (Async I/O)
```python
# 一邊處理一邊載入下一批
async_load_next_batch(next_video_list)
```

---

## 性能目標 (Performance Targets)

| Tier | 單視頻時間 | 命中率 | 累積通過率 |
|------|-----------|--------|-----------|
| Tier 0 | 0.5s | 40% | 40% |
| Tier 1 | 1.0s | 30% | 70% |
| Tier 2 | 2.0s | 15% | 85% |
| Tier 3 | 1.0s | 5% | 90% |
| Tier 4 | 1.5s | 5% | 95% |
| Tier 5 | 0.5s | 5% | 100% |

**平均執行時間**:
```
0.4×0.5 + 0.3×(0.5+1.0) + 0.15×(0.5+1.0+2.0) + ...
= 0.2 + 0.45 + 0.525 + 0.225 + 0.2 + 0.025
= 1.625秒/視頻
```

**批量處理吞吐量**:
```
16 並行 × (3600秒/小時) / 1.625秒
= 35,446 視頻/小時
= 850,000 視頻/天
```

---

## 總結 (Summary)

**沙皇炸彈 (Tsar Bomba)**:
- 三階段級聯放大
- 97% 物理不可偽造特性
- 無基本限制的檢測能力

**猛禽3 (Raptor 3)**:
- 極致推重比（精度/時間）
- 激進簡化（移除冗餘）
- 快速迭代（測試驅動）

**機台性能**:
- 16C/32T + 128GB + RTX 5090
- 850,000 視頻/天處理能力
- < 2秒平均執行時間

**AI 流水線 (Tier 0-7)**:
- 智能降級（早停）
- 雙引擎整合（規則+機器學習）
- 多級並行（視頻/模組/幀/GPU）
