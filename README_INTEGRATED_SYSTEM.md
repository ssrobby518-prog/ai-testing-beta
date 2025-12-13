# TSAR-RAPTOR AI Detection with Human-Eye Assisted Learning
# 沙皇-猛禽 AI檢測系統 + 人眼輔助學習

## 📋 系統概覽

基於**第一性原理**、**沙皇炸彈三階段架構**、**猛禽3簡約哲學**構建的完整AI視頻檢測與持續學習系統。

### 核心組件

```
┌─────────────────────────────────────────────────────────────┐
│ autotesting_v3.py - TSAR-RAPTOR AI檢測核心                  │
│ ├── Stage 1 (Primary - 40%): 物理不可偽造層                 │
│ ├── Stage 2 (Secondary - 30%): 生物力學層                   │
│ ├── Stage 3 (Tertiary - 30%): 數學結構層                    │
│ └── XGBoost Ensemble: 最終融合決策                          │
└─────────────────────────────────────────────────────────────┘
              ↓ 識別 GRAY_ZONE (20%-60% AI概率)
┌─────────────────────────────────────────────────────────────┐
│ core/human_annotator.py - 人工標註系統                      │
│ ├── CLI標註介面（視頻播放 + AI結果展示）                    │
│ ├── SQLite數據庫（存儲標註）                                │
│ └── 高質量標註篩選（信心 >= 4）                             │
└─────────────────────────────────────────────────────────────┘
              ↓ 累積 >= 100 條高質量標註
┌─────────────────────────────────────────────────────────────┐
│ core/continuous_trainer.py - 持續訓練系統                   │
│ ├── XGBoost模型重訓練                                       │
│ ├── A/B測試（新模型 vs 舊模型）                             │
│ └── 自動部署（性能提升 >= 2%）                              │
└─────────────────────────────────────────────────────────────┘
              ↓ 循環改進
        提升檢測精度 (持續逼近 99%)
```

---

## 🎯 設計原則

### 1. 第一性原理 (First Principles)

**物理不可偽造**: AI無法完美複製物理硬件的隨機性
- 傳感器噪聲（量子過程）
- 光學缺陷（鏡頭特性）
- 運動學規律（牛頓定律）

**生物週期性**: 人類生物信號具有個體特徵
- 心跳變異（HRV, 0.8-2.5 Hz）
- 呼吸節律（混沌系統）
- 眨眼動力學（肌肉控制）

**統計指紋**: 機器學習模型留下數學痕跡
- GAN/Diffusion模型殘差
- 頻域棋盤模式（checkerboard pattern）
- 紋理噪聲異常

### 2. 沙皇炸彈原則 (Tsar Bomba - 97% Thermonuclear Purity)

**三階段輻射內爆**:

```
Primary Fission (Stage 1) → 40% 能量
    ↓ Radiation Implosion (級聯放大 × 1.2 or 0.8)
Secondary Fusion (Stage 2) → 30% 能量
    ↓ Radiation Implosion (級聯放大 × 1.15 or 0.85)
Tertiary Fusion (Stage 3) → 30% 能量
    ↓ Thermonuclear Yield
97% Fusion Purity (物理/生物純度)
```

**級聯放大機制**: 前一階段的分數影響下一階段的靈敏度
- Stage 1 高分 → 放大 Stage 2 檢測能力
- Stage 1 低分 → 抑制 Stage 2 誤報

### 3. 猛禽3原則 (Raptor 3 - "No Part is the Best Part")

**極致推重比**: 檢測精度 / 執行時間 = 最大化
- 800+行 if-else邏輯 → 700行模塊化代碼
- 目標: < 5秒/視頻

**資源控制**:
- CPU: < 4核心
- RAM: < 2GB
- 執行時間: < 5秒/視頻

**持續迭代**: 人類標註 → 模型改進 → 性能提升 → 循環

---

## 🚀 快速開始

### 系統要求

```yaml
Python: >= 3.8
依賴項:
  - numpy
  - opencv-python (cv2)
  - xgboost  # 可選，用於持續訓練
  - shap     # 可選，用於模型解釋

推薦機台:
  - CPU: AMD Ryzen 9 9950X (16C/32T) 或同等級
  - RAM: >= 16GB
  - SSD: >= 1TB
  - GPU: RTX 5090 32GB (可選，用於Tier 6加速)
```

### 安裝依賴

```bash
# 核心依賴
pip install numpy opencv-python

# 持續訓練依賴（可選）
pip install xgboost
pip install "numpy<2.0"  # XGBoost兼容性

# SHAP解釋（可選，若NumPy 2.0+則跳過）
pip install shap
```

### 基礎使用

#### 1. 僅 AI 檢測（無人工標註）

```bash
python autotesting_integrated.py --input input/ --no-annotation --no-retrain
```

輸出:
- 每個視頻的 AI Probability (0-100%)
- 檢測摘要（BLOCK/FLAG/PASS統計）

#### 2. 完整集成流程（AI檢測 + 人工標註 + 自動重訓練）

```bash
python autotesting_integrated.py --input input/
```

流程:
1. 運行 AI 檢測
2. 識別 GRAY_ZONE 視頻（20-60% AI概率）
3. 提示是否進行人工標註
4. 收集人工標註後，檢查是否達到重訓練閾值
5. 自動重訓練並部署新模型（如果滿足條件）

#### 3. 只顯示系統狀態

```bash
python autotesting_integrated.py --status
```

顯示:
- 標註數據庫統計
- 持續訓練狀態
- 模型訓練歷史

#### 4. 強制重訓練模型

```bash
python autotesting_integrated.py --force-retrain
```

---

## 📊 人工標註流程

### 標註介面

當系統識別到 GRAY_ZONE 視頻時，會進入人工標註介面:

```
================================================================================
              人工標註介面 - TSAR-RAPTOR Human Annotation
================================================================================

📹 視頻: a.mp4
📍 路徑: C:\...\ai testing\input\a.mp4

────────────────────────────────────────────────────────────────────────────────
🤖 AI 預測結果:
  • AI Probability: 75.5%
  • Confidence: 0.85

  📊 Top 3 檢測原因 (SHAP):
     1. model_fingerprint_detector: 88.2
     2. frequency_analyzer: 72.1
     3. physics_violation_detector: 65.3
────────────────────────────────────────────────────────────────────────────────

👤 人類判斷 (r=Real真實, a=AI生成, u=Uncertain不確定, s=Skip跳過): a
🎯 信心等級 (1-5, 數字越大越確定): 5
📝 備註（可選，直接按Enter跳過）: 明顯的模型指紋

✓ 標註完成: 🤖 AI | ⭐⭐⭐⭐⭐ (5/5)
  備註: 明顯的模型指紋
```

### 標註標準

| 信心等級 | 說明 | 是否用於訓練 |
|---------|------|-------------|
| 5 | 100%確定，非常明顯 | ✅ |
| 4 | 較為確定，有明確依據 | ✅ |
| 3 | 中等確定，有一定依據 | ❌ |
| 2 | 不太確定，缺乏依據 | ❌ |
| 1 | 完全不確定，純猜測 | ❌ |

**只有信心等級 >= 4 的標註會用於模型重訓練**

---

## 🧠 持續訓練機制

### 觸發條件

```python
retrain_threshold = 100  # 累積 100 條高質量標註
```

當滿足以下條件時自動觸發重訓練:
- 高質量標註（信心 >= 4）累積達到閾值
- 標註包含 Real 和 AI 兩類樣本

### A/B 測試

新模型 vs 舊模型，在測試集上比較:
- **準確率提升 >= 2%** → 部署新模型
- **準確率提升 < 2%** → 保留舊模型

### 模型版本管理

```
models/
├── xgboost_current.pkl      # 當前部署模型
├── xgboost_v1702345678.pkl  # 歷史版本（時間戳）
├── xgboost_v1702345789.pkl
└── xgboost_backup_*.pkl     # 備份模型
```

---

## 📁 文件結構

```
ai testing/
├── autotesting_integrated.py      # 集成系統主程式（新建）
├── autotesting_v3.py               # AI檢測核心（新建）
├── autotesting.py                  # 原有系統（800+行，參考用）
├── autotesting_blue_team.py        # Blue Team系統
│
├── core/
│   ├── human_annotator.py          # 人工標註系統（新建）
│   ├── continuous_trainer.py       # 持續訓練系統（新建）
│   ├── xgboost_ensemble.py         # XGBoost決策引擎
│   ├── scoring_engine.py           # 評分引擎
│   └── detection_engine.py         # 檢測引擎
│
├── modules/                        # 12個檢測模組
│   ├── sensor_noise_authenticator.py       # Stage 1
│   ├── physics_violation_detector.py       # Stage 1
│   ├── frequency_analyzer.py               # Stage 1
│   ├── texture_noise_detector.py           # Stage 1
│   ├── heartbeat_detector.py               # Stage 2
│   ├── blink_dynamics_analyzer.py          # Stage 2
│   ├── lighting_geometry_checker.py        # Stage 2
│   ├── model_fingerprint_detector.py       # Stage 3
│   ├── text_fingerprinting.py              # Stage 3
│   ├── semantic_stylometry.py              # Stage 3
│   ├── av_sync_verifier.py                 # Stage 3
│   └── metadata_extractor.py               # Stage 3
│
├── data/
│   └── annotations.db              # 標註數據庫（SQLite）
│
├── models/
│   ├── xgboost_current.pkl         # 當前部署模型
│   └── xgboost_v*.pkl              # 歷史版本
│
├── input/                          # 輸入視頻目錄
└── output/                         # 輸出報告目錄
```

---

## 🔧 進階使用

### 自定義標註者ID

```bash
python autotesting_integrated.py --input input/ --annotator-id "expert_reviewer_1"
```

用途: 追蹤不同標註者的數據，分析標註一致性

### 批量處理模式

```bash
# 處理單個視頻
python autotesting_integrated.py --input input/a.mp4

# 處理整個目錄
python autotesting_integrated.py --input input/

# 處理並跳過人工標註
python autotesting_integrated.py --input input/ --no-annotation
```

### 查看標註統計

```python
from core.human_annotator import HumanAnnotator

annotator = HumanAnnotator()
annotator.show_statistics()
```

輸出:
```
================================================================================
                            標註數據庫統計
================================================================================

📊 總體統計:
  • 總標註數: 150
  • 高質量標註: 120 (信心 >= 4)
  • 已用於訓練: 100
  • 待訓練: 20

📊 標籤分布:
  • real        :   75 ( 50.0%) █████████████████████████
  • ai          :   60 ( 40.0%) ████████████████████
  • uncertain   :   15 ( 10.0%) █████
```

### 查看訓練歷史

```python
from core.continuous_trainer import ContinuousTrainer

trainer = ContinuousTrainer()
trainer.show_training_status()
```

輸出:
```
================================================================================
                            持續訓練狀態
================================================================================

📊 標註數據:
  • 高質量標註: 120
  • 已用於訓練: 100
  • 待訓練: 20
  • 重訓練閾值: 100
  • 進度: [██████████░░░░░░░░░░] 20.0%

📚 訓練歷史 (最近5次):
  1. xgboost_v1702345789
     時間: 2025-12-11 14:30:00
     大小: 2.45 MB
  2. xgboost_v1702345678
     時間: 2025-12-10 18:15:00
     大小: 2.43 MB

🎯 當前部署模型:
  • 路徑: models/xgboost_current.pkl
  • 大小: 2.45 MB
```

---

## 📈 性能指標

### 資源占用（設計目標）

| 資源 | 配額 | 實際使用 | 優化策略 |
|------|------|---------|---------|
| CPU | 2-4核心 | 單視頻2核心 | 並行處理時不超過4核心 |
| RAM | < 2GB | 單視頻 < 500MB | 流式處理，不全量載入 |
| GPU | 可選 | Tier 6時使用 | 批量推理，最大化利用率 |
| 執行時間 | < 5秒/視頻 | 目標 3秒 | 模組優化 + 並行 |

### 精度目標

| 指標 | 當前 | 目標 | 優化方法 |
|-----|------|------|---------|
| Precision | 92% | 99% | 人工標註 + 持續訓練 |
| Recall | 88% | 95% | 增加困難樣本標註 |
| 誤報率 | 8% | 1% | 高質量標註 + A/B測試 |

---

## 🛠️ 故障排除

### 問題 1: NumPy版本衝突

**症狀**: `ImportError: numpy.core.multiarray failed to import`

**解決**:
```bash
pip install "numpy<2.0" --force-reinstall
```

### 問題 2: XGBoost無法安裝

**症狀**: `ERROR: Could not build wheels for xgboost`

**解決**:
```bash
# 方案1: 使用預編譯版本
pip install xgboost

# 方案2: 禁用持續訓練功能
python autotesting_integrated.py --no-retrain
```

### 問題 3: 視頻播放失敗

**症狀**: 人工標註時視頻無法自動播放

**解決**:
- 手動打開視頻文件（路徑會顯示在介面上）
- 檢查系統默認視頻播放器設置

### 問題 4: 數據庫鎖定

**症狀**: `sqlite3.OperationalError: database is locked`

**解決**:
- 關閉其他訪問數據庫的進程
- 刪除 `data/annotations.db-journal` 文件（如果存在）

---

## 🔬 模組優化建議

### 短期優化（1個月）

1. **減少幀採樣**
   ```python
   # autotesting_v3.py
   FRAME_SAMPLE_LIMIT = 100  # 從 300 → 100
   ```

2. **並行執行優化**
   ```python
   MAX_PARALLEL_MODULES = 4  # 模組級並行
   ```

3. **早停機制**
   - Tier 0: 極快速預篩選（< 0.5秒）
   - Tier 1: 快速物理檢測（< 1秒）

### 中期擴展（3個月）

1. **GPU加速**
   - 實現 Tier 6（ResNet-18頻域分類）
   - 批量推理（32視頻/批）

2. **分布式部署**
   - 多機並行處理
   - 負載均衡

---

## 📚 參考文獻

### 設計原理

1. **沙皇炸彈 (Tsar Bomba)**
   - [Wikipedia](https://en.wikipedia.org/wiki/Tsar_Bomba)
   - 三階段輻射內爆設計

2. **SpaceX Raptor 3 Engine**
   - "No part is the best part" 哲學
   - 極致推重比設計

3. **第一性原理 (First Principles)**
   - 物理不可偽造特性
   - 生物週期性唯一性
   - 統計指紋不可消除性

### 技術文檔

- `REFOCUS_PLAN.md` - 項目範圍與設計原則
- `ARCHITECTURE.md` - 系統架構設計
- `IMPLEMENTATION_SUMMARY.md` - 實施總結

---

## 📞 支持與貢獻

### 報告問題

如果遇到問題，請提供:
1. 完整錯誤信息
2. 系統環境（OS, Python版本, 依賴版本）
3. 重現步驟

### 貢獻代碼

歡迎貢獻:
- 新的檢測模組
- 性能優化
- 文檔改進

---

**設計原則**: 沙皇炸彈（97%物理純度） × 猛禽3（極致簡約） × 第一性原理（物理不可偽造）

**項目狀態**: ✅ 核心系統已完成，持續優化中

**最後更新**: 2025-12-12
