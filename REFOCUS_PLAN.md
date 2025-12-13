# AI Detection System - Refocused Plan
# AI 檢測系統 - 重新聚焦計劃

## 📋 項目範圍澄清

### ✅ 本項目範圍（AI Detection）
1. **AI 檢測升級版**
   - 基於白皮書 Part 1-7
   - 沙皇炸彈三階段架構（Stage 1-3）
   - XGBoost Ensemble 決策引擎（Part 6-7）
   - Blue Team Defense 機制

2. **人眼輔助學習功能**
   - 人工標註介面
   - 標註數據收集
   - 持續訓練機制
   - 反饋循環

3. **資源控制**
   - 在整條 AI 流水線中占用合理資源
   - 輕量化設計
   - 快速執行（< 5秒/視頻）

### ❌ 不屬於本項目（下一個項目）
1. Tier 0-7 智能流水線
2. Smart Downgrade 系統
3. 多級調度器
4. API Gateway
5. Dashboard 系統

---

## 🎯 核心設計原則

### 1. 沙皇炸彈三階段（97% 物理純度）

```
┌─────────────────────────────────────────┐
│ Stage 1: PRIMARY (物理不可偽造層)       │
│ - Sensor Noise Authenticator            │
│ - Physics Violation Detector            │
│ - Frequency Analyzer                    │
│ - Texture Noise Detector                │
│ 權重: 40%                               │
└─────────────────────────────────────────┘
              ↓ 級聯放大
┌─────────────────────────────────────────┐
│ Stage 2: SECONDARY (生物力學層)         │
│ - Heartbeat Detector                    │
│ - Blink Dynamics Analyzer               │
│ - Lighting Geometry Checker             │
│ 權重: 30%                               │
└─────────────────────────────────────────┘
              ↓ 級聯放大
┌─────────────────────────────────────────┐
│ Stage 3: TERTIARY (數學結構層)          │
│ - Model Fingerprint Detector            │
│ - Text Fingerprinting                   │
│ - Semantic Stylometry                   │
│ - AV Sync Verifier                      │
│ - Metadata Extractor                    │
│ 權重: 30%                               │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ XGBoost Ensemble Brain                  │
│ - 25維特徵向量                          │
│ - SHAP 可解釋性                         │
│ - 最終決策                              │
└─────────────────────────────────────────┘
```

### 2. 猛禽3原則（極簡高效）

- **移除冗餘**: 簡化 autotesting.py 的 800+行邏輯
- **統一接口**: 所有模組統一 `detect(video_path) -> float` 接口
- **快速執行**: < 5秒/視頻（單視頻檢測）
- **輕量資源**: CPU < 2核心，RAM < 2GB

### 3. 第一性原理（物理不可偽造）

- **Stage 1**: 100% 基於物理特性
- **Stage 2**: 90% 基於生物特性
- **Stage 3**: 70% 基於數學特性
- **總體**: 97% 物理/生物純度

---

## 🧠 人眼輔助學習架構

### 核心概念

**人類標註 + 機器學習持續循環**

```
┌─────────────────────────────────────────────────────┐
│ 1. AI 檢測系統輸出結果                              │
│    - AI Probability: 0-100                          │
│    - Confidence: 0-1                                │
│    - SHAP 解釋                                      │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ 2. 灰色地帶篩選（20 < AI_P < 60）                  │
│    - 自動標記為「需要人工復審」                     │
│    - 加入人工標註佇列                               │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ 3. 人眼標註介面                                     │
│    - 播放視頻                                       │
│    - 顯示 AI 預測 + SHAP 解釋                       │
│    - 人類專家標註：Real / AI / Uncertain            │
│    - 信心等級：1-5                                  │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ 4. 標註數據存儲                                     │
│    - SQLite 數據庫                                  │
│    - 包含：視頻特徵、AI預測、人類標註、時間戳      │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ 5. 持續訓練機制                                     │
│    - 每收集 100 條標註 → 重訓練 XGBoost             │
│    - A/B 測試：新模型 vs 舊模型                     │
│    - 性能提升 → 部署新模型                          │
└─────────────────────────────────────────────────────┘
              ↓ 循環
          (回到步驟1)
```

### 人眼標註介面設計

```python
# 簡化的 CLI 介面（MVP）
class HumanAnnotator:
    def annotate_video(self, video_path, ai_result):
        """
        人工標註單個視頻

        Args:
            video_path: 視頻路徑
            ai_result: AI 檢測結果

        Returns:
            annotation: 人類標註結果
        """
        print(f"\n{'='*70}")
        print(f"視頻: {os.path.basename(video_path)}")
        print(f"AI 預測: {ai_result.ai_probability:.1f}% (置信度: {ai_result.confidence:.2f})")
        print(f"Top 3 原因:")
        for feature, score in ai_result.top_reasons[:3]:
            print(f"  - {feature}: {score:.1f}")
        print(f"{'='*70}")

        # 播放視頻（使用系統默認播放器）
        os.system(f'start {video_path}')  # Windows

        # 人類標註
        label = input("\n標註結果 (r=Real, a=AI, u=Uncertain): ").lower()
        confidence = int(input("信心等級 (1-5): "))
        notes = input("備註（可選）: ")

        return {
            'video_path': video_path,
            'ai_prediction': ai_result.ai_probability,
            'human_label': label,
            'human_confidence': confidence,
            'notes': notes,
            'timestamp': time.time()
        }
```

### 持續訓練流程

```python
class ContinuousTrainer:
    def check_and_retrain(self):
        """
        檢查標註數量，決定是否重訓練
        """
        # 從數據庫加載標註
        annotations = self.load_annotations()

        # 過濾高質量標註（信心等級 >= 4）
        high_quality = [a for a in annotations if a['human_confidence'] >= 4]

        # 檢查是否達到重訓練閾值
        if len(high_quality) >= 100 and len(high_quality) % 100 == 0:
            logging.info("達到重訓練閾值，開始訓練新模型...")

            # 準備訓練數據
            X, y = self.prepare_training_data(high_quality)

            # 訓練新模型
            new_model = self.train_xgboost(X, y)

            # A/B 測試
            improvement = self.ab_test(new_model, self.current_model)

            # 如果性能提升，部署新模型
            if improvement > 0.02:  # 2% 提升
                self.deploy_model(new_model)
                logging.info(f"部署新模型，性能提升: {improvement*100:.1f}%")
```

---

## 📊 資源占用控制

### 在 AI 流水線中的定位

假設整條 AI 流水線包含：
1. **視頻採集**（5%資源）
2. **預處理**（10%資源）
3. **AI檢測**（本項目，目標20%資源）
4. **後處理**（15%資源）
5. **分發/存儲**（50%資源）

### 資源預算

| 資源 | 配額 | 實際使用 | 優化策略 |
|------|------|---------|---------|
| CPU | 2-4核心 | 單視頻2核心 | 並行處理時不超過4核心 |
| RAM | < 2GB | 單視頻 < 500MB | 流式處理，不全量載入 |
| GPU | 可選 | Tier 6時使用 | 批量推理，最大化利用率 |
| 磁碟I/O | 適中 | 順序讀取 | 避免隨機訪問 |
| 執行時間 | < 5秒/視頻 | 目標 3秒 | 模組優化 + 並行 |

### 輕量化策略

1. **減少幀採樣**：300幀 → 100幀（精度損失 < 1%）
2. **懶加載模組**：需要時才載入
3. **特徵緩存**：重複視頻不重新計算
4. **早停機制**：絕對判定時跳過後續模組

---

## 🔧 簡化後的系統架構

### 核心流程（單視頻）

```python
def detect_video_simple(video_path: str) -> DetectionResult:
    """
    簡化的檢測流程（移除流水線複雜性）

    1. 執行 Stage 1-3 檢測（並行）
    2. XGBoost 融合決策
    3. 返回結果
    """
    # Stage 1: 物理層（並行執行）
    stage1_scores = execute_stage_parallel(video_path, STAGE_1_MODULES)
    stage1_avg = np.mean(list(stage1_scores.values()))

    # 級聯放大
    amplification_2 = 1.2 if stage1_avg >= 75 else (0.8 if stage1_avg <= 25 else 1.0)

    # Stage 2: 生物層（並行執行 + 放大）
    stage2_scores = execute_stage_parallel(video_path, STAGE_2_MODULES, amplification_2)
    stage2_avg = np.mean(list(stage2_scores.values()))

    # 級聯放大
    avg_12 = (stage1_avg + stage2_avg) / 2.0
    amplification_3 = 1.15 if avg_12 >= 70 else (0.85 if avg_12 <= 30 else 1.0)

    # Stage 3: 數學層（並行執行 + 放大）
    stage3_scores = execute_stage_parallel(video_path, STAGE_3_MODULES, amplification_3)

    # 合併所有分數
    all_scores = {**stage1_scores, **stage2_scores, **stage3_scores}

    # XGBoost 最終決策
    metadata = get_video_metadata(video_path)
    xgb_result = xgboost_ensemble.predict(all_scores, metadata)

    return DetectionResult(
        ai_probability=xgb_result.ai_probability,
        confidence=xgb_result.confidence,
        stage_scores={
            'stage1': stage1_avg,
            'stage2': stage2_avg,
            'stage3': stage3_avg
        },
        shap_values=xgb_result.shap_values,
        top_reasons=xgb_result.top_reasons
    )
```

### 檔案結構（簡化）

```
ai testing/
├── autotesting_v3.py               # 新的簡化總控（本次創建）
├── core/
│   ├── xgboost_ensemble.py         # XGBoost 引擎（已存在）
│   ├── human_annotator.py          # 人眼標註系統（新建）
│   └── continuous_trainer.py       # 持續訓練（新建）
├── modules/                        # 12個檢測模組（已存在）
├── data/
│   └── annotations.db              # 標註數據庫（SQLite）
├── models/
│   ├── xgboost_v1.pkl              # 當前模型
│   └── xgboost_v2.pkl              # 新訓練模型
└── output/                         # 檢測結果輸出
```

---

## ✅ 下一步行動

1. **創建簡化版總控** (`autotesting_v3.py`)
   - 移除流水線複雜性
   - 聚焦三階段檢測
   - 資源控制

2. **實現人眼標註系統** (`core/human_annotator.py`)
   - CLI 標註介面
   - SQLite 數據存儲
   - 標註佇列管理

3. **實現持續訓練機制** (`core/continuous_trainer.py`)
   - 數據加載與準備
   - XGBoost 重訓練
   - A/B 測試與部署

4. **優化模組性能**
   - 減少幀採樣
   - 並行執行優化
   - 目標：< 3秒/視頻

5. **測試與驗證**
   - 資源占用測試
   - 精度驗證
   - 人眼標註流程測試

---

**聚焦原則**: 簡單、快速、可擴展，為未來的 AI 流水線項目預留接口。
