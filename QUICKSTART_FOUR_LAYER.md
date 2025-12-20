# 四層架構快速開始指南

**5 分鐘快速上手四層架構 AI 檢測系統**

---

## 📋 前置要求

```bash
# Python 3.8+
python --version

# 必要的庫
pip install numpy opencv-python pandas openpyxl pymediainfo
```

---

## 🚀 方法 1：最簡單的使用方式

### Step 1: 準備視頻

```bash
# 將幾個測試視頻放入 input/ 目錄
# 支持格式：.mp4, .avi, .mov, .mkv
cp your_video.mp4 input/
```

### Step 2: 運行四層系統

```bash
python autotesting_four_layer.py
```

### Step 3: 查看結果

```bash
# 查看各個 Pool 的分配結果
ls data_pools/ai_pool/          # AI生成影片（高置信度）
ls data_pools/real_pool/        # 真人拍攝影片（高置信度）
ls data_pools/disagreement_pool/ # 需要人工審核（模組分歧）
ls data_pools/excluded_pool/    # 電影/動畫（永久排除）

# 查看詳細報告
ls output/four_layer_reports/   # JSON 格式的完整分析報告
```

**就這麼簡單！** 🎉

---

## 🎯 方法 2：完整的月度循環

### 月度 1：初始訓練（100 個視頻）

```bash
# ========== 第一層：收集候選影片池 ==========

# 方法 A：從 TikTok 下載
cd tiktok_labeler/downloader
python tiktok_downloader.py \
    --url-list urls.txt \
    --output ../../input/

# 方法 B：使用本地視頻
cp ~/Downloads/*.mp4 input/

# ========== 第二層 + 第三層：自動分析與仲裁 ==========

cd ../..
python autotesting_four_layer.py

# 預期輸出：
# ✅ 總處理數: 100
# ✅ AI Pool: 45 個 (45%)
# ✅ Real Pool: 35 個 (35%)
# ⚠️  Disagreement Pool: 18 個 (18%) - 需要人工審核
# ✅ Excluded Pool: 2 個 (2%)

# ========== 第三層：人工審核 Disagreement Pool ==========

python tiktok_tinder_labeler.py

# Tinder 式快速確認（18 個視頻 ≈ 1 分鐘）
# ← (A) = REAL
# → (D) = AI
# ↑ (W) = UNCERTAIN（仍不確定，稍後再看）
# ↓ (S) = MOVIE（電影/動畫，排除）

# 確認後，視頻會自動移動到正確的 Pool

# ========== 生成機制標籤穩定集 ==========

# AI Pool: 45 + 10 (人工確認) = 55 個
# Real Pool: 35 + 8 (人工確認) = 43 個
# 總計: 98 個高質量標籤
# 人工介入比例: 18% (18/100)
```

### 月度 2：自動擴增（+100 個視頻）

```bash
# 下載新視頻
cd tiktok_labeler/downloader
python tiktok_downloader.py --output ../../input/

# 運行四層系統
cd ../..
python autotesting_four_layer.py

# 預期輸出：
# ✅ 總處理數: 100
# ✅ AI Pool: 60 個 (60%) ↑
# ✅ Real Pool: 30 個 (30%)
# ⚠️  Disagreement Pool: 8 個 (8%) ↓ (人工介入比例下降！)
# ✅ Excluded Pool: 2 個 (2%)

# 人工審核（8 個視頻 ≈ 30 秒）
python tiktok_tinder_labeler.py

# 累積訓練集：
# AI Pool: 55 + 60 + 5 = 120 個
# Real Pool: 43 + 30 + 3 = 76 個
# 總計: 196 個高質量標籤
# 人工介入比例: 13% (26/200) ↓
```

### 月度 3-12：持續收斂

```bash
# 重複月度 2 的流程

# 月度 3: 人工介入 8% (8/100)
# 月度 6: 人工介入 5% (5/100)
# 月度 12: 人工介入 <2% (2/100)

# 🎯 最終結果：
# - 累積訓練集: 10,000+ 高質量標籤
# - 人工介入比例: <2%
# - 數據純度: 99%+
```

---

## 🔍 理解輸出結果

### 1. 終端輸出

```
================================================================================
處理進度: 1/10 - test_video.mp4
================================================================================

[第二層] 開始分析: test_video.mp4
[第二層] Bitrate: 1.25 Mbps
[第二層] Face Presence: 0.85
[第二層] Static Ratio: 0.12
[第二層] frequency_analyzer: 78.5
[第二層] model_fingerprint_detector: 65.2
[第二層] physics_violation_detector: 82.1
...
[第二層] AI概率: 76.3%
[第二層] 生成機制: ai
[第二層] 置信度: 0.88

[第三層] 執行資料仲裁...
[第三層] 仲裁結果: ai_pool
[第三層] 決策理由: 高置信度AI（一致性=0.88，AI_P=76.3）

[第四層] 執行經濟行為分類...
[第四層] 分類結果: 未分類

[四層系統] 處理完成

================================================================================
處理統計
================================================================================
總處理數: 10
AI Pool: 6 (60.0%)
Real Pool: 3 (30.0%)
Disagreement Pool: 1 (10.0%)
Excluded Pool: 0 (0.0%)
================================================================================

⚠️  發現 1 個需要人工審核的視頻
   請查看: data_pools/disagreement_pool
   使用 Tinder 式標註工具進行快速確認
```

### 2. JSON 報告（詳細）

```json
{
  "candidate_video": {
    "file_name": "test_video.mp4",
    "bitrate": 1250000,
    "fps": 30.0,
    ...
  },
  "generation_analysis": {
    "generation_mechanism": "ai",
    "ai_probability": 76.3,
    "module_scores": {
      "frequency_analyzer": 78.5,
      "model_fingerprint_detector": 65.2,
      ...
    },
    "confidence": 0.88
  },
  "arbitration": {
    "pool_assignment": "ai_pool",
    "module_consensus_score": 0.88,
    "human_review_required": false,
    "decision_rationale": "高置信度AI（一致性=0.88，AI_P=76.3）"
  },
  "economic_classification": {
    "economic_behavior": "未分類",
    ...
  }
}
```

### 3. Excel 匯總報告

打開 `output/four_layer/四層分析匯總_YYYYMMDD_HHMMSS.xlsx`：

| 文件名 | AI概率 | 生成機制 | 置信度 | 資料池分配 | 需要人工審核 | 仲裁理由 |
|-------|-------|---------|--------|-----------|------------|---------|
| video1.mp4 | 85.2 | ai | 0.92 | ai_pool | False | 高置信度AI |
| video2.mp4 | 15.3 | real | 0.89 | real_pool | False | 高置信度REAL |
| video3.mp4 | 55.7 | uncertain | 0.45 | disagreement_pool | True | 模組分歧 |

---

## 🛠️ 常見場景

### 場景 1：批量處理 TikTok 視頻

```bash
# Step 1: 準備 URL 列表
echo "https://www.tiktok.com/@user/video/123" > urls.txt
echo "https://www.tiktok.com/@user/video/456" >> urls.txt
# ... 添加更多 URL

# Step 2: 批量下載
cd tiktok_labeler/downloader
python tiktok_downloader.py --url-list urls.txt --output ../../input/

# Step 3: 自動分析
cd ../..
python autotesting_four_layer.py

# Step 4: 審核 Disagreement Pool（如果有）
python tiktok_tinder_labeler.py
```

### 場景 2：只想快速檢測單個視頻

```bash
# 使用傳統系統（更快，但無四層架構優勢）
cp your_video.mp4 input/
python autotesting.py

# 查看結果
cat output/report_your_video_mp4.xlsx
```

### 場景 3：分析 AI vs Real 差異（優化模組）

```bash
# 從 AI Pool 和 Real Pool 提取特徵
cd tiktok_labeler/analyzer
python feature_extractor.py --input ../../data_pools/ai_pool/
python feature_extractor.py --input ../../data_pools/real_pool/

# 分析差異
python big_data_analyzer.py

# 查看優化建議
# 打開 output/feature_analysis.xlsx
# 根據建議更新 core/generation_analyzer.py 中的模組權重
```

---

## ⚠️ 常見問題

### Q1: Disagreement Pool 的視頻太多（>30%），怎麼辦？

**原因**：模組權重可能不適合您的數據集

**解決**：
1. 人工確認前 50 個 Disagreement 視頻
2. 使用特徵分析工具找出關鍵區分特徵
3. 調整 `core/generation_analyzer.py` 中的模組權重
4. 重新運行系統

### Q2: 如何提高檢測準確率？

**方法**：
1. **收集更多訓練數據**（月度擴增）
2. **分析特徵差異**（使用 analyzer 工具）
3. **優化模組權重**（基於 Cohen's d 效應量）
4. **開發新模組**（針對發現的新特徵）

### Q3: 系統運行很慢，怎麼辦？

**優化**：
1. 減少輸入視頻數量（每次處理 10-20 個）
2. 使用更快的存儲設備（SSD）
3. 調整視頻預處理參數（減少採樣幀數）

### Q4: 想使用舊的檢測系統，怎麼辦？

**向後兼容**：
```bash
# 使用傳統系統（autotesting.py）
python autotesting.py

# 兩個系統可以並存，互不干擾
```

---

## 📚 下一步

### 學習更多

- **完整文檔**：閱讀 `README.md`
- **設計原理**：閱讀 `FOUR_LAYER_ARCHITECTURE_SUMMARY.md`
- **TikTok 工具**：閱讀 `tiktok_labeler/README.md`

### 進階使用

- **自定義仲裁閾值**：修改 `core/four_layer_system.py`
- **添加新模組**：在 `modules/` 目錄創建新檢測模組
- **開發經濟分類器**：完善 `EconomicBehaviorClassifier`

### 貢獻

- **報告問題**：在 GitHub Issues 提交
- **分享數據集**：幫助改進模組權重
- **開發新功能**：提交 Pull Request

---

**🎯 記住**：
- 第一次運行會有較多 Disagreement Pool（正常）
- 月度循環後，人工介入比例會自動下降
- 目標：12 個月後人工介入 <2%，數據純度 99%+

**🤖 Generated with Four-Layer Architecture × First Principles**
