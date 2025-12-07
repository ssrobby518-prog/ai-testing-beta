# Blue Team Defense System - 快速開始

## ⚡ 5分鐘快速部署

### 1. 安裝最小依賴（核心功能）

```bash
pip install opencv-python numpy pandas openpyxl pymediainfo
```

### 2. 運行藍隊系統

```bash
# 將視頻放入 input/ 目錄
cp your_video.mp4 input/

# 運行檢測
python autotesting_blue_team.py

# 查看結果
ls output/blue_team_report_*.xlsx
```

**就這麼簡單！** 系統會自動：
- ✅ 加載所有可用模組
- ✅ 對缺失依賴自動fallback
- ✅ 生成Excel報告

---

## 🔥 完整功能部署

### 安裝所有依賴

```bash
# 核心功能
pip install opencv-python numpy pandas openpyxl pymediainfo

# Phase I: 面部剛性檢測
pip install mediapipe

# Phase II: 頻域CNN
pip install torch torchvision

# Phase III: XGBoost決策
pip install xgboost shap
```

### 運行測試

```bash
# 測試所有新模組
python -m pytest tests/

# 或手動測試
python autotesting_blue_team.py
```

---

## 📊 查看結果

### Excel報告格式

```
blue_team_report_video_mp4.xlsx
┌─────────────────┬───────────┬────────────────┬──────────────┬──────────────┐
│ File Path       │ Timestamp │ AI Probability │ Threat Level │ Decision ... │
├─────────────────┼───────────┼────────────────┼──────────────┼──────────────┤
│ input/test.mp4  │ 20251207  │ 85.3           │ KILL_ZONE    │ XGBoost      │
└─────────────────┴───────────┴────────────────┴──────────────┴──────────────┘
```

### 威脅等級解讀

- **SAFE_ZONE** (AI_P ≤ 20%): 真實視頻，通過
- **GRAY_ZONE** (20% < AI_P ≤ 60%): 可疑，建議人工審核
- **KILL_ZONE** (AI_P > 60%): AI視頻，攔截

---

## 🛠️ 進階配置

### 啟用/禁用特定模組

編輯 `autotesting_blue_team.py`:

```python
BLUE_TEAM_MODULES = {
    'facial_rigidity_analyzer': {
        'enabled': False,  # 禁用（如果MediaPipe安裝失敗）
        'weight': 2.5,
        'fallback': 50.0
    },
    # ...
}
```

### 調整檢測閾值

```python
# 更嚴格（減少誤報）
if ai_prob <= 15:  # 原為 20
    threat_level = "SAFE_ZONE"
elif ai_prob <= 70:  # 原為 60
    threat_level = "GRAY_ZONE"
```

---

## 🔬 對比測試（舊 vs 新）

```bash
# 運行舊系統
python autotesting.py
mv output/report_video_mp4.xlsx output/report_old.xlsx

# 運行藍隊系統
python autotesting_blue_team.py
mv output/blue_team_report_video_mp4.xlsx output/report_new.xlsx

# 對比
python -c "
import pandas as pd
old = pd.read_excel('output/report_old.xlsx')
new = pd.read_excel('output/report_new.xlsx')
print('Old AI_P:', old['AI Probability'].values[0])
print('New AI_P:', new['AI Probability'].values[0])
"
```

---

## ❓ 常見問題

### Q: 沒有GPU，Spectral CNN能用嗎？
**A**: 可以！CPU推理稍慢（~2秒/視頻），但完全可用。

### Q: MediaPipe安裝失敗怎麼辦？
**A**: 系統會自動跳過 Facial Rigidity Analyzer，使用其他11個模組。

### Q: 沒有訓練XGBoost模型怎麼辦？
**A**: 系統自動使用規則引擎（加權平均），效果接近XGBoost。

### Q: 如何批量處理？
**A**: 將所有視頻放入 `input/` 目錄，系統自動批量處理。

```bash
cp /path/to/videos/*.mp4 input/
python autotesting_blue_team.py
```

---

## 🎯 下一步

1. **收集數據訓練模型** → [訓練指南](./BLUE_TEAM_UPGRADE_GUIDE.md#模型訓練)
2. **理解第一性原理** → [設計文檔](./BLUE_TEAM_UPGRADE_GUIDE.md#第一性原理)
3. **開發自定義模組** → [貢獻指南](./BLUE_TEAM_UPGRADE_GUIDE.md#貢獻)

---

**藍隊口號**: "物理不可偽造，數學無所遁形"
**設計原則**: 沙皇炸彈 × 猛禽3引擎
