# Aigis 使用指南

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
