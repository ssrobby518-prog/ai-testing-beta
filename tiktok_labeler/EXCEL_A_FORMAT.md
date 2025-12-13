# Excel A 格式說明
# Excel A (標註數據) 格式文檔

## 📊 新格式（已優化）

### 列順序（從左到右）

| 列號 | 列名 | 說明 | 示例 | 必填 |
|------|------|------|------|------|
| A | **序號** | 自動編號（1, 2, 3...） | 1 | ✅ |
| B | **影片網址** | TikTok 視頻完整URL | https://www.tiktok.com/@user/video/7123456789 | ✅ |
| C | **判定結果** | 人工判定（大寫顯示） | REAL / AI / UNCERTAIN / EXCLUDE | ✅ |
| D | **標註時間** | ISO 8601 格式時間戳 | 2025-12-12T10:30:00.123Z | ✅ |
| E | **視頻ID** | TikTok 視頻ID | 7123456789 | ✅ |
| F | **作者** | TikTok 用戶名 | @username | ⭕ |
| G | **標題** | 視頻標題/描述 | Amazing dance video #fyp | ⭕ |
| H | **點贊數** | 點贊數（帶單位） | 1.2M | ⭕ |
| I | **來源** | 標註來源 | tiktok_chrome_extension | ✅ |
| J | **版本** | 擴展版本號 | 2.0.0 | ✅ |

---

## 📋 示例數據

```
序號  影片網址                                          判定結果    標註時間                     視頻ID       作者        標題                  點贊數   來源                      版本
1     https://www.tiktok.com/@user/video/7123456789   REAL        2025-12-12T10:30:00.123Z     7123456789   @dancer     Amazing dance #fyp    1.2M     tiktok_chrome_extension   2.0.0
2     https://www.tiktok.com/@ai/video/7234567890     AI          2025-12-12T10:31:15.456Z     7234567890   @aiart      AI generated beauty   890K     tiktok_chrome_extension   2.0.0
3     https://www.tiktok.com/@user2/video/7345678901  UNCERTAIN   2025-12-12T10:32:30.789Z     7345678901   @makeup     Makeup tutorial       2.5M     tiktok_chrome_extension   2.0.0
4     https://www.tiktok.com/@movie/video/7456789012  EXCLUDE     2025-12-12T10:33:45.012Z     7456789012   @clips      Movie scene clip      3.1M     tiktok_chrome_extension   2.0.0
```

---

## 🔄 與舊格式的對比

### 舊格式（已棄用）
```
timestamp | url | video_id | author | title | likes | label | source | version
```

### 新格式（當前使用）
```
序號 | 影片網址 | 判定結果 | 標註時間 | 視頻ID | 作者 | 標題 | 點贊數 | 來源 | 版本
```

### 改進點
1. ✅ **重要信息前置**: 影片網址和判定結果在最前面，方便查看
2. ✅ **序號自動編號**: 方便追蹤和引用
3. ✅ **中文列名**: 更直觀易讀
4. ✅ **判定結果大寫**: REAL/AI/UNCERTAIN/EXCLUDE 更醒目
5. ✅ **兼容性**: 代碼自動兼容新舊格式

---

## 🎯 判定結果說明

| 判定 | 英文 | 說明 | 快捷鍵 |
|------|------|------|--------|
| **REAL** | Real | 真實視頻（人類拍攝） | ← 左箭頭 |
| **AI** | AI | AI生成視頻 | → 右箭頭 |
| **UNCERTAIN** | Uncertain | 不確定（需復審） | ↑ 上箭頭 |
| **EXCLUDE** | Exclude | 電影/動畫（排除訓練） | ↓ 下箭頭 |

---

## 💡 使用建議

### 1. Excel 查看技巧
- **凍結首行**: 選擇第2行 → 視圖 → 凍結窗格 → 凍結首行
- **自動調整列寬**: 全選 → 格式 → 自動調整列寬
- **篩選數據**: 選擇首行 → 數據 → 篩選

### 2. 快速篩選
```excel
# 只看真實視頻
判定結果 = "REAL"

# 只看AI視頻
判定結果 = "AI"

# 查看需要復審的
判定結果 = "UNCERTAIN"

# 排除電影/動畫
判定結果 <> "EXCLUDE"
```

### 3. 統計分析
```excel
# 統計各類別數量
=COUNTIF(C:C,"REAL")     # 真實視頻數
=COUNTIF(C:C,"AI")       # AI視頻數
=COUNTIF(C:C,"UNCERTAIN")# 不確定數
=COUNTIF(C:C,"EXCLUDE")  # 排除數

# 計算比例
=COUNTIF(C:C,"REAL")/COUNTA(C:C)-1  # 真實視頻占比
```

---

## 🔧 技術細節

### 1. 自動編號邏輯
```python
'序號': len(df_existing) + 1  # 當前行數 + 1
```

### 2. 判定結果大寫
```python
'判定結果': label.upper()  # real → REAL
```

### 3. 兼容性處理
```python
# 代碼自動檢測列名
url_col = '影片網址' if '影片網址' in df.columns else 'url'
label_col = '判定結果' if '判定結果' in df.columns else 'label'
```

---

## 📂 文件位置

```
data/
└── tiktok_labels/
    └── excel_a_labels_raw.xlsx  ← Excel A 文件
```

---

## 🚀 開始使用

### 1. 啟動後端服務器
```bash
cd tiktok_labeler/backend
python server.py
```

### 2. 安裝Chrome擴展並開始標註
標註後會自動生成新格式的 Excel A

### 3. 查看Excel A
```bash
# Windows
start data/tiktok_labels/excel_a_labels_raw.xlsx

# macOS
open data/tiktok_labels/excel_a_labels_raw.xlsx

# Linux
xdg-open data/tiktok_labels/excel_a_labels_raw.xlsx
```

---

## ✅ 驗證格式

打開 Excel A 後，應該看到：
1. ✅ 第一列是「序號」（1, 2, 3...）
2. ✅ 第二列是「影片網址」（完整TikTok URL）
3. ✅ 第三列是「判定結果」（REAL/AI/UNCERTAIN/EXCLUDE）
4. ✅ 所有列名都是中文
5. ✅ 判定結果全部大寫

---

**設計原則**: 重要信息前置 + 中文友好 + 兼容舊格式

**最後更新**: 2025-12-12
