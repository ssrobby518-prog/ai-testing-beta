# Excel D 格式說明
# Excel D (AI自動檢測結果) 格式文檔

## 📊 設計理念

Excel D 是 **Layer 2 AI主導自動化** 的核心數據記錄，用於：
1. 記錄 AI 檢測模組的分類結果
2. 保存關鍵特徵值供自我訓練
3. 追蹤分類信心度
4. 支持人工復審後的持續優化

**設計原則**: 第一性原理（物理不可偽造特徵） × 沙皇炸彈（海量數據） × 猛禽3（極簡高效）

---

## 📋 列順序（從左到右）

| 列號 | 列名 | 說明 | 示例 | 必填 |
|------|------|------|------|------|
| A | **序號** | 自動編號（1, 2, 3...） | 1 | ✅ |
| B | **影片網址** | TikTok 視頻完整URL | https://www.tiktok.com/@user/video/7123456789 | ✅ |
| C | **AI檢測分類** | AI自動判定（大寫顯示） | REAL / AI / NOT_SURE / 電影動畫 | ✅ |
| D | **信心度** | AI分類信心度 (0-100) | 92.5 | ✅ |
| E | **視頻ID** | TikTok 視頻ID | 7123456789 | ✅ |
| F | **檔案路徑** | 本地視頻文件路徑 | real/real_7123456789.mp4 | ✅ |
| G | **分析時間** | ISO 8601 格式時間戳 | 2025-12-12T10:30:00.123Z | ✅ |

### 關鍵特徵值（自我訓練用）

| 列號 | 列名 | 說明 | 物理意義 |
|------|------|------|----------|
| H | **fps** | 幀率 | 視頻流暢度（AI常用24/30，真實25/30/60） |
| I | **width** | 寬度（像素） | 分辨率（AI常用固定值如512/1024） |
| J | **height** | 高度（像素） | 分辨率（真實常用1080/1920） |
| K | **duration** | 時長（秒） | 視頻長度 |
| L | **avg_brightness** | 平均亮度 (0-255) | AI視頻常過度曝光或過暗 |
| M | **avg_contrast** | 平均對比度 | AI視頻對比度常異常高 |
| N | **avg_saturation** | 平均飽和度 | AI視頻飽和度常過高（不自然） |
| O | **avg_blur** | 平均模糊度 | 真實視頻有自然運動模糊 |
| P | **avg_optical_flow** | 平均光流（運動強度） | AI視頻運動常不符合物理定律 |
| Q | **scene_changes** | 場景變化次數 | 剪輯頻率 |
| R | **dct_energy** | DCT高頻能量 | AI生成常有頻域異常 |
| S | **spectral_entropy** | 頻譜熵 | 頻域複雜度（AI常過於規則） |
| T | **audio_sample_rate** | 音頻採樣率 | 音頻質量指標 |
| U | **audio_channels** | 音頻聲道數 | 立體聲/單聲道 |
| V | **bitrate** | 視頻碼率（kbps） | 壓縮率指標 |

### 輔助信息

| 列號 | 列名 | 說明 | 示例 |
|------|------|------|------|
| W | **人工復審結果** | 人工復審後的標籤 | REAL / AI / 電影動畫 / 空白 |
| X | **復審時間** | 人工復審時間戳 | 2025-12-12T11:00:00.123Z |
| Y | **備註** | 其他備註信息 | 夜店環境/美顏濾鏡/手持拍攝 |

---

## 📐 示例數據

```
序號  影片網址                                          AI檢測分類  信心度  視頻ID       檔案路徑                    分析時間                     fps   width  height  duration  avg_brightness  avg_contrast  avg_saturation  avg_blur  avg_optical_flow  scene_changes  dct_energy  spectral_entropy  ...
1     https://www.tiktok.com/@user/video/7123456789   REAL        95.2    7123456789   real/real_7123456789.mp4    2025-12-12T10:30:00.123Z     30    1080   1920    15.3      128.5           45.2          112.3           8.5       12.4              3              1250.3      4.56              ...
2     https://www.tiktok.com/@ai/video/7234567890     AI          98.7    7234567890   ai/ai_7234567890.mp4        2025-12-12T10:31:15.456Z     24    1024   1024    12.0      180.2           78.9          185.6           2.1       8.9               1              2850.7      3.21              ...
3     https://www.tiktok.com/@user2/video/7345678901  NOT_SURE    45.8    7345678901   not sure/not_sure_7345678901.mp4  2025-12-12T10:32:30.789Z  30    1080   1920    18.7      145.3           52.1          125.8           6.2       10.8              5              1680.5      4.12              ...
4     https://www.tiktok.com/@movie/video/7456789012  電影動畫     99.1    7456789012   電影動畫/movie_7456789012.mp4      2025-12-12T10:33:45.012Z  24    3840   2160    120.5     135.8           65.3          98.7            3.5       15.2              45             1950.2      4.89              ...
```

---

## 🎯 AI檢測分類說明

| 分類 | 英文 | 說明 | 信心度範圍 |
|------|------|------|------------|
| **REAL** | Real | 真實視頻（人類拍攝） | 通常 > 70% |
| **AI** | AI | AI生成視頻 | 通常 > 70% |
| **NOT_SURE** | Not Sure | 不確定（需人工復審） | 通常 30-70% |
| **電影動畫** | Movie/Anime | 電影/動畫/遊戲（排除訓練） | 通常 > 80% |

---

## 🔍 特徵值解讀

### 1. 視覺特徵差異（AI vs 真實）

**AI視頻典型特徵**:
- ✓ `avg_brightness`: 過度曝光 (> 150) 或過暗 (< 80)
- ✓ `avg_contrast`: 對比度異常高 (> 70)
- ✓ `avg_saturation`: 飽和度過高 (> 150)，顏色不自然
- ✓ `avg_blur`: 模糊度極低 (< 3)，過於銳利
- ✓ `dct_energy`: DCT高頻能量異常 (> 2000)
- ✓ `spectral_entropy`: 頻譜熵低 (< 3.5)，過於規則

**真實視頻典型特徵**:
- ✓ `avg_brightness`: 正常範圍 (100-150)
- ✓ `avg_contrast`: 中等對比度 (40-60)
- ✓ `avg_saturation`: 自然飽和度 (90-130)
- ✓ `avg_blur`: 自然運動模糊 (5-10)
- ✓ `avg_optical_flow`: 符合物理定律的運動
- ✓ `spectral_entropy`: 高複雜度 (> 4.0)

### 2. 運動特徵差異

**物理違規指標**:
- `avg_optical_flow`: AI視頻運動常不符合慣性定律
- `scene_changes`: 真實視頻剪輯自然，AI視頻常無剪輯或過度剪輯

### 3. 分辨率特徵

**AI生成常用分辨率**:
- 512×512, 1024×1024 (正方形，Stable Diffusion/Midjourney)
- 768×768 (DALL-E)

**真實視頻常用分辨率**:
- 1080×1920 (豎屏手機)
- 1920×1080 (橫屏)
- 720×1280 (較舊手機)

---

## 💡 自我訓練邏輯

### 1. 特徵權重自動調整

基於 Excel C 的統計分析結果，動態調整特徵權重：

```python
# 從 Excel C 讀取 discrimination_score
feature_weights = {
    'dct_energy': 0.25,        # 最強區分能力
    'avg_optical_flow': 0.20,  # 物理違規檢測
    'avg_saturation': 0.15,    # 視覺異常
    'spectral_entropy': 0.15,  # 頻域規律性
    'avg_contrast': 0.10,      # 對比度異常
    # ... 其他特徵
}
```

### 2. 閾值動態優化

根據 Excel D 的累積數據，自動計算最優閾值：

```python
# Real vs AI 分界線
threshold_brightness = (real_avg + ai_avg) / 2
threshold_dct = real_avg + 2 * real_std  # 2倍標準差
```

### 3. NOT_SURE 觸發條件

```python
# 當特徵值接近邊界時，標記為 NOT_SURE
if abs(feature_value - threshold) < uncertainty_margin:
    classification = "NOT_SURE"
    confidence = 50.0  # 中等信心度
```

---

## 📂 文件位置

```
C:\Users\s_robby518\Documents\trae_projects\ai testing\
└── tiktok_labeler\
    └── tiktok videos download\
        ├── data\
        │   └── excel_d_detection_results.xlsx  ← Excel D 文件
        ├── real\                               ← 真實視頻文件夾
        │   └── real_7123456789.mp4
        ├── ai\                                 ← AI視頻文件夾
        │   └── ai_7234567890.mp4
        ├── not sure\                           ← 不確定視頻文件夾
        │   └── not_sure_7345678901.mp4
        └── 電影動畫\                            ← 電影/動畫文件夾
            └── movie_7456789012.mp4
```

---

## 🔄 工作流程

```
[批量下載2000個視頻]
        ↓
[AI檢測模組分析每個視頻]
        ↓
[提取15+特徵值]
        ↓
[分類: REAL/AI/NOT_SURE/電影動畫]
        ↓
[記錄到 Excel D]
        ↓
[自動移動文件到對應文件夾]
        ↓
[提取 NOT_SURE 視頻]
        ↓
[本地Tinder復審系統]
        ↓
[人工標註後移動到正確文件夾]
        ↓
[更新 Excel D 人工復審結果]
        ↓
[重新訓練 → 模組優化]
        ↓
循環提升 → 99% 準確率
```

---

## ✅ 驗證格式

打開 Excel D 後，應該看到：
1. ✅ 第一列是「序號」（1, 2, 3...）
2. ✅ 第二列是「影片網址」（完整TikTok URL）
3. ✅ 第三列是「AI檢測分類」（REAL/AI/NOT_SURE/電影動畫）
4. ✅ 第四列是「信心度」（0-100）
5. ✅ 後續列包含15+個關鍵特徵值
6. ✅ 所有分類結果全部大寫
7. ✅ 包含人工復審列（初始為空）

---

**設計原則**: 物理不可偽造特徵 + 海量數據訓練 + 持續優化循環

**最後更新**: 2025-12-12
