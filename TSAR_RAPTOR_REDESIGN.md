# TSAR-RAPTOR System Redesign
**沙皇炸彈 × 猛禽3 架構重構 - 第一性原理驅動**

Date: 2025-12-14
Problem: 11/12 模組無判別力（差距<10分），ensemble投票失效

---

## 🎯 第一性原理分析

### 當前問題根源

**數據驅動發現**：
```
有效模組: model_fingerprint_detector (差距46.7分)
無效模組: 11個模組 (差距<10分)
  - frequency_analyzer: +4.1 (幾乎無用)
  - sensor_noise_authenticator: -6.0 (反向！)
  - physics_violation_detector: -0.5 (完全無用)
  - heartbeat_detector: +1.7 (無用)
  - 其他7個: <3分 (完全無用)
```

**第一性原理診斷**：
1. ❌ **模組設計錯誤** - 檢測的不是AI無法模擬的物理特性
2. ❌ **閾值設定錯誤** - 所有視頻都給50分（預設值）
3. ❌ **權重混亂** - 無用模組權重過高，有用模組被稀釋
4. ❌ **架構失效** - 沒有級聯放大，ensemble投票被單一模組主導

---

## 💣 沙皇炸彈架構 - 三階段輻射內爆

### 設計原理（Tsar Bomba Physics）

```
Primary Fission (初級裂變) → 40% 能量
    ↓ Radiation Implosion (輻射內爆)
Secondary Fusion (次級聚變) → 30% 能量
    ↓ Radiation Implosion
Tertiary Fusion (三級聚變) → 30% 能量
    ↓ Thermonuclear Yield
Final Score (97% 物理純度)
```

**級聯放大機制**：
- Primary高分(>70) → 放大Secondary敏感度 ×1.2 → 強化AI判定
- Primary低分(<30) → 抑制Secondary敏感度 ×0.8 → 保護真實視頻
- Secondary結果 → 調節Tertiary → 最終融合決策

---

## 🚀 Phase I - Primary Fission (物理不可偽造層)

**能量佔比**: 40%
**原理**: AI無法完美複製物理硬件的隨機性

### 1.1 sensor_noise_authenticator (傳感器噪聲)

**第一性原理**：
- 真實相機傳感器有**量子散粒噪聲** (shot noise)
- AI生成視頻的噪聲是**算法噪聲**，非物理噪聲
- 兩者的**頻譜特徵**和**空間分佈**完全不同

**當前問題**：
- 差距-6.0（KILL=61.8, SAFE=67.8）**反向！**
- 說明當前實現完全錯誤

**修正方案**：
```python
# 原理：真實傳感器噪聲在暗區更明顯（ISO噪聲）
def detect_sensor_noise_authentic(video_path):
    # 1. 提取暗區幀（亮度<50）
    dark_frames = extract_dark_regions(video)

    # 2. 計算噪聲頻譜
    noise_spectrum = fft2d(dark_frames)

    # 3. 真實傳感器特徵（關鍵）
    # - 白噪聲（平坦頻譜）
    # - 暗電流噪聲（低頻成分）
    # - 固定模式噪聲（空間相關）

    white_noise_ratio = compute_white_noise(noise_spectrum)
    dark_current = compute_low_freq_noise(noise_spectrum)
    fixed_pattern = compute_spatial_correlation(dark_frames)

    # AI生成視頻：白噪聲低，固定模式高（算法痕跡）
    # 真實視頻：白噪聾高，固定模式低（隨機性）

    if white_noise_ratio > 0.7 and fixed_pattern < 0.3:
        return 20  # 真實（authentic sensor noise）
    elif white_noise_ratio < 0.4 and fixed_pattern > 0.6:
        return 85  # AI（algorithmic noise）
    else:
        return 50  # 不確定
```

**技術要點**：
- 不要全畫面分析（壓縮會破壞噪聲）
- **只分析暗區** (亮度<50的像素)
- 計算**噪聲自相關函數**（真實=低，AI=高）
- 檢測**讀出噪聲**（真實傳感器特有）

**預期改善**：
- 從差距-6.0 → +40分（成為關鍵判別器）

---

### 1.2 physics_violation_detector (物理規律檢測)

**第一性原理**：
- **牛頓第一定律**：物體保持運動/靜止（慣性）
- **光學規律**：景深、焦距、透視遵循光學定律
- **AI視頻違規**：運動突變、焦距跳躍、透視扭曲

**當前問題**：
- 差距-0.5（KILL=78.7, SAFE=79.2）**完全無用！**
- 說明檢測的特徵在AI和真實片中都一樣

**修正方案**：
```python
# 原理：AI生成的運動不遵守牛頓定律
def detect_physics_violation(video_path):
    frames = extract_frames(video, sample=60)

    # 1. 光流分析（運動場）
    optical_flow = compute_optical_flow(frames)

    # 2. 加速度跳變檢測（關鍵）
    # 真實視頻：加速度連續（慣性）
    # AI視頻：加速度突變（幀間不連續）

    acceleration = compute_acceleration(optical_flow)
    jerk = np.diff(acceleration, axis=0)  # 加加速度（三階導數）

    # 物理違規：jerk過大（運動突變）
    jerk_violations = np.sum(np.abs(jerk) > JERK_THRESHOLD)

    # 3. 景深一致性（光學規律）
    # 真實視頻：景深符合鏡頭焦距
    # AI視頻：前景清晰但背景也清晰（物理不可能）

    depth_consistency = check_depth_of_field(frames)

    # 4. 透視扭曲
    # AI人像：臉部透視異常（額頭/下巴比例錯誤）
    perspective_score = check_perspective_distortion(frames)

    # 綜合判定
    if jerk_violations > 5 or depth_consistency < 0.3:
        return 85  # 物理違規 = AI
    elif jerk_violations < 2 and depth_consistency > 0.7:
        return 20  # 物理正常 = 真實
    else:
        return 50
```

**技術要點**：
- **不要只看運動幅度**（真實和AI都可能有大運動）
- **檢測運動的連續性**（jerk = 加加速度）
- **檢測景深矛盾**（全畫面清晰 = AI特徵）
- **檢測透視扭曲**（AI人臉常見問題）

**預期改善**：
- 從差距-0.5 → +30分（中等判別力）

---

### 1.3 frequency_analyzer (頻域分析)

**第一性原理**：
- **傅里葉不變性**：自然視頻遵循1/f噪聲（粉紅噪聲）
- **AI指紋**：GAN/Diffusion模型在高頻有棋盤模式
- **壓縮痕跡**：TikTok等平台的高頻截斷

**當前問題**：
- 差距+4.1（KILL=82.9, SAFE=78.8）**幾乎無用**
- 原因：所有視頻都有高頻截斷（壓縮），無法區分

**修正方案**：
```python
# 原理：檢測頻域的AI特徵，而非壓縮痕跡
def analyze_frequency(video_path):
    frames = extract_frames(video, sample=100)

    # 1. 2D FFT（關鍵：空間頻域）
    fft_2d = np.fft.fft2(frames, axes=(1, 2))
    magnitude = np.abs(fft_2d)

    # 2. 檢測棋盤模式（Checkerboard Pattern）
    # AI特徵：上採樣層產生的週期性模式
    # 檢測方法：在高頻區域尋找週期性峰值

    high_freq = magnitude[:, -20:, -20:]  # 右上角 = 高頻
    checkerboard_score = detect_periodic_peaks(high_freq)

    # 3. 頻譜熵（Spectral Entropy）
    # 真實視頻：高熵（頻率分佈廣）
    # AI視頻：低熵（特定頻率集中）

    entropy = compute_spectral_entropy(magnitude)

    # 4. 1/f 噪聲偏離度
    # 真實視頻：遵循 1/f (粉紅噪聲)
    # AI視頻：偏離 1/f (白噪聲或更平坦)

    pink_noise_fit = fit_1_over_f(magnitude)

    # 關鍵：不考慮低bitrate的高頻截斷
    # 只檢測AI特有的棋盤模式和熵異常

    if checkerboard_score > 0.6 or entropy < 0.4:
        return 80  # AI指紋
    elif checkerboard_score < 0.2 and entropy > 0.7:
        return 30  # 真實
    else:
        return 50
```

**技術要點**：
- **不檢測高頻截斷**（壓縮都有，無判別力）
- **檢測棋盤模式**（AI上採樣層特有）
- **計算頻譜熵**（AI更規律，熵更低）
- **忽略低bitrate視頻**（壓縮破壞頻域）

**預期改善**：
- 從差距+4.1 → +25分（中等判別力）

---

### 1.4 texture_noise_detector (紋理噪聲)

**第一性原理**：
- **真實紋理**：來自物理表面（皮膚、衣服、牆面）
- **AI紋理**：來自生成網絡（平滑、規律、無細節）
- **關鍵差異**：真實紋理有**高頻細節**和**隨機性**

**當前問題**：
- 差距+3.3（KILL=18.3, SAFE=15.0）**幾乎無用**
- 說明當前檢測方法無效

**修正方案**：
```python
# 原理：AI生成的紋理缺乏真實的隨機細節
def detect_texture_noise(video_path):
    frames = extract_frames(video, sample=50)

    # 1. 提取紋理區域（關鍵：皮膚、衣服）
    # 不要分析天空、牆面（真實也平滑）

    skin_regions = detect_skin_regions(frames)
    cloth_regions = detect_cloth_regions(frames)

    # 2. 計算紋理複雜度（Texture Complexity）
    # 真實皮膚：有毛孔、細紋、雀斑
    # AI皮膚：過度平滑（美顏效果）

    skin_complexity = compute_texture_complexity(skin_regions)

    # 3. 高頻細節比例
    # 真實衣服：織物紋理（高頻）
    # AI衣服：平滑或重複紋理

    high_freq_ratio = compute_high_freq_ratio(cloth_regions)

    # 4. 紋理隨機性（Randomness）
    # 真實紋理：隨機分佈
    # AI紋理：週期性或過於規律

    randomness = compute_texture_randomness(skin_regions)

    # 綜合判定
    if skin_complexity < 0.3 or randomness < 0.4:
        return 75  # 過度平滑 = AI
    elif skin_complexity > 0.6 and high_freq_ratio > 0.5:
        return 25  # 真實紋理
    else:
        return 50
```

**技術要點**：
- **只分析皮膚和衣服**（不分析背景）
- **檢測過度平滑**（AI美顏效果）
- **檢測紋理隨機性**（真實=高，AI=低）
- **避免低bitrate影響**（壓縮會降低細節）

**預期改善**：
- 從差距+3.3 → +20分（中等判別力）

---

## ⚡ Phase II - Secondary Fusion (生物力學層)

**能量佔比**: 30%
**原理**: 人類生物信號具有個體特徵和混沌性
**級聯放大**: Phase I 結果調節 Phase II 敏感度

### 2.1 heartbeat_detector (心跳檢測)

**第一性原理**：
- **心率變異性 (HRV)**：真實心跳有不規則性（混沌系統）
- **AI模擬心跳**：過於規律（週期性太強）
- **關鍵頻率**：0.8-2.5 Hz（心跳範圍）

**當前問題**：
- 差距+1.7（KILL=51.7, SAFE=50.0）**幾乎無用**
- 所有視頻都給50分（預設值），說明檢測失敗

**修正方案**：
```python
# 原理：真實心跳有HRV（心率變異），AI心跳過於規律
def detect_heartbeat(video_path):
    frames = extract_frames(video, sample=300, fps=30)  # 10秒視頻

    # 1. 提取臉部ROI（關鍵：額頭、臉頰）
    # 心跳信號在皮膚微循環中可見（rPPG）

    face_roi = detect_face_roi(frames, region='forehead')

    if face_roi is None or len(face_roi) < 100:
        return 50  # 無臉部 = 不確定

    # 2. 提取RGB信號（綠色通道最敏感）
    green_signal = extract_green_channel(face_roi)

    # 3. 帶通濾波（0.8-2.5 Hz = 48-150 BPM）
    filtered_signal = bandpass_filter(green_signal, 0.8, 2.5, fps=30)

    # 4. FFT找心跳頻率
    fft_signal = np.fft.fft(filtered_signal)
    dominant_freq = find_dominant_frequency(fft_signal)

    # 5. 計算HRV（關鍵判別特徵）
    # 真實心跳：HRV > 50ms（不規則）
    # AI心跳：HRV < 20ms（過於規律）

    peak_intervals = find_peak_intervals(filtered_signal)
    hrv = np.std(peak_intervals) * 1000 / fps  # 轉換為ms

    # 6. 判定
    if hrv > 50 and 0.8 < dominant_freq < 2.5:
        return 25  # 真實心跳
    elif hrv < 20 or dominant_freq < 0.5:
        return 80  # AI（無心跳或過於規律）
    else:
        return 50
```

**技術要點**：
- **使用rPPG技術**（remote photoplethysmography）
- **檢測HRV**（心率變異性），不只是心率
- **需要臉部特寫**（臉佔比>30%）
- **需要靜態視頻**（運動會干擾）

**預期改善**：
- 從差距+1.7 → +35分（中高判別力）
- **但僅對有臉部特寫的視頻有效**

---

### 2.2 blink_dynamics_analyzer (眨眼動力學)

**第一性原理**：
- **眨眼速度**：真實眨眼150-200ms（肌肉控制）
- **眨眼頻率**：15-20次/分鐘（個體差異）
- **AI眨眼**：速度不自然、頻率過於規律

**當前問題**：
- 差距+0.0（KILL=50.0, SAFE=50.0）**完全無用**
- 所有視頻都給50分，檢測完全失敗

**修正方案**：
```python
# 原理：真實眨眼有特定速度曲線（快閉慢開）
def analyze_blink_dynamics(video_path):
    frames = extract_frames(video, sample=600, fps=30)  # 20秒

    # 1. 檢測眼睛ROI
    eyes = detect_eyes(frames)

    if eyes is None:
        return 50  # 無眼睛 = 不確定

    # 2. 計算EAR（Eye Aspect Ratio）
    # 眼睛高度/寬度比例，眨眼時下降

    ear_values = []
    for frame in frames:
        ear = compute_eye_aspect_ratio(frame, eyes)
        ear_values.append(ear)

    # 3. 檢測眨眼事件（EAR下降>30%）
    blinks = detect_blink_events(ear_values, threshold=0.3)

    if len(blinks) < 3:
        return 50  # 眨眼次數太少，不確定

    # 4. 分析眨眼速度曲線（關鍵）
    # 真實眨眼：閉眼快（50-80ms），開眼慢（100-150ms）
    # AI眨眼：對稱（速度一致）或過快/過慢

    close_speeds = []
    open_speeds = []

    for blink in blinks:
        close_speed = compute_close_speed(blink)
        open_speed = compute_open_speed(blink)
        close_speeds.append(close_speed)
        open_speeds.append(open_speed)

    avg_close = np.mean(close_speeds)
    avg_open = np.mean(open_speeds)

    # 真實特徵：close < open（快閉慢開）
    asymmetry = avg_open / avg_close

    # 5. 眨眼間隔變異性
    # 真實：變異性高（3-10秒不等）
    # AI：變異性低（過於規律）

    blink_intervals = np.diff([b['timestamp'] for b in blinks])
    interval_std = np.std(blink_intervals)

    # 6. 判定
    if 1.3 < asymmetry < 2.5 and interval_std > 1.0:
        return 25  # 真實眨眼
    elif asymmetry < 1.1 or interval_std < 0.3:
        return 75  # AI眨眼（對稱或過於規律）
    else:
        return 50
```

**技術要點**：
- **檢測眨眼速度曲線**（快閉慢開）
- **計算眨眼間隔變異性**（真實=高，AI=低）
- **需要清晰眼部**（低解析度會失敗）
- **至少3次眨眼**（統計顯著性）

**預期改善**：
- 從差距+0.0 → +30分（中等判別力）
- **但僅對有清晰眼部的視頻有效**

---

### 2.3 lighting_geometry_checker (光照幾何)

**第一性原理**：
- **手持抖動**：真實手機視頻有微小抖動
- **三腳架穩定**：完全靜止或AI生成
- **光照一致性**：真實視頻光照符合物理（單一光源）

**當前問題**：
- 差距-2.1（KILL=21.7, SAFE=23.8）**反向且無用**

**修正方案**：
```python
# 原理：真實手持視頻有微小抖動（0.5-2度/秒）
def check_lighting_geometry(video_path):
    frames = extract_frames(video, sample=100)

    # 1. 計算相機抖動（陀螺儀模擬）
    # 使用光流估計相機旋轉角度

    rotation_angles = []
    for i in range(len(frames)-1):
        flow = compute_optical_flow(frames[i], frames[i+1])
        rotation = estimate_rotation_from_flow(flow)
        rotation_angles.append(rotation)

    # 真實手持：微小抖動（0.5-2度）
    # 三腳架：幾乎無抖動（<0.1度）
    # AI：可能完全靜止或異常抖動

    avg_jitter = np.mean(np.abs(rotation_angles))

    # 2. 光照一致性（檢測多光源）
    # 真實視頻：單一主光源
    # AI視頻：多光源或光照不合理（陰影方向矛盾）

    light_sources = estimate_light_sources(frames)
    light_consistency = check_shadow_consistency(frames)

    # 3. 判定
    if 0.5 < avg_jitter < 2.0 and light_consistency > 0.7:
        return 25  # 真實手持
    elif avg_jitter < 0.1 or light_consistency < 0.3:
        return 70  # 三腳架或AI（光照異常）
    else:
        return 50
```

**技術要點**：
- **檢測微小抖動**（手持特徵）
- **檢測光照矛盾**（AI常見問題）
- **使用光流估計旋轉**（不需要陀螺儀數據）

**預期改善**：
- 從差距-2.1 → +20分（中等判別力）

---

## 🧮 Phase III - Tertiary Fusion (數學結構層)

**能量佔比**: 30%
**原理**: 機器學習模型留下數學痕跡
**級聯調節**: Phase I+II 結果決定 Phase III 權重

### 3.1 model_fingerprint_detector (模型指紋)

**當前狀態**: 唯一有效模組（差距+46.7）

**優化方案**：
```python
# 保持當前檢測邏輯，但加入級聯調節
def detect_model_fingerprint(video_path, phase1_score, phase2_score):
    # 原有檢測邏輯...
    base_score = current_detection_logic(video_path)

    # 級聯調節（關鍵）
    if phase1_score > 70:  # Phase I 說AI
        # 提高敏感度：更容易檢測到AI特徵
        adjusted_score = base_score * 1.2
    elif phase1_score < 30:  # Phase I 說真實
        # 降低敏感度：避免誤報
        adjusted_score = base_score * 0.8
    else:
        adjusted_score = base_score

    return np.clip(adjusted_score, 0, 100)
```

**不需要重新設計**（已經有效），只需級聯調節

---

### 3.2 text_fingerprinting (文本指紋)

**第一性原理**：
- **AI帶貨片特徵**：大量文字overlay、固定模板
- **真實視頻**：無文字或少量文字

**當前問題**：
- 差距+2.4（幾乎無用）

**修正方案**：
```python
# 原理：AI帶貨片有固定文本模板
def detect_text_fingerprint(video_path):
    frames = extract_frames(video, sample=30)

    # 1. OCR檢測文本
    texts = []
    for frame in frames:
        text = ocr_extract(frame)
        texts.append(text)

    # 2. 檢測文本穩定性（AI特徵）
    # AI帶貨片：文本位置固定、樣式一致
    # 真實視頻：無文本或文本移動

    text_stability = compute_text_stability(texts)

    # 3. 檢測營銷關鍵詞
    # AI帶貨：「限時」「優惠」「立即」等

    marketing_keywords = ['限時', '優惠', '立即', '搶購', '折扣']
    keyword_count = sum(any(kw in t for kw in marketing_keywords) for t in texts)

    # 4. 判定
    if text_stability > 0.8 and keyword_count > 3:
        return 85  # AI帶貨片
    elif text_stability < 0.3:
        return 30  # 真實視頻
    else:
        return 50
```

**預期改善**：
- 從差距+2.4 → +40分（針對AI帶貨片）

---

### 3.3 其他Phase III模組

**semantic_stylometry, av_sync_verifier, metadata_extractor**：

**猛禽3原則**: "No part is the best part"

**決策**: **移除或降權到0.1**
- 這3個模組差距<3分，完全無判別力
- 保留它們只會增加計算成本
- 簡化系統，提升效率

---

## 🏗️ 新架構實現

### 三階段級聯評分系統

```python
def tsar_raptor_detection(video_path):
    # ========== Phase I - Primary Fission (40%) ==========
    sna_score = sensor_noise_authenticator(video_path)
    pvd_score = physics_violation_detector(video_path)
    fa_score = frequency_analyzer(video_path)
    tn_score = texture_noise_detector(video_path)

    # Phase I 加權平均
    phase1_score = (sna_score * 0.3 +
                    pvd_score * 0.3 +
                    fa_score * 0.25 +
                    tn_score * 0.15)

    # ========== Radiation Implosion 1 (級聯放大) ==========
    if phase1_score > 70:
        phase2_multiplier = 1.2  # AI可能性高，放大Phase II
    elif phase1_score < 30:
        phase2_multiplier = 0.8  # 真實可能性高，抑制Phase II
    else:
        phase2_multiplier = 1.0

    # ========== Phase II - Secondary Fusion (30%) ==========
    hb_score = heartbeat_detector(video_path) * phase2_multiplier
    bd_score = blink_dynamics_analyzer(video_path) * phase2_multiplier
    lg_score = lighting_geometry_checker(video_path) * phase2_multiplier

    # Phase II 加權平均
    phase2_score = (hb_score * 0.4 +
                    bd_score * 0.35 +
                    lg_score * 0.25)

    # ========== Radiation Implosion 2 (級聯放大) ==========
    combined_12 = (phase1_score * 0.6 + phase2_score * 0.4)

    if combined_12 > 65:
        phase3_multiplier = 1.15  # 強化AI判定
    elif combined_12 < 35:
        phase3_multiplier = 0.85  # 保護真實視頻
    else:
        phase3_multiplier = 1.0

    # ========== Phase III - Tertiary Fusion (30%) ==========
    mfp_score = model_fingerprint_detector(video_path) * phase3_multiplier
    tf_score = text_fingerprinting(video_path)

    # Phase III 加權平均（移除無用模組）
    phase3_score = (mfp_score * 0.7 + tf_score * 0.3)

    # ========== Final Thermonuclear Yield ==========
    final_score = (phase1_score * 0.4 +
                   phase2_score * 0.3 +
                   phase3_score * 0.3)

    return {
        'final_score': final_score,
        'phase1': phase1_score,
        'phase2': phase2_score,
        'phase3': phase3_score,
        'threat_level': classify_threat(final_score)
    }
```

---

## 📊 預期改善效果

### 模組判別力提升

| 模組 | 優化前差距 | 優化後預期 | 提升 |
|------|-----------|-----------|------|
| sensor_noise_authenticator | -6.0 | +40 | +46 ⭐⭐⭐ |
| heartbeat_detector | +1.7 | +35 | +33 ⭐⭐⭐ |
| blink_dynamics_analyzer | 0.0 | +30 | +30 ⭐⭐⭐ |
| physics_violation_detector | -0.5 | +30 | +30 ⭐⭐⭐ |
| frequency_analyzer | +4.1 | +25 | +21 ⭐⭐ |
| texture_noise_detector | +3.3 | +20 | +17 ⭐⭐ |
| lighting_geometry_checker | -2.1 | +20 | +22 ⭐⭐ |
| text_fingerprinting | +2.4 | +40 | +38 ⭐⭐⭐ |
| model_fingerprint_detector | +46.7 | +46.7 | 保持 ⭐⭐⭐ |

**移除模組**（猛禽3原則）：
- semantic_stylometry（差距0.0）
- av_sync_verifier（差距0.0）
- metadata_extractor（差距0.0）

### 系統性能提升

| 指標 | 優化前 | 優化後預期 |
|-----|-------|-----------|
| 有效模組數 | 1/12 (8%) | 9/9 (100%) |
| Ensemble效能 | 失效 | 正常 |
| 誤報率 | 23.8% | <5% |
| 準確率 | 7.1% | >90% |
| 執行時間 | 100% | 75%（移除3模組） |

---

## 🚀 實施計劃

### 第1階段（1週）- Phase I優化
- [ ] 重寫 sensor_noise_authenticator（暗區噪聲分析）
- [ ] 重寫 physics_violation_detector（jerk檢測）
- [ ] 重寫 frequency_analyzer（棋盤模式）
- [ ] 重寫 texture_noise_detector（皮膚紋理）

### 第2階段（1週）- Phase II優化
- [ ] 重寫 heartbeat_detector（rPPG + HRV）
- [ ] 重寫 blink_dynamics_analyzer（眨眼曲線）
- [ ] 重寫 lighting_geometry_checker（抖動檢測）

### 第3階段（3天）- Phase III優化
- [ ] 優化 text_fingerprinting（營銷關鍵詞）
- [ ] 移除無用模組（3個）

### 第4階段（3天）- 級聯系統
- [ ] 實現三階段級聯評分
- [ ] 實現輻射內爆機制
- [ ] 測試驗證

---

**設計原則總結**：
1. **第一性原理** - 檢測AI無法模擬的物理/生物特性
2. **沙皇炸彈** - 三階段級聯放大，97%物理純度
3. **猛禽3** - 移除無用部分，極致簡化

**預期結果**: 從1個有效模組 → 9個有效模組，ensemble真正發揮作用
