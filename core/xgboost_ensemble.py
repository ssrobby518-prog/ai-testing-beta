#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blue Team Phase III - Module 6: XGBoost Ensemble Brain (集成決策系統)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

第一性原理 (First Principle Axiom):
    "真實與偽造的邊界在高維特徵空間中是非線性的。
    簡單的規則判斷 (If-Else) 必將失效，只有梯度提升樹能自動學習複雜的組合模式。"

替換目標 (Replacement Target):
    取代 core/scoring_engine.py 中的大量if-else規則邏輯

技術棧 (Technical Stack):
    - XGBoost / LightGBM (Gradient Boosting Decision Tree)
    - SHAP Values (可解釋性分析)
    - 輸入向量：模組分數 + 視頻元數據（20-30維）

功能 (Functionality):
    - 非線性決策：自動學習特徵間的組合關係
    - 可解釋性：輸出SHAP Values，解釋判定為假的具體原因

成功標準 (Success Criteria):
    - Precision > 99% (減少誤殺真實影片)
    - Recall > 95% (抓出絕大多數 AI 影片)

沙皇炸彈原則 (Tsar Bomba):
    機器學習能發現人類無法察覺的模式組合 - 這是對頂級AI偽造的終極武器

猛禽3引擎原則 (Raptor 3):
    模型推理可並行化，與現有架構完全解耦
"""

import logging
import numpy as np
import os
import pickle
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

# === 模型路徑配置 ===
MODEL_PATH = "models/xgboost_ensemble.pkl"
SHAP_EXPLAINER_PATH = "models/shap_explainer.pkl"


@dataclass
class XGBoostResult:
    """XGBoost決策結果"""
    ai_probability: float  # [0, 100]
    confidence: float  # 模型置信度 [0, 1]
    shap_values: Dict[str, float]  # 各特徵的SHAP貢獻值
    top_reasons: List[Tuple[str, float]]  # Top 3 判定原因


class XGBoostEnsemble:
    """
    XGBoost集成決策引擎

    替代傳統的if-else規則邏輯，使用機器學習自動發現模式
    """

    def __init__(self, model_path: str = MODEL_PATH):
        """
        初始化XGBoost模型

        Args:
            model_path: 預訓練模型路徑
        """
        self.model_path = model_path
        self.model = None
        self.shap_explainer = None
        self.feature_names = None

        # 嘗試加載模型
        if os.path.exists(model_path):
            self._load_model()
        else:
            logging.warning(f"XGBoost model not found at {model_path}, will use fallback")

    def _load_model(self):
        """加載預訓練的XGBoost模型"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.feature_names = model_data['feature_names']
                logging.info(f"Loaded XGBoost model from {self.model_path}")

            # 加載SHAP解釋器（如果存在）
            if os.path.exists(SHAP_EXPLAINER_PATH):
                with open(SHAP_EXPLAINER_PATH, 'rb') as f:
                    self.shap_explainer = pickle.load(f)
                    logging.info("Loaded SHAP explainer")

        except Exception as e:
            logging.error(f"Failed to load XGBoost model: {e}")
            self.model = None

    def predict(
        self,
        module_scores: Dict[str, float],
        video_metadata: Dict[str, any]
    ) -> XGBoostResult:
        """
        XGBoost預測主函數

        Args:
            module_scores: 各模組的檢測分數 (12個模組)
            video_metadata: 視頻元數據 (bitrate, fps, face_presence等)

        Returns:
            XGBoostResult: 決策結果
        """
        # === 特徵向量化 ===
        feature_vector = self._vectorize_features(module_scores, video_metadata)

        # === 如果模型未訓練，使用fallback ===
        if self.model is None:
            return self._fallback_prediction(module_scores, video_metadata)

        # === XGBoost預測 ===
        try:
            import xgboost as xgb

            # 轉換為DMatrix（XGBoost格式）
            dmatrix = xgb.DMatrix(
                feature_vector.reshape(1, -1),
                feature_names=self.feature_names
            )

            # 預測概率
            ai_prob = float(self.model.predict(dmatrix)[0]) * 100.0

            # === SHAP可解釋性分析 ===
            shap_values, top_reasons = self._explain_prediction(feature_vector)

            return XGBoostResult(
                ai_probability=ai_prob,
                confidence=0.95,  # XGBoost通常高置信度
                shap_values=shap_values,
                top_reasons=top_reasons
            )

        except ImportError:
            logging.warning("XGBoost not installed, using fallback")
            return self._fallback_prediction(module_scores, video_metadata)
        except Exception as e:
            logging.error(f"XGBoost prediction error: {e}")
            return self._fallback_prediction(module_scores, video_metadata)

    def _vectorize_features(
        self,
        module_scores: Dict[str, float],
        video_metadata: Dict[str, any]
    ) -> np.ndarray:
        """
        特徵向量化（Feature Vectorization）

        將模組分數和元數據轉換為固定維度的向量

        特徵列表（按順序）：
        1-12: 模組分數 (12維)
        13-18: 元數據 (6維)
        19-25: 交叉特徵 (7維)

        Total: 25維特徵向量

        Returns:
            np.ndarray: (25,) 特徵向量
        """
        features = []

        # === 1-12: 模組分數（原始）===
        module_order = [
            'metadata_extractor', 'frequency_analyzer', 'texture_noise_detector',
            'model_fingerprint_detector', 'lighting_geometry_checker', 'heartbeat_detector',
            'blink_dynamics_analyzer', 'av_sync_verifier', 'text_fingerprinting',
            'semantic_stylometry', 'sensor_noise_authenticator', 'physics_violation_detector'
        ]

        for name in module_order:
            features.append(module_scores.get(name, 50.0) / 100.0)  # 歸一化到[0,1]

        # === 13-18: 元數據（歸一化）===
        bitrate = video_metadata.get('bitrate', 0)
        fps = video_metadata.get('fps', 30.0)
        face_presence = video_metadata.get('face_presence', 0.0)
        static_ratio = video_metadata.get('static_ratio', 0.0)
        width = video_metadata.get('width', 1920)
        height = video_metadata.get('height', 1080)

        features.append(bitrate / 10_000_000.0)  # 歸一化（假設最大10Mbps）
        features.append(fps / 60.0)  # 歸一化（假設最大60fps）
        features.append(face_presence)
        features.append(static_ratio)
        features.append(width / 4096.0)  # 歸一化（假設最大4K）
        features.append(height / 2160.0)

        # === 19-25: 交叉特徵（手工工程）===
        # 這些是人類專家知識的編碼

        mfp = module_scores.get('model_fingerprint_detector', 50.0)
        fa = module_scores.get('frequency_analyzer', 50.0)
        sna = module_scores.get('sensor_noise_authenticator', 50.0)
        pvd = module_scores.get('physics_violation_detector', 50.0)

        # 交叉特徵1: MFP * FA（兩個強特徵的乘積）
        features.append((mfp / 100.0) * (fa / 100.0))

        # 交叉特徵2: SNA * PVD（物理違規的組合）
        features.append((sna / 100.0) * (pvd / 100.0))

        # 交叉特徵3: Face Presence * MFP（人臉場景下的MFP更可靠）
        features.append(face_presence * (mfp / 100.0))

        # 交叉特徵4: 高分模組數量
        high_score_count = sum(1 for s in module_scores.values() if s >= 70)
        features.append(high_score_count / 12.0)

        # 交叉特徵5: 低分模組數量
        low_score_count = sum(1 for s in module_scores.values() if s <= 30)
        features.append(low_score_count / 12.0)

        # 交叉特徵6: 是否手機視頻
        is_phone_video = 1.0 if 800000 < bitrate < 1800000 else 0.0
        features.append(is_phone_video)

        # 交叉特徵7: 是否社交媒體視頻
        is_social = 1.0 if (400000 < bitrate < 1500000) else 0.0
        features.append(is_social)

        return np.array(features, dtype=np.float32)

    def _explain_prediction(
        self,
        feature_vector: np.ndarray
    ) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
        """
        使用SHAP解釋預測結果

        Args:
            feature_vector: 特徵向量

        Returns:
            (shap_values_dict, top_reasons)
        """
        if self.shap_explainer is None or self.feature_names is None:
            return {}, []

        try:
            import shap

            # 計算SHAP值
            shap_values = self.shap_explainer.shap_values(feature_vector.reshape(1, -1))

            # 轉換為字典
            shap_dict = {
                name: float(val)
                for name, val in zip(self.feature_names, shap_values[0])
            }

            # 找出Top 3貢獻最大的特徵
            sorted_features = sorted(
                shap_dict.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]

            return shap_dict, sorted_features

        except Exception as e:
            logging.error(f"SHAP explanation error: {e}")
            return {}, []

    def _fallback_prediction(
        self,
        module_scores: Dict[str, float],
        video_metadata: Dict[str, any]
    ) -> XGBoostResult:
        """
        備用預測（當XGBoost模型未訓練時）

        使用簡化的加權平均邏輯

        Args:
            module_scores: 模組分數
            video_metadata: 元數據

        Returns:
            XGBoostResult: 決策結果
        """
        # 簡單加權平均
        weights = {
            'model_fingerprint_detector': 2.2,
            'frequency_analyzer': 1.5,
            'sensor_noise_authenticator': 2.0,
            'physics_violation_detector': 1.8,
            'texture_noise_detector': 1.3,
            'text_fingerprinting': 1.4,
        }

        weighted_sum = 0.0
        weight_total = 0.0

        for name, score in module_scores.items():
            weight = weights.get(name, 1.0)
            weighted_sum += score * weight
            weight_total += weight

        ai_prob = weighted_sum / weight_total if weight_total > 0 else 50.0

        # 簡單的top reasons
        sorted_scores = sorted(
            module_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        top_reasons = [(name, score) for name, score in sorted_scores]

        return XGBoostResult(
            ai_probability=ai_prob,
            confidence=0.7,  # 備用方案置信度較低
            shap_values={},
            top_reasons=top_reasons
        )


# === 訓練函數（獨立運行）===
def train_xgboost_ensemble(
    training_data_path: str,
    output_model_path: str = MODEL_PATH,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.05
):
    """
    訓練XGBoost集成模型

    數據集格式（CSV）：
        feature_1, feature_2, ..., feature_25, label
        0.45, 0.67, ..., 0.89, 1  # AI
        0.12, 0.23, ..., 0.15, 0  # Real
        ...

    Args:
        training_data_path: 訓練數據CSV路徑
        output_model_path: 輸出模型路徑
        n_estimators: 樹的數量
        max_depth: 樹的最大深度
        learning_rate: 學習率
    """
    try:
        import xgboost as xgb
        import shap
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import precision_score, recall_score, f1_score

        logging.info("=== Training XGBoost Ensemble ===")

        # 加載數據
        df = pd.read_csv(training_data_path)
        logging.info(f"Loaded {len(df)} samples")

        # 分離特徵和標籤
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        feature_names = list(df.columns[:-1])

        # 訓練/測試分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 創建XGBoost分類器
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            use_label_encoder=False
        )

        # 訓練
        logging.info("Training model...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=True
        )

        # 評估
        y_pred = model.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")

        # 保存模型
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        model_data = {
            'model': model,
            'feature_names': feature_names
        }
        with open(output_model_path, 'wb') as f:
            pickle.dump(model_data, f)
        logging.info(f"Model saved to {output_model_path}")

        # 創建SHAP解釋器
        logging.info("Creating SHAP explainer...")
        explainer = shap.TreeExplainer(model)
        with open(SHAP_EXPLAINER_PATH, 'wb') as f:
            pickle.dump(explainer, f)
        logging.info(f"SHAP explainer saved to {SHAP_EXPLAINER_PATH}")

        # 特徵重要性
        feature_importance = model.feature_importances_
        importance_dict = dict(zip(feature_names, feature_importance))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        logging.info("Top 10 Important Features:")
        for name, importance in sorted_importance[:10]:
            logging.info(f"  {name}: {importance:.4f}")

    except ImportError:
        logging.error("XGBoost or SHAP not installed. Install with: pip install xgboost shap")
    except Exception as e:
        logging.error(f"Training failed: {e}")


if __name__ == "__main__":
    # 示例：訓練模型
    # train_xgboost_ensemble("datasets/training_data.csv")
    pass
