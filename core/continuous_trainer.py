#!/usr/bin/env python3
"""
Continuous Training System - 持續訓練系統
基於人工標註數據持續改進 XGBoost 模型

設計原則:
- 第一性原理: 人類標註是真相，機器學習逼近真相
- 沙皇炸彈純度: 只使用高質量標註（信心 >= 4）
- 猛禽3迭代: 快速測試 → 部署 → 持續改進

訓練觸發條件:
- 累積 >= 100 條高質量標註
- 每 100 條觸發一次重訓練

A/B 測試:
- 新模型 vs 舊模型
- 性能提升 >= 2% → 部署新模型
"""

import os
import sys
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.human_annotator import AnnotationDatabase

# 嘗試導入 XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost 未安裝，持續訓練功能不可用")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """訓練性能指標"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    model_version: str
    training_samples: int
    timestamp: float


class ContinuousTrainer:
    """持續訓練管理器"""

    def __init__(
        self,
        annotation_db_path: str = "data/annotations.db",
        model_save_dir: str = "models",
        min_confidence: int = 4,
        retrain_threshold: int = 100
    ):
        """
        Args:
            annotation_db_path: 標註數據庫路徑
            model_save_dir: 模型保存目錄
            min_confidence: 最低信心等級（1-5）
            retrain_threshold: 重訓練閾值（標註數量）
        """
        self.db = AnnotationDatabase(annotation_db_path)
        self.model_save_dir = Path(project_root) / model_save_dir
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.min_confidence = min_confidence
        self.retrain_threshold = retrain_threshold

        logger.info(f"持續訓練器初始化完成")
        logger.info(f"  • 最低信心等級: {min_confidence}")
        logger.info(f"  • 重訓練閾值: {retrain_threshold} 條標註")

    def check_and_retrain(self, force: bool = False) -> Optional[str]:
        """
        檢查是否達到重訓練條件，如果達到則執行訓練

        Args:
            force: 強制重訓練（忽略閾值檢查）

        Returns:
            新模型路徑，如果未訓練則返回 None
        """
        if not XGBOOST_AVAILABLE:
            logger.error("XGBoost 未安裝，無法執行訓練")
            return None

        # 獲取統計信息
        stats = self.db.get_annotation_stats()
        pending = stats['pending_training']
        high_quality = stats['high_quality']

        logger.info(f"檢查訓練條件:")
        logger.info(f"  • 高質量標註: {high_quality}")
        logger.info(f"  • 待訓練標註: {pending}")
        logger.info(f"  • 重訓練閾值: {self.retrain_threshold}")

        # 檢查是否達到閾值
        if not force and pending < self.retrain_threshold:
            logger.info(f"未達到重訓練閾值 ({pending} < {self.retrain_threshold})")
            return None

        logger.info("✅ 達到重訓練條件，開始訓練新模型...")

        # 加載訓練數據
        X, y, annotation_ids = self.prepare_training_data()

        if len(X) == 0:
            logger.error("沒有可用的訓練數據")
            return None

        logger.info(f"訓練數據準備完成: {len(X)} 樣本")

        # 訓練新模型
        new_model = self.train_xgboost(X, y)

        # 保存新模型
        import time
        model_version = f"xgboost_v{int(time.time())}"
        model_path = self.model_save_dir / f"{model_version}.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump(new_model, f)

        logger.info(f"新模型已保存: {model_path}")

        # A/B 測試（如果有舊模型）
        current_model_path = self.model_save_dir / "xgboost_current.pkl"
        if current_model_path.exists():
            improvement = self.ab_test(new_model, current_model_path, X, y)

            if improvement >= 0.02:  # 2% 提升
                logger.info(f"✅ 新模型性能提升 {improvement*100:.1f}%，部署新模型")
                self.deploy_model(model_path)
            else:
                logger.info(f"⚠️  新模型性能提升不足 ({improvement*100:.1f}% < 2%)，保留舊模型")
        else:
            logger.info("首次訓練，直接部署新模型")
            self.deploy_model(model_path)

        # 標記標註為已使用
        self.db.mark_as_used_for_training(annotation_ids)

        return str(model_path)

    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        從數據庫加載並準備訓練數據

        Returns:
            (X, y, annotation_ids)
            X: 特徵矩陣 (N, 25)
            y: 標籤向量 (N,) - 0=Real, 1=AI
            annotation_ids: 標註ID列表
        """
        # 獲取高質量標註
        annotations = self.db.get_high_quality_annotations(self.min_confidence)

        if len(annotations) == 0:
            logger.warning("沒有可用的高質量標註")
            return np.array([]), np.array([]), []

        X_list = []
        y_list = []
        annotation_ids = []

        for ann in annotations:
            # 解析 SHAP 原因（特徵分數）
            try:
                shap_reasons = json.loads(ann['shap_top_reasons'])

                # 構建特徵向量（簡化版：使用 AI 預測 + Top 3 SHAP 分數）
                # 實際應用中需要提取完整的 12 模組分數 + 元數據
                features = [
                    ann['ai_prediction'] / 100.0,  # 歸一化到 0-1
                    ann['ai_confidence']
                ]

                # 添加 Top 3 SHAP 分數（如果不足3個則填0）
                for i in range(3):
                    if i < len(shap_reasons):
                        features.append(shap_reasons[i][1] / 100.0)  # 歸一化
                    else:
                        features.append(0.0)

                # 補齊到 25 維（實際應用中需要完整特徵）
                while len(features) < 25:
                    features.append(0.0)

                X_list.append(features[:25])

                # 標籤：real=0, ai=1
                label = 1 if ann['human_label'] == 'ai' else 0
                y_list.append(label)

                annotation_ids.append(ann['id'])

            except Exception as e:
                logger.error(f"解析標註失敗 (ID={ann['id']}): {e}")
                continue

        X = np.array(X_list)
        y = np.array(y_list)

        logger.info(f"訓練數據準備完成:")
        logger.info(f"  • 樣本數: {len(X)}")
        logger.info(f"  • Real: {np.sum(y == 0)} 個")
        logger.info(f"  • AI: {np.sum(y == 1)} 個")

        return X, y, annotation_ids

    def train_xgboost(self, X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
        """
        訓練 XGBoost 分類器

        Args:
            X: 特徵矩陣 (N, 25)
            y: 標籤向量 (N,)

        Returns:
            訓練好的 XGBoost 模型
        """
        logger.info("開始訓練 XGBoost 模型...")

        # XGBoost 參數（基於沙皇炸彈原則：高純度檢測）
        params = {
            'max_depth': 6,  # 深度限制，防止過擬合
            'learning_rate': 0.1,  # 學習率
            'n_estimators': 100,  # 樹的數量
            'objective': 'binary:logistic',  # 二分類
            'eval_metric': 'auc',  # AUC評估
            'random_state': 42,
            'tree_method': 'hist',  # GPU加速（如果可用）
        }

        model = xgb.XGBClassifier(**params)

        # 訓練
        model.fit(
            X, y,
            eval_set=[(X, y)],
            verbose=False
        )

        logger.info("✅ XGBoost 訓練完成")

        # 顯示特徵重要性
        feature_importance = model.feature_importances_
        top_features = np.argsort(feature_importance)[::-1][:5]
        logger.info("Top 5 重要特徵:")
        for i, idx in enumerate(top_features, 1):
            logger.info(f"  {i}. Feature {idx}: {feature_importance[idx]:.3f}")

        return model

    def ab_test(
        self,
        new_model: xgb.XGBClassifier,
        old_model_path: Path,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> float:
        """
        A/B 測試：比較新舊模型性能

        Args:
            new_model: 新訓練的模型
            old_model_path: 舊模型路徑
            X_test: 測試特徵
            y_test: 測試標籤

        Returns:
            性能提升比例（0-1）
        """
        logger.info("開始 A/B 測試...")

        # 加載舊模型
        with open(old_model_path, 'rb') as f:
            old_model = pickle.load(f)

        # 新模型預測
        new_pred = new_model.predict(X_test)
        new_acc = np.mean(new_pred == y_test)

        # 舊模型預測
        old_pred = old_model.predict(X_test)
        old_acc = np.mean(old_pred == y_test)

        improvement = new_acc - old_acc

        logger.info(f"A/B 測試結果:")
        logger.info(f"  • 舊模型準確率: {old_acc*100:.2f}%")
        logger.info(f"  • 新模型準確率: {new_acc*100:.2f}%")
        logger.info(f"  • 性能提升: {improvement*100:.2f}%")

        return improvement

    def deploy_model(self, model_path: Path):
        """
        部署新模型（複製為 current 版本）

        Args:
            model_path: 新模型路徑
        """
        current_path = self.model_save_dir / "xgboost_current.pkl"

        # 備份舊模型
        if current_path.exists():
            import time
            backup_path = self.model_save_dir / f"xgboost_backup_{int(time.time())}.pkl"
            current_path.rename(backup_path)
            logger.info(f"舊模型已備份: {backup_path}")

        # 複製新模型
        import shutil
        shutil.copy(model_path, current_path)
        logger.info(f"✅ 新模型已部署: {current_path}")

    def get_training_history(self) -> List[Dict]:
        """獲取訓練歷史（從模型目錄）"""
        models = list(self.model_save_dir.glob("xgboost_v*.pkl"))
        history = []

        for model_path in sorted(models):
            version = model_path.stem
            timestamp = int(version.split('_v')[-1])
            history.append({
                'version': version,
                'path': str(model_path),
                'timestamp': timestamp,
                'size_mb': model_path.stat().st_size / (1024 * 1024)
            })

        return sorted(history, key=lambda x: x['timestamp'], reverse=True)

    def show_training_status(self):
        """顯示訓練狀態"""
        print(f"\n{'='*80}")
        print(f"{'持續訓練狀態'.center(80)}")
        print(f"{'='*80}")

        # 標註統計
        stats = self.db.get_annotation_stats()
        print(f"\n📊 標註數據:")
        print(f"  • 高質量標註: {stats['high_quality']}")
        print(f"  • 已用於訓練: {stats['used_for_training']}")
        print(f"  • 待訓練: {stats['pending_training']}")
        print(f"  • 重訓練閾值: {self.retrain_threshold}")

        progress = stats['pending_training'] / self.retrain_threshold
        bar_length = 50
        filled = int(bar_length * min(progress, 1.0))
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"  • 進度: [{bar}] {progress*100:.1f}%")

        # 訓練歷史
        history = self.get_training_history()
        if history:
            print(f"\n📚 訓練歷史 (最近5次):")
            for i, record in enumerate(history[:5], 1):
                import datetime
                dt = datetime.datetime.fromtimestamp(record['timestamp'])
                print(f"  {i}. {record['version']}")
                print(f"     時間: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"     大小: {record['size_mb']:.2f} MB")
        else:
            print(f"\n📚 訓練歷史: 尚未訓練")

        # 當前模型
        current_path = self.model_save_dir / "xgboost_current.pkl"
        if current_path.exists():
            size_mb = current_path.stat().st_size / (1024 * 1024)
            print(f"\n🎯 當前部署模型:")
            print(f"  • 路徑: {current_path}")
            print(f"  • 大小: {size_mb:.2f} MB")
        else:
            print(f"\n🎯 當前部署模型: 無")

        print(f"\n{'='*80}\n")


def main():
    """測試持續訓練系統"""
    print("TSAR-RAPTOR Continuous Training System - 持續訓練系統測試")
    print("="*80)

    if not XGBOOST_AVAILABLE:
        print("❌ XGBoost 未安裝，請先安裝: pip install xgboost")
        return

    # 創建訓練器
    trainer = ContinuousTrainer(
        retrain_threshold=10  # 測試用較低閾值
    )

    # 顯示當前狀態
    trainer.show_training_status()

    # 檢查是否需要重訓練
    print("\n檢查重訓練條件...")
    new_model_path = trainer.check_and_retrain(force=False)

    if new_model_path:
        print(f"\n✅ 訓練完成，新模型: {new_model_path}")
        trainer.show_training_status()
    else:
        print(f"\n⏸️  未達到重訓練條件")


if __name__ == "__main__":
    main()
