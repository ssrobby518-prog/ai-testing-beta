#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSAR-RAPTOR Module Auto-Reconstructor
模組自動重構系統 - 根據Excel B/C優化AI檢測模組

設計原則:
- 第一性原理: 數據驅動優化
- 沙皇炸彈: 級聯優化，多層反饋
- 猛禽3: 自動化重構，零人工干預

功能:
1. 讀取 Excel C 的 Top 特徵排序
2. 分析哪些檢測模組對應哪些特徵
3. 自動調整模組權重和閾值
4. 生成新的配置文件
5. 可選：自動重寫模組代碼
"""

import pandas as pd
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModuleOptimizer:
    """AI檢測模組自動優化器"""

    def __init__(
        self,
        excel_c_path: str,
        modules_dir: str = "../../../modules",
        config_output: str = "optimized_config.json"
    ):
        """
        Args:
            excel_c_path: Excel C 路徑（分析結果）
            modules_dir: AI檢測模組目錄
            config_output: 優化配置輸出路徑
        """
        self.excel_c_path = Path(excel_c_path)
        self.modules_dir = Path(modules_dir)
        self.config_output = Path(config_output)

        # 特徵 → 模組映射
        self.feature_to_module = {
            # 頻域特徵 → Frequency Analyzer
            'dct_energy': 'frequency_analyzer',
            'spectral_entropy': 'frequency_analyzer',

            # 運動特徵 → Physics Violation Detector + Optical Flow
            'avg_optical_flow': 'physics_violation_detector',
            'scene_changes': 'physics_violation_detector',

            # 視覺特徵 → Texture Noise Detector
            'avg_brightness': 'texture_noise_detector',
            'avg_contrast': 'texture_noise_detector',
            'avg_saturation': 'texture_noise_detector',
            'avg_blur': 'texture_noise_detector',

            # 音頻特徵 → AV Sync Verifier
            'audio_sample_rate': 'av_sync_verifier',
            'audio_channels': 'av_sync_verifier',
            'audio_bitrate': 'av_sync_verifier',

            # 視頻基本特徵 → Metadata Extractor
            'fps': 'metadata_extractor',
            'bitrate': 'metadata_extractor',
            'duration': 'metadata_extractor',
        }

        logger.info("模組優化器初始化完成")

    def load_feature_ranking(self) -> pd.DataFrame:
        """
        加載 Excel C 的特徵排序

        Returns:
            DataFrame (Feature_Ranking sheet)
        """
        if not self.excel_c_path.exists():
            logger.error(f"❌ Excel C 不存在: {self.excel_c_path}")
            return pd.DataFrame()

        df = pd.read_excel(self.excel_c_path, sheet_name='Feature_Ranking')
        logger.info(f"✅ 已加載 {len(df)} 個特徵排序")
        return df

    def calculate_module_importance(self, df_ranking: pd.DataFrame) -> Dict[str, float]:
        """
        計算各模組的重要性分數

        基於特徵的 discrimination_score 加總

        Returns:
            {module_name: importance_score}
        """
        module_scores = {}

        for _, row in df_ranking.iterrows():
            feature = row['feature']
            score = row['discrimination_score']

            # 查找對應模組
            module = self.feature_to_module.get(feature, 'unknown')
            if module == 'unknown':
                continue

            # 累積分數
            if module not in module_scores:
                module_scores[module] = 0.0
            module_scores[module] += score

        # 歸一化
        total = sum(module_scores.values())
        if total > 0:
            module_scores = {k: v/total for k, v in module_scores.items()}

        # 排序
        module_scores = dict(sorted(module_scores.items(), key=lambda x: x[1], reverse=True))

        logger.info(f"✅ 計算模組重要性完成")
        for module, score in module_scores.items():
            logger.info(f"   {module}: {score:.3f}")

        return module_scores

    def generate_threshold_recommendations(self, df_ranking: pd.DataFrame) -> Dict[str, Dict]:
        """
        生成閾值建議

        基於 Excel C 的 real_mean 和 ai_mean

        Returns:
            {
                feature: {
                    'real_mean': float,
                    'ai_mean': float,
                    'suggested_threshold': float,
                    'direction': 'higher_is_ai' | 'lower_is_ai'
                }
            }
        """
        recommendations = {}

        for _, row in df_ranking.iterrows():
            feature = row['feature']
            real_mean = row['real_mean']
            ai_mean = row['ai_mean']

            # 計算建議閾值（中點）
            threshold = (real_mean + ai_mean) / 2.0

            # 判斷方向
            if ai_mean > real_mean:
                direction = 'higher_is_ai'
            else:
                direction = 'lower_is_ai'

            recommendations[feature] = {
                'real_mean': real_mean,
                'ai_mean': ai_mean,
                'suggested_threshold': threshold,
                'direction': direction,
                'cohen_d': row['cohen_d']
            }

        return recommendations

    def generate_optimized_config(
        self,
        module_importance: Dict[str, float],
        threshold_recommendations: Dict[str, Dict]
    ) -> Dict:
        """
        生成優化配置文件

        Returns:
            配置字典
        """
        config = {
            'meta': {
                'version': '2.0.0',
                'generated_by': 'TSAR-RAPTOR Auto-Reconstructor',
                'optimization_source': str(self.excel_c_path),
                'description': '基於數據驅動的自動優化配置'
            },

            'module_weights': module_importance,

            'thresholds': threshold_recommendations,

            'stage_weights': {
                'stage1': 0.40,  # 保持沙皇炸彈原則
                'stage2': 0.30,
                'stage3': 0.30
            },

            'recommendations': []
        }

        # 生成建議
        top_modules = list(module_importance.keys())[:3]
        config['recommendations'].append(
            f"Top 3 重要模組: {', '.join(top_modules)}"
        )

        # 找出最強區分特徵
        top_feature = max(threshold_recommendations.items(),
                         key=lambda x: x[1]['cohen_d'])[0]
        config['recommendations'].append(
            f"最強區分特徵: {top_feature} (Cohen's d={threshold_recommendations[top_feature]['cohen_d']:.3f})"
        )

        return config

    def save_config(self, config: Dict):
        """保存配置到JSON文件"""
        with open(self.config_output, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ 優化配置已保存: {self.config_output}")

    def generate_code_snippets(self, config: Dict) -> str:
        """
        生成代碼片段建議

        Returns:
            Python代碼字符串
        """
        code = f"""# TSAR-RAPTOR Auto-Generated Configuration
# 生成時間: {pd.Timestamp.now()}
# 數據來源: {self.excel_c_path}

# ========== 模組權重（基於數據分析） ==========
MODULE_WEIGHTS = {{
"""
        for module, weight in config['module_weights'].items():
            code += f"    '{module}': {weight:.4f},\n"

        code += "}\n\n"

        code += f"""# ========== 閾值建議（基於統計分析） ==========
THRESHOLDS = {{
"""
        for feature, rec in config['thresholds'].items():
            code += f"    '{feature}': {{\n"
            code += f"        'value': {rec['suggested_threshold']:.4f},\n"
            code += f"        'direction': '{rec['direction']}',\n"
            code += f"        'real_mean': {rec['real_mean']:.4f},\n"
            code += f"        'ai_mean': {rec['ai_mean']:.4f},\n"
            code += f"        'effect_size': {rec['cohen_d']:.4f}\n"
            code += f"    }},\n"

        code += "}\n\n"

        code += "# ========== 使用建議 ==========\n"
        for rec in config['recommendations']:
            code += f"# {rec}\n"

        return code

    def optimize(self) -> Dict:
        """
        完整優化流程

        Returns:
            優化配置字典
        """
        logger.info("🚀 開始模組自動優化...")

        # 1. 加載特徵排序
        df_ranking = self.load_feature_ranking()
        if df_ranking.empty:
            return {}

        # 2. 計算模組重要性
        module_importance = self.calculate_module_importance(df_ranking)

        # 3. 生成閾值建議
        threshold_recommendations = self.generate_threshold_recommendations(df_ranking)

        # 4. 生成優化配置
        config = self.generate_optimized_config(module_importance, threshold_recommendations)

        # 5. 保存配置
        self.save_config(config)

        # 6. 生成代碼片段
        code_snippet = self.generate_code_snippets(config)
        code_output = self.config_output.parent / "optimized_code_snippet.py"
        with open(code_output, 'w', encoding='utf-8') as f:
            f.write(code_snippet)
        logger.info(f"✅ 代碼片段已生成: {code_output}")

        logger.info(f"\n{'='*80}")
        logger.info("模組優化完成！")
        logger.info(f"{'='*80}")
        logger.info(f"配置文件: {self.config_output}")
        logger.info(f"代碼片段: {code_output}")
        logger.info(f"\n建議：")
        for rec in config['recommendations']:
            logger.info(f"  • {rec}")
        logger.info(f"{'='*80}\n")

        return config


def main():
    """測試優化器"""
    import argparse

    parser = argparse.ArgumentParser(description="AI檢測模組自動優化器")
    parser.add_argument(
        '--excel-c',
        type=str,
        default='../../data/tiktok_labels/excel_c_analysis.xlsx',
        help='Excel C 路徑'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./optimized_config.json',
        help='輸出配置路徑'
    )

    args = parser.parse_args()

    # 創建優化器
    optimizer = ModuleOptimizer(
        excel_c_path=args.excel_c,
        config_output=args.output
    )

    # 執行優化
    config = optimizer.optimize()

    if config:
        print("\n✅ 優化成功！")
        print(f"   配置文件: {args.output}")
        print(f"\n下一步：將配置應用到 autotesting_v3.py 或現有模組")


if __name__ == "__main__":
    main()
