#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSAR-RAPTOR Big Data Analyzer - Excel C生成器
分析AI vs 真實視頻的統計差異

設計原則:
- 第一性原理: 從數據中發現物理規律
- 沙皇炸彈: 多維度對比，級聯分析
- 猛禽3: 自動化洞察，可視化輸出

Excel C 分析內容:
1. 描述性統計: 平均值, 標準差, 最大值, 最小值
2. 對比分析: Real vs AI 各特徵差異
3. 顯著性檢驗: t-test, 效應量
4. 特徵排序: 按區分能力排序
5. 可視化建議: 哪些特徵最能區分AI
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BigDataAnalyzer:
    """大數據分析引擎"""

    def __init__(
        self,
        excel_b_path: str,
        output_excel_c: str = "excel_c_analysis.xlsx"
    ):
        """
        Args:
            excel_b_path: Excel B 路徑（特徵數據）
            output_excel_c: Excel C 輸出路徑（分析結果）
        """
        self.excel_b_path = Path(excel_b_path)
        self.output_excel_c = Path(output_excel_c)

        logger.info("大數據分析器初始化完成")
        logger.info(f"  • Excel B: {self.excel_b_path}")
        logger.info(f"  • Excel C: {self.output_excel_c}")

    def load_data(self) -> pd.DataFrame:
        """加載Excel B數據"""
        if not self.excel_b_path.exists():
            logger.error(f"❌ Excel B 不存在: {self.excel_b_path}")
            return pd.DataFrame()

        df = pd.read_excel(self.excel_b_path)
        logger.info(f"✅ 已加載 {len(df)} 條數據")

        # 過濾 Real 和 AI（排除 Uncertain）
        df_filtered = df[df['label'].isin(['real', 'ai'])]
        logger.info(f"   過濾後: Real={len(df_filtered[df_filtered['label']=='real'])}, "
                   f"AI={len(df_filtered[df_filtered['label']=='ai'])}")

        return df_filtered

    def descriptive_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        描述性統計（分組：Real vs AI）

        Returns:
            DataFrame with columns: feature, real_mean, real_std, ai_mean, ai_std, difference
        """
        # 數值型特徵列
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        # 排除 video_id 等非特徵列
        exclude_cols = ['video_id', 'file_size_mb']
        feature_cols = [f for f in numeric_features if f not in exclude_cols]

        results = []

        for feature in feature_cols:
            real_data = df[df['label'] == 'real'][feature].dropna()
            ai_data = df[df['label'] == 'ai'][feature].dropna()

            if len(real_data) == 0 or len(ai_data) == 0:
                continue

            result = {
                'feature': feature,
                'real_mean': real_data.mean(),
                'real_std': real_data.std(),
                'real_min': real_data.min(),
                'real_max': real_data.max(),
                'ai_mean': ai_data.mean(),
                'ai_std': ai_data.std(),
                'ai_min': ai_data.min(),
                'ai_max': ai_data.max(),
                'difference': abs(real_data.mean() - ai_data.mean()),
                'difference_pct': abs(real_data.mean() - ai_data.mean()) / (real_data.mean() + 1e-10) * 100
            }

            results.append(result)

        df_stats = pd.DataFrame(results)
        return df_stats

    def significance_testing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        顯著性檢驗（t-test + 效應量）

        Returns:
            DataFrame with columns: feature, t_statistic, p_value, cohen_d, significant
        """
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['video_id', 'file_size_mb']
        feature_cols = [f for f in numeric_features if f not in exclude_cols]

        results = []

        for feature in feature_cols:
            real_data = df[df['label'] == 'real'][feature].dropna()
            ai_data = df[df['label'] == 'ai'][feature].dropna()

            if len(real_data) < 2 or len(ai_data) < 2:
                continue

            # t-test
            t_stat, p_value = stats.ttest_ind(real_data, ai_data)

            # Cohen's d (效應量)
            pooled_std = np.sqrt(((len(real_data)-1)*real_data.std()**2 + (len(ai_data)-1)*ai_data.std()**2) /
                                 (len(real_data) + len(ai_data) - 2))
            cohen_d = abs(real_data.mean() - ai_data.mean()) / (pooled_std + 1e-10)

            # 顯著性判定
            significant = 'Yes' if p_value < 0.05 else 'No'

            # 效應量大小
            if cohen_d >= 0.8:
                effect_size = 'Large'
            elif cohen_d >= 0.5:
                effect_size = 'Medium'
            elif cohen_d >= 0.2:
                effect_size = 'Small'
            else:
                effect_size = 'Negligible'

            result = {
                'feature': feature,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohen_d': cohen_d,
                'effect_size': effect_size,
                'significant': significant
            }

            results.append(result)

        df_sig = pd.DataFrame(results)
        return df_sig

    def feature_ranking(self, df_stats: pd.DataFrame, df_sig: pd.DataFrame) -> pd.DataFrame:
        """
        特徵排序（按區分能力）

        排序依據:
        1. Cohen's d (效應量)
        2. p-value (顯著性)
        3. Difference (絕對差異)

        Returns:
            DataFrame with top features
        """
        # 合併統計和顯著性
        df_merged = pd.merge(df_stats, df_sig, on='feature', how='inner')

        # 計算綜合分數
        # 歸一化 Cohen's d, 1/p_value, difference_pct
        df_merged['cohen_d_norm'] = (df_merged['cohen_d'] - df_merged['cohen_d'].min()) / \
                                    (df_merged['cohen_d'].max() - df_merged['cohen_d'].min() + 1e-10)

        df_merged['p_value_inv'] = 1 / (df_merged['p_value'] + 1e-10)
        df_merged['p_value_inv_norm'] = (df_merged['p_value_inv'] - df_merged['p_value_inv'].min()) / \
                                        (df_merged['p_value_inv'].max() - df_merged['p_value_inv'].min() + 1e-10)

        df_merged['diff_pct_norm'] = (df_merged['difference_pct'] - df_merged['difference_pct'].min()) / \
                                     (df_merged['difference_pct'].max() - df_merged['difference_pct'].min() + 1e-10)

        # 綜合分數 (權重: Cohen's d 50%, p-value 30%, difference 20%)
        df_merged['discrimination_score'] = (
            0.5 * df_merged['cohen_d_norm'] +
            0.3 * df_merged['p_value_inv_norm'] +
            0.2 * df_merged['diff_pct_norm']
        )

        # 排序
        df_ranked = df_merged.sort_values('discrimination_score', ascending=False)

        # 添加排名
        df_ranked['rank'] = range(1, len(df_ranked) + 1)

        return df_ranked

    def generate_insights(self, df_ranked: pd.DataFrame) -> List[str]:
        """
        生成分析洞察

        Returns:
            洞察列表
        """
        insights = []

        # Top 5 區分特徵
        top_5 = df_ranked.head(5)
        insights.append("=== Top 5 區分特徵 ===")
        for _, row in top_5.iterrows():
            insights.append(
                f"{int(row['rank'])}. {row['feature']}: "
                f"Cohen's d={row['cohen_d']:.3f}, "
                f"p-value={row['p_value']:.4f}, "
                f"Real={row['real_mean']:.2f}, AI={row['ai_mean']:.2f}"
            )

        insights.append("")

        # 顯著特徵數量
        significant_count = len(df_ranked[df_ranked['significant'] == 'Yes'])
        insights.append(f"顯著特徵數量: {significant_count} / {len(df_ranked)}")

        # Large effect size
        large_effect = df_ranked[df_ranked['effect_size'] == 'Large']
        insights.append(f"大效應量特徵: {len(large_effect)}")

        insights.append("")

        # 建議
        insights.append("=== 優化建議 ===")
        if len(top_5) > 0:
            top_feature = top_5.iloc[0]
            insights.append(f"1. 強化檢測模組: {top_feature['feature']} (最強區分能力)")
            insights.append(f"2. 調整閾值: Real平均={top_feature['real_mean']:.2f}, AI平均={top_feature['ai_mean']:.2f}")

        if significant_count < len(df_ranked) * 0.5:
            insights.append("3. ⚠️  顯著特徵不足50%，建議增加更多檢測維度")

        return insights

    def analyze(self) -> Dict:
        """
        完整分析流程

        Returns:
            分析結果字典
        """
        # 1. 加載數據
        df = self.load_data()
        if df.empty:
            logger.error("❌ 無可用數據")
            return {}

        # 2. 描述性統計
        logger.info("📊 計算描述性統計...")
        df_stats = self.descriptive_statistics(df)

        # 3. 顯著性檢驗
        logger.info("📊 進行顯著性檢驗...")
        df_sig = self.significance_testing(df)

        # 4. 特徵排序
        logger.info("📊 計算特徵排序...")
        df_ranked = self.feature_ranking(df_stats, df_sig)

        # 5. 生成洞察
        insights = self.generate_insights(df_ranked)

        # 6. 保存到Excel C（多個Sheet）
        with pd.ExcelWriter(self.output_excel_c, engine='openpyxl') as writer:
            df_ranked.to_excel(writer, sheet_name='Feature_Ranking', index=False)
            df_stats.to_excel(writer, sheet_name='Descriptive_Stats', index=False)
            df_sig.to_excel(writer, sheet_name='Significance_Testing', index=False)

            # 洞察（文本）
            insights_df = pd.DataFrame({'Insights': insights})
            insights_df.to_excel(writer, sheet_name='Insights', index=False)

        logger.info(f"\n✅ Excel C 已生成: {self.output_excel_c}")
        logger.info(f"   Sheet: Feature_Ranking, Descriptive_Stats, Significance_Testing, Insights")

        # 打印洞察
        print(f"\n{'='*80}")
        print("分析洞察:")
        print(f"{'='*80}")
        for insight in insights:
            print(insight)
        print(f"{'='*80}\n")

        return {
            'ranked_features': df_ranked,
            'stats': df_stats,
            'significance': df_sig,
            'insights': insights
        }


def main():
    """測試大數據分析器"""
    import argparse

    parser = argparse.ArgumentParser(description="TikTok視頻大數據分析器")
    parser.add_argument(
        '--excel-b',
        type=str,
        default='../../data/tiktok_labels/excel_b_features.xlsx',
        help='Excel B 路徑'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../../data/tiktok_labels/excel_c_analysis.xlsx',
        help='輸出 Excel C 路徑'
    )

    args = parser.parse_args()

    # 創建分析器
    analyzer = BigDataAnalyzer(
        excel_b_path=args.excel_b,
        output_excel_c=args.output
    )

    # 執行分析
    results = analyzer.analyze()

    if results:
        print(f"\n✅ 分析完成！")
        print(f"   Top 5 區分特徵:")
        top_5 = results['ranked_features'].head(5)
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"      {i}. {row['feature']}: 分數={row['discrimination_score']:.3f}")


if __name__ == "__main__":
    main()
