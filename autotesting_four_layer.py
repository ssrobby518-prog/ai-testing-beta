#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四層架構總控系統 (Four-Layer Architecture Master Controller)
===========================================================

基於第一性原理的AI視頻檢測與資料生成系統

四層架構：
┌─────────────────────────────────────────────────────────────┐
│ 第一層：候選影片池（2025+）                                   │
│ - 所有可能進入系統的短影片來源                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 第二層：生成機制分析層                                        │
│ - AI / Real / Uncertain / Movie-Anime                        │
│ - 純物理檢測，不受商業特徵污染                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 第三層：資料仲裁與去噪層（核心）                               │
│ ├─ AI Pool → 可進訓練（生成機制：AI）                         │
│ ├─ Real Pool → 可進訓練（生成機制：真人）                     │
│ ├─ Disagreement Pool → 高價值樣本（唯一允許人工介入）         │
│ └─ Excluded Pool → 電影/動畫/影集（永久排除）                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
        【人工僅介入 Disagreement Pool】
        （Tinder 式快速確認，修正生成機制標籤）
                            ↓
      【生成機制標籤穩定集（Stable Generation Dataset）】
      （AI / Real 標籤已收斂、可重現、可月度擴增）
                            ↓
        ────────────────────────────────
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 第四層：經濟行為分類層（獨立）                                 │
│ ├─ 粉片 Pool（Account Growth / 漲粉導向）                    │
│ ├─ 貨片 Pool（Commerce / 轉化導向）                          │
│ └─ 無效內容 Pool（不具經濟價值）                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
        【經濟子模型訓練 / 帳號與帶貨策略優化】

設計原則：
1. 正交維度分離：生成機制（Detection）vs 經濟行為（Monetization）
2. 數據純度防火牆：第三層仲裁確保標籤收斂
3. 長期演化能力：月度擴增 + 雙模型解耦

Author: Claude Sonnet 4.5
Date: 2025-12-20
"""

import os
import logging
import pandas as pd
from datetime import datetime
from typing import List

from core.generation_analyzer import GenerationMechanismAnalyzer
from core.four_layer_system import (
    FourLayerSystemCoordinator,
    PoolType,
    GenerationMechanism,
    EconomicBehavior
)

logging.basicConfig(level=logging.INFO)


class FourLayerMasterController:
    """
    四層架構總控制器

    職責：
    1. 協調第二層（生成機制分析）
    2. 協調第三層（資料仲裁）
    3. 協調第四層（經濟行為分類）
    4. 管理數據流向（AI Pool/Real Pool/Disagreement Pool/Excluded Pool）
    5. 生成訓練數據集
    """

    def __init__(self,
                 input_dir: str = "input",
                 output_dir: str = "output/four_layer",
                 data_pools_dir: str = "data_pools"):
        """
        初始化四層系統

        Args:
            input_dir: 輸入視頻目錄
            output_dir: 輸出報告目錄
            data_pools_dir: 資料池目錄（AI Pool/Real Pool/etc）
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.data_pools_dir = data_pools_dir

        # 創建目錄結構
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(data_pools_dir, exist_ok=True)

        # 創建資料池子目錄
        self.pool_dirs = {
            PoolType.AI_POOL: os.path.join(data_pools_dir, "ai_pool"),
            PoolType.REAL_POOL: os.path.join(data_pools_dir, "real_pool"),
            PoolType.DISAGREEMENT_POOL: os.path.join(data_pools_dir, "disagreement_pool"),
            PoolType.EXCLUDED_POOL: os.path.join(data_pools_dir, "excluded_pool")
        }

        for pool_dir in self.pool_dirs.values():
            os.makedirs(pool_dir, exist_ok=True)

        # 初始化各層組件
        self.generation_analyzer = GenerationMechanismAnalyzer()
        self.four_layer_coordinator = FourLayerSystemCoordinator(output_dir)

        # 統計信息
        self.stats = {
            'total_processed': 0,
            'ai_pool': 0,
            'real_pool': 0,
            'disagreement_pool': 0,
            'excluded_pool': 0
        }

    def _get_input_files(self) -> List[str]:
        """獲取輸入目錄中的所有視頻文件"""
        files = [
            os.path.join(self.input_dir, f)
            for f in os.listdir(self.input_dir)
            if os.path.isfile(os.path.join(self.input_dir, f))
            and f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]

        logging.info(f"[總控] 找到 {len(files)} 個視頻文件")
        return files

    def _move_to_pool(self, file_path: str, pool_type: PoolType):
        """將視頻文件移動到相應的資料池（可選，暫時不移動）"""
        # TODO: 實現文件移動邏輯
        # 目前只記錄，不實際移動文件
        pool_dir = self.pool_dirs[pool_type]
        logging.info(f"[總控] 文件 {os.path.basename(file_path)} → {pool_type.value}")

    def process_all_videos(self):
        """處理所有視頻，執行四層分析"""
        print("\n" + "="*80)
        print("四層架構總控系統啟動")
        print("="*80 + "\n")

        # 獲取所有視頻文件
        video_files = self._get_input_files()

        if not video_files:
            logging.warning("[總控] 未找到視頻文件，請將視頻放入 input/ 目錄")
            return

        # 處理每個視頻
        all_reports = []

        for idx, file_path in enumerate(video_files, 1):
            try:
                print(f"\n{'='*80}")
                print(f"處理進度: {idx}/{len(video_files)} - {os.path.basename(file_path)}")
                print(f"{'='*80}\n")

                # 第二層：生成機制分析
                generation_result = self.generation_analyzer.analyze(file_path)

                # 第三層 + 第四層：仲裁與分類
                report = self.four_layer_coordinator.process_video(generation_result)

                # 更新統計
                pool_type = report.arbitration.pool_assignment
                self.stats['total_processed'] += 1

                if pool_type == PoolType.AI_POOL:
                    self.stats['ai_pool'] += 1
                elif pool_type == PoolType.REAL_POOL:
                    self.stats['real_pool'] += 1
                elif pool_type == PoolType.DISAGREEMENT_POOL:
                    self.stats['disagreement_pool'] += 1
                elif pool_type == PoolType.EXCLUDED_POOL:
                    self.stats['excluded_pool'] += 1

                # 移動到相應的資料池（可選）
                # self._move_to_pool(file_path, pool_type)

                all_reports.append(report)

            except Exception as e:
                logging.error(f"[總控] 處理 {file_path} 時發生錯誤: {e}")
                continue

        # 生成匯總報告
        self._generate_summary_report(all_reports)

        # 顯示統計信息
        self._print_statistics()

    def _generate_summary_report(self, reports):
        """生成Excel匯總報告"""
        if not reports:
            return

        logging.info("\n[總控] 生成匯總報告...")

        # 構建DataFrame
        rows = []
        for report in reports:
            gen = report.generation_analysis
            arb = report.arbitration
            eco = report.economic_classification

            row = {
                '文件名': gen.metadata.file_name,
                '文件路徑': gen.metadata.file_path,
                'Bitrate (Mbps)': gen.metadata.bitrate / 1_000_000,
                'FPS': gen.metadata.fps,
                '生成機制': gen.generation_mechanism.value,
                'AI概率': gen.ai_probability,
                '置信度': gen.confidence,
                '資料池分配': arb.pool_assignment.value,
                '模組一致性': arb.module_consensus_score,
                '需要人工審核': arb.human_review_required,
                '仲裁理由': arb.decision_rationale,
                'Face Presence': gen.face_presence,
                'Static Ratio': gen.static_ratio,
                '經濟行為': eco.economic_behavior.value if eco else 'N/A'
            }

            # 添加各模組分數
            for module_name, score in gen.module_scores.items():
                row[f'Module_{module_name}'] = score

            rows.append(row)

        df = pd.DataFrame(rows)

        # 保存為Excel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(self.output_dir, f"四層分析匯總_{timestamp}.xlsx")

        try:
            df.to_excel(summary_file, index=False)
            logging.info(f"[總控] 匯總報告已保存: {summary_file}")
        except Exception as e:
            logging.error(f"[總控] 保存匯總報告失敗: {e}")

        # 生成分Pool的報告
        self._generate_pool_reports(df)

    def _generate_pool_reports(self, df: pd.DataFrame):
        """生成各個Pool的獨立報告"""
        for pool_type in PoolType:
            pool_df = df[df['資料池分配'] == pool_type.value]

            if len(pool_df) > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pool_file = os.path.join(
                    self.pool_dirs[pool_type],
                    f"{pool_type.value}_report_{timestamp}.xlsx"
                )

                try:
                    pool_df.to_excel(pool_file, index=False)
                    logging.info(f"[總控] {pool_type.value} 報告已保存: {pool_file}")
                except Exception as e:
                    logging.error(f"[總控] 保存 {pool_type.value} 報告失敗: {e}")

    def _print_statistics(self):
        """顯示統計信息"""
        print("\n" + "="*80)
        print("處理統計")
        print("="*80)
        print(f"總處理數: {self.stats['total_processed']}")
        print(f"AI Pool: {self.stats['ai_pool']} ({self.stats['ai_pool']/max(self.stats['total_processed'], 1)*100:.1f}%)")
        print(f"Real Pool: {self.stats['real_pool']} ({self.stats['real_pool']/max(self.stats['total_processed'], 1)*100:.1f}%)")
        print(f"Disagreement Pool: {self.stats['disagreement_pool']} ({self.stats['disagreement_pool']/max(self.stats['total_processed'], 1)*100:.1f}%)")
        print(f"Excluded Pool: {self.stats['excluded_pool']} ({self.stats['excluded_pool']/max(self.stats['total_processed'], 1)*100:.1f}%)")
        print("="*80 + "\n")

        if self.stats['disagreement_pool'] > 0:
            print(f"⚠️  發現 {self.stats['disagreement_pool']} 個需要人工審核的視頻")
            print(f"   請查看: {self.pool_dirs[PoolType.DISAGREEMENT_POOL]}")
            print(f"   使用 Tinder 式標註工具進行快速確認\n")

    def get_disagreement_videos(self) -> List[str]:
        """獲取需要人工審核的視頻列表"""
        # TODO: 從報告中讀取Disagreement Pool的視頻列表
        return []


def main():
    """主函數"""
    print("\n" + "="*80)
    print("四層架構 AI 視頻檢測系統")
    print("="*80)
    print("\n設計原則：")
    print("1. 正交維度分離：生成機制（Detection）vs 經濟行為（Monetization）")
    print("2. 數據純度防火牆：第三層仲裁確保標籤收斂")
    print("3. 長期演化能力：月度擴增 + 雙模型解耦\n")

    # 初始化總控制器
    controller = FourLayerMasterController()

    # 處理所有視頻
    controller.process_all_videos()

    print("\n處理完成！")
    print("\n下一步:")
    print("1. 查看 data_pools/disagreement_pool/ 中需要人工審核的視頻")
    print("2. 使用 Tinder 式標註工具確認生成機制標籤")
    print("3. 系統將自動學習並優化\n")


if __name__ == "__main__":
    main()
