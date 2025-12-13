#!/usr/bin/env python3
"""
Human Annotation System - 人眼標註系統
基於 REFOCUS_PLAN.md 設計

設計原則:
- 第一性原理: 人類視覺判斷是終極真相來源
- 猛禽3簡約: CLI介面，最小化複雜度
- 沙皇炸彈純度: 高質量標註（信心 >= 4）才用於訓練

功能:
1. 管理人工標註佇列（GRAY_ZONE 視頻）
2. 提供標註介面（視頻播放 + AI結果展示）
3. 存儲標註數據（SQLite）
4. 篩選高質量標註用於持續訓練
"""

import os
import sys
import time
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import subprocess
import platform

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class HumanAnnotation:
    """人類標註結果"""
    video_path: str
    ai_prediction: float  # AI預測的 AI_P (0-100)
    ai_confidence: float  # AI的信心度 (0-1)
    human_label: str  # 'real', 'ai', 'uncertain'
    human_confidence: int  # 1-5 信心等級
    notes: str  # 備註
    timestamp: float  # Unix timestamp
    annotator_id: str  # 標註者ID（可選）
    shap_top_reasons: str  # JSON格式的SHAP前3原因


class AnnotationDatabase:
    """標註數據庫管理（SQLite）"""

    def __init__(self, db_path: str = "data/annotations.db"):
        self.db_path = Path(project_root) / db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """初始化數據庫表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 創建標註表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_path TEXT NOT NULL,
                ai_prediction REAL NOT NULL,
                ai_confidence REAL NOT NULL,
                human_label TEXT NOT NULL,
                human_confidence INTEGER NOT NULL,
                notes TEXT,
                timestamp REAL NOT NULL,
                annotator_id TEXT,
                shap_top_reasons TEXT,
                used_for_training BOOLEAN DEFAULT 0
            )
        ''')

        # 創建索引
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_video_path
            ON annotations(video_path)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_human_label
            ON annotations(human_label)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_used_for_training
            ON annotations(used_for_training)
        ''')

        conn.commit()
        conn.close()
        logger.info(f"數據庫初始化完成: {self.db_path}")

    def save_annotation(self, annotation: HumanAnnotation) -> int:
        """保存標註"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO annotations
            (video_path, ai_prediction, ai_confidence, human_label,
             human_confidence, notes, timestamp, annotator_id, shap_top_reasons)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            annotation.video_path,
            annotation.ai_prediction,
            annotation.ai_confidence,
            annotation.human_label,
            annotation.human_confidence,
            annotation.notes,
            annotation.timestamp,
            annotation.annotator_id,
            annotation.shap_top_reasons
        ))

        annotation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        logger.info(f"標註已保存: ID={annotation_id}, 視頻={os.path.basename(annotation.video_path)}")
        return annotation_id

    def get_high_quality_annotations(self, min_confidence: int = 4) -> List[Dict]:
        """獲取高質量標註（用於訓練）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM annotations
            WHERE human_confidence >= ? AND human_label IN ('real', 'ai')
            ORDER BY timestamp DESC
        ''', (min_confidence,))

        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        logger.info(f"獲取高質量標註: {len(results)} 條（信心 >= {min_confidence}）")
        return results

    def get_annotation_stats(self) -> Dict:
        """獲取標註統計信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 總數
        cursor.execute('SELECT COUNT(*) FROM annotations')
        total = cursor.fetchone()[0]

        # 各標籤數量
        cursor.execute('''
            SELECT human_label, COUNT(*)
            FROM annotations
            GROUP BY human_label
        ''')
        label_counts = dict(cursor.fetchall())

        # 高質量標註數量
        cursor.execute('''
            SELECT COUNT(*) FROM annotations
            WHERE human_confidence >= 4 AND human_label IN ('real', 'ai')
        ''')
        high_quality = cursor.fetchone()[0]

        # 已用於訓練的數量
        cursor.execute('''
            SELECT COUNT(*) FROM annotations
            WHERE used_for_training = 1
        ''')
        used_for_training = cursor.fetchone()[0]

        conn.close()

        return {
            'total': total,
            'label_counts': label_counts,
            'high_quality': high_quality,
            'used_for_training': used_for_training,
            'pending_training': high_quality - used_for_training
        }

    def mark_as_used_for_training(self, annotation_ids: List[int]):
        """標記標註已用於訓練"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        placeholders = ','.join('?' * len(annotation_ids))
        cursor.execute(f'''
            UPDATE annotations
            SET used_for_training = 1
            WHERE id IN ({placeholders})
        ''', annotation_ids)

        conn.commit()
        conn.close()
        logger.info(f"標記 {len(annotation_ids)} 條標註為已訓練")


class HumanAnnotator:
    """人工標註介面"""

    def __init__(self, annotator_id: str = "default"):
        self.db = AnnotationDatabase()
        self.annotator_id = annotator_id
        logger.info(f"人工標註器初始化完成 (標註者: {annotator_id})")

    def annotate_video(
        self,
        video_path: str,
        ai_result: Dict,
        auto_play: bool = True
    ) -> Optional[HumanAnnotation]:
        """
        標註單個視頻

        Args:
            video_path: 視頻文件路徑
            ai_result: AI檢測結果字典，包含:
                - ai_probability: float
                - confidence: float
                - top_reasons: List[Tuple[str, float]]
            auto_play: 是否自動播放視頻

        Returns:
            HumanAnnotation 或 None（如果跳過）
        """
        # 檢查文件是否存在
        if not os.path.exists(video_path):
            logger.error(f"視頻文件不存在: {video_path}")
            return None

        # 顯示標註介面
        self._display_annotation_interface(video_path, ai_result)

        # 播放視頻
        if auto_play:
            self._play_video(video_path)

        # 獲取人類標註
        annotation = self._collect_human_input(video_path, ai_result)

        # 保存標註
        if annotation:
            self.db.save_annotation(annotation)

        return annotation

    def _display_annotation_interface(self, video_path: str, ai_result: Dict):
        """顯示標註介面"""
        print(f"\n{'='*80}")
        print(f"{'人工標註介面 - TSAR-RAPTOR Human Annotation'.center(80)}")
        print(f"{'='*80}")
        print(f"\n📹 視頻: {os.path.basename(video_path)}")
        print(f"📍 路徑: {video_path}")
        print(f"\n{'─'*80}")
        print(f"🤖 AI 預測結果:")
        print(f"  • AI Probability: {ai_result.get('ai_probability', 0):.1f}%")
        print(f"  • Confidence: {ai_result.get('confidence', 0):.2f}")

        # 顯示 SHAP Top 3 原因
        top_reasons = ai_result.get('top_reasons', [])
        if top_reasons:
            print(f"\n  📊 Top 3 檢測原因 (SHAP):")
            for i, (feature, score) in enumerate(top_reasons[:3], 1):
                print(f"     {i}. {feature}: {score:.1f}")

        print(f"{'─'*80}\n")

    def _play_video(self, video_path: str):
        """播放視頻（使用系統默認播放器）"""
        try:
            system = platform.system()
            if system == 'Windows':
                os.startfile(video_path)
            elif system == 'Darwin':  # macOS
                subprocess.run(['open', video_path])
            elif system == 'Linux':
                subprocess.run(['xdg-open', video_path])
            logger.info(f"視頻播放: {os.path.basename(video_path)}")
        except Exception as e:
            logger.error(f"無法播放視頻: {e}")
            print(f"⚠️  無法自動播放視頻，請手動打開: {video_path}")

    def _collect_human_input(
        self,
        video_path: str,
        ai_result: Dict
    ) -> Optional[HumanAnnotation]:
        """收集人類標註輸入"""
        # 標註標籤
        while True:
            label_input = input("👤 人類判斷 (r=Real真實, a=AI生成, u=Uncertain不確定, s=Skip跳過): ").lower().strip()
            if label_input == 's':
                print("⏭️  跳過此視頻")
                return None
            elif label_input in ['r', 'a', 'u']:
                label_map = {'r': 'real', 'a': 'ai', 'u': 'uncertain'}
                label = label_map[label_input]
                break
            else:
                print("❌ 無效輸入，請輸入 r/a/u/s")

        # 信心等級
        while True:
            try:
                confidence_input = input("🎯 信心等級 (1-5, 數字越大越確定): ").strip()
                confidence = int(confidence_input)
                if 1 <= confidence <= 5:
                    break
                else:
                    print("❌ 請輸入 1-5 之間的數字")
            except ValueError:
                print("❌ 請輸入有效的數字")

        # 備註
        notes = input("📝 備註（可選，直接按Enter跳過）: ").strip()

        # 顯示確認信息
        label_emoji = {'real': '✅', 'ai': '🤖', 'uncertain': '❓'}
        confidence_stars = '⭐' * confidence
        print(f"\n✓ 標註完成: {label_emoji[label]} {label.upper()} | {confidence_stars} ({confidence}/5)")
        if notes:
            print(f"  備註: {notes}")

        # 創建標註對象
        import json
        annotation = HumanAnnotation(
            video_path=video_path,
            ai_prediction=ai_result.get('ai_probability', 0),
            ai_confidence=ai_result.get('confidence', 0),
            human_label=label,
            human_confidence=confidence,
            notes=notes,
            timestamp=time.time(),
            annotator_id=self.annotator_id,
            shap_top_reasons=json.dumps(ai_result.get('top_reasons', [])[:3])
        )

        return annotation

    def batch_annotate(self, video_results: List[Tuple[str, Dict]]) -> int:
        """
        批量標註多個視頻

        Args:
            video_results: List of (video_path, ai_result) tuples

        Returns:
            完成的標註數量
        """
        total = len(video_results)
        completed = 0

        print(f"\n{'='*80}")
        print(f"開始批量標註: {total} 個視頻")
        print(f"{'='*80}\n")

        for i, (video_path, ai_result) in enumerate(video_results, 1):
            print(f"\n進度: [{i}/{total}]")

            annotation = self.annotate_video(video_path, ai_result)
            if annotation:
                completed += 1

            # 每5個視頻顯示一次統計
            if i % 5 == 0:
                self._show_progress_stats(completed, i)

        # 最終統計
        print(f"\n{'='*80}")
        print(f"批量標註完成:")
        print(f"  • 總計: {total} 個視頻")
        print(f"  • 已標註: {completed} 個")
        print(f"  • 跳過: {total - completed} 個")
        print(f"{'='*80}\n")

        return completed

    def _show_progress_stats(self, completed: int, total_processed: int):
        """顯示進度統計"""
        stats = self.db.get_annotation_stats()
        print(f"\n📊 當前標註統計:")
        print(f"  • 數據庫總計: {stats['total']} 條")
        print(f"  • 高質量標註: {stats['high_quality']} 條（信心 >= 4）")
        print(f"  • 待訓練: {stats['pending_training']} 條")

        if stats['label_counts']:
            print(f"  • 標籤分布: ", end="")
            for label, count in stats['label_counts'].items():
                print(f"{label}={count} ", end="")
            print()

    def show_statistics(self):
        """顯示完整統計信息"""
        stats = self.db.get_annotation_stats()

        print(f"\n{'='*80}")
        print(f"{'標註數據庫統計'.center(80)}")
        print(f"{'='*80}")
        print(f"\n📊 總體統計:")
        print(f"  • 總標註數: {stats['total']}")
        print(f"  • 高質量標註: {stats['high_quality']} (信心 >= 4)")
        print(f"  • 已用於訓練: {stats['used_for_training']}")
        print(f"  • 待訓練: {stats['pending_training']}")

        print(f"\n📊 標籤分布:")
        for label, count in stats['label_counts'].items():
            percentage = (count / stats['total'] * 100) if stats['total'] > 0 else 0
            bar = '█' * int(percentage / 2)
            print(f"  • {label:12s}: {count:4d} ({percentage:5.1f}%) {bar}")

        print(f"\n{'='*80}\n")


def main():
    """測試人工標註系統"""
    print("TSAR-RAPTOR Human Annotation System - 人眼標註系統測試")
    print("="*80)

    # 創建標註器
    annotator = HumanAnnotator(annotator_id="test_user")

    # 顯示當前統計
    annotator.show_statistics()

    # 測試標註（需要提供實際視頻路徑和AI結果）
    test_video = r"C:\Users\s_robby518\Documents\trae_projects\ai testing\input\a.mp4"
    test_ai_result = {
        'ai_probability': 75.5,
        'confidence': 0.85,
        'top_reasons': [
            ('model_fingerprint_detector', 88.2),
            ('frequency_analyzer', 72.1),
            ('physics_violation_detector', 65.3)
        ]
    }

    if os.path.exists(test_video):
        print(f"\n測試視頻: {test_video}")
        annotation = annotator.annotate_video(test_video, test_ai_result, auto_play=False)

        if annotation:
            print("\n✅ 測試標註成功")
            annotator.show_statistics()
    else:
        print(f"\n⚠️  測試視頻不存在: {test_video}")
        print("請修改 test_video 路徑後重試")


if __name__ == "__main__":
    main()
