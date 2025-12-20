#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSAR-RAPTOR Layer 1 Pipeline
人工主導標註完整流水線

設計原則:
- 第一性原理: 人類判定 → 數據分析 → 模組優化
- 沙皇炸彈: 級聯學習，數據驅動
- 猛禽3: 一鍵執行，全自動

完整流程:
1. Chrome擴展標註 → Excel A
2. 批量下載並自動分類到文件夾
3. 特徵提取 → Excel B
4. 大數據分析 → Excel C
5. 模組自動優化
"""

import sys
from pathlib import Path
import logging
import argparse
from typing import Dict

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 導入配置
from config import (
    EXCEL_A_PATH, EXCEL_B_PATH, EXCEL_C_PATH,
    LAYER1_BASE_DIR, LAYER1_DATA_DIR,
    ensure_directories
)

# 導入各組件
from downloader.tiktok_downloader_classified import TikTokDownloaderClassified

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Layer1Pipeline:
    """Layer 1 自我學習流水線總控"""

    def __init__(self):
        """初始化"""
        # 確保所有目錄存在
        ensure_directories()

        logger.info("Layer 1 流水線初始化完成")
        logger.info(f"  • 基礎目錄: {LAYER1_BASE_DIR}")
        logger.info(f"  • Excel A: {EXCEL_A_PATH}")
        logger.info(f"  • Excel B: {EXCEL_B_PATH}")
        logger.info(f"  • Excel C: {EXCEL_C_PATH}")

    def run_redo_download(self, download: bool = True) -> Dict:
        from datetime import datetime
        import shutil
        from config import LAYER1_VIDEO_FOLDERS

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_root = (LAYER1_DATA_DIR / 'redo_backup' / ts)
        backup_root.mkdir(parents=True, exist_ok=True)

        moved = 0
        moved_by = {k: 0 for k in LAYER1_VIDEO_FOLDERS.keys()}

        for folder_key, folder in LAYER1_VIDEO_FOLDERS.items():
            if not folder.exists():
                continue
            target_dir = backup_root / folder_key
            target_dir.mkdir(parents=True, exist_ok=True)
            for p in folder.glob('*.mp4'):
                try:
                    shutil.move(str(p), str(target_dir / p.name))
                    moved += 1
                    moved_by[folder_key] = moved_by.get(folder_key, 0) + 1
                except Exception as e:
                    logger.warning(f"⚠️  備份失敗: {p} ({e})")

        moved_data = 0
        data_target = backup_root / "_data"
        for p in LAYER1_DATA_DIR.rglob('*.mp4'):
            try:
                rp = p.resolve()
                if rp.is_relative_to(backup_root.resolve()):
                    continue
            except Exception:
                pass

            try:
                rel = p.relative_to(LAYER1_DATA_DIR)
                dst = data_target / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(dst))
                moved_data += 1
            except Exception as e:
                logger.warning(f"⚠️  備份失敗: {p} ({e})")

        if not download:
            logger.info("📦 已完成備份（跳過重新下載）")
            return {
                'backup_dir': str(backup_root),
                'moved': moved,
                'moved_by': moved_by,
                'moved_data': moved_data,
                'download': {'skipped': True}
            }

        logger.info("📥 重做下載：已備份現有影片，開始重新下載...")
        downloader = TikTokDownloaderClassified(
            excel_a_path=str(EXCEL_A_PATH),
            max_workers=8
        )
        download_stats = downloader.download_from_excel_a()
        return {
            'backup_dir': str(backup_root),
            'moved': moved,
            'moved_by': moved_by,
            'moved_data': moved_data,
            'download': download_stats
        }

    def run_full_pipeline(self) -> Dict:
        """
        運行完整 Layer 1 流水線

        流程:
        1. 檢查 Excel A 是否有標註
        2. 批量下載視頻並自動分類
        3. 特徵提取 → Excel B
        4. 大數據分析 → Excel C
        5. 模組優化

        Returns:
            執行統計
        """
        logger.info(f"\n{'='*80}")
        logger.info("🚀 TSAR-RAPTOR Layer 1 人工主導標註流水線 - 啟動")
        logger.info(f"{'='*80}\n")

        stats = {}

        # Step 1: 檢查 Excel A
        if not EXCEL_A_PATH.exists():
            logger.error(f"❌ Excel A 不存在: {EXCEL_A_PATH}")
            logger.error("   請先使用 Chrome 擴展進行標註")
            return {}

        # Step 2: 批量下載並自動分類
        logger.info("📥 [Step 1/4] 批量下載並自動分類視頻...")
        downloader = TikTokDownloaderClassified(
            excel_a_path=str(EXCEL_A_PATH),
            max_workers=8
        )
        download_stats = downloader.download_from_excel_a()
        stats['download'] = download_stats

        from analyzer.feature_extractor_layer1 import FeatureExtractorLayer1
        from analyzer.big_data_analyzer import BigDataAnalyzer
        from auto_reconstructor.module_optimizer import ModuleOptimizer

        # Step 3: 特徵提取
        logger.info("\n🔬 [Step 2/4] 特徵提取...")
        extractor = FeatureExtractorLayer1(
            output_excel_b=str(EXCEL_B_PATH),
            max_workers=4,
            sample_frames=30
        )
        df_features = extractor.batch_extract()
        stats['features'] = {'total': len(df_features)}

        # Step 4: 大數據分析
        logger.info("\n📊 [Step 3/4] 大數據分析...")
        analyzer = BigDataAnalyzer(
            excel_b_path=str(EXCEL_B_PATH),
            output_excel_c=str(EXCEL_C_PATH)
        )
        analysis_results = analyzer.analyze()
        stats['analysis'] = {'features_analyzed': len(analysis_results.get('ranked_features', []))}

        # Step 5: 模組優化
        logger.info("\n⚙️  [Step 4/4] 模組自動優化...")
        optimized_config_path = LAYER1_DATA_DIR / "optimized_config.json"
        optimizer = ModuleOptimizer(
            excel_c_path=str(EXCEL_C_PATH),
            config_output=str(optimized_config_path)
        )
        optimized_config = optimizer.optimize()
        stats['optimization'] = {'modules_optimized': len(optimized_config.get('module_weights', {}))}

        # 最終統計
        logger.info(f"\n{'='*80}")
        logger.info("✅ Layer 1 流水線完成！")
        logger.info(f"{'='*80}")
        logger.info(f"  • 下載視頻: {download_stats.get('success', 0)} 成功, {download_stats.get('failed', 0)} 失敗")
        if 'by_category' in download_stats:
            logger.info(f"    分類統計:")
            logger.info(f"      - Real: {download_stats['by_category']['real']}")
            logger.info(f"      - AI: {download_stats['by_category']['ai']}")
            logger.info(f"      - Uncertain: {download_stats['by_category']['uncertain']}")
            logger.info(f"      - Movies: {download_stats['by_category']['exclude']}")
        logger.info(f"  • 提取特徵: {len(df_features)} 個視頻")
        logger.info(f"  • 分析特徵: {len(analysis_results.get('ranked_features', []))} 個特徵")
        logger.info(f"  • 優化模組: {len(optimized_config.get('module_weights', {}))} 個模組")
        logger.info(f"{'='*80}\n")

        return stats

    def run_download_only(self) -> Dict:
        if not EXCEL_A_PATH.exists():
            logger.error(f"❌ Excel A 不存在: {EXCEL_A_PATH}")
            logger.error("   請先使用 Chrome 擴展進行標註")
            return {}

        logger.info("📥 批量下載並自動分類視頻（download-only）...")
        downloader = TikTokDownloaderClassified(
            excel_a_path=str(EXCEL_A_PATH),
            max_workers=8
        )
        download_stats = downloader.download_from_excel_a()
        logger.info(f"  • 下載視頻: {download_stats.get('success', 0)} 成功, {download_stats.get('failed', 0)} 失敗")
        return {'download': download_stats}

    def run_download_detect_report(self) -> Dict:
        import json
        import os
        import re
        import shutil
        import subprocess
        import hashlib
        from datetime import datetime
        import pandas as pd

        if not EXCEL_A_PATH.exists():
            logger.error(f"❌ Excel A 不存在: {EXCEL_A_PATH}")
            return {}

        downloader = TikTokDownloaderClassified(
            excel_a_path=str(EXCEL_A_PATH),
            max_workers=8
        )
        download_stats = downloader.download_from_excel_a()

        df_a = pd.read_excel(EXCEL_A_PATH)
        url_col = '影片網址' if '影片網址' in df_a.columns else 'url'
        label_col = '判定結果' if '判定結果' in df_a.columns else 'label'
        video_id_col = '視頻ID' if '視頻ID' in df_a.columns else 'video_id'

        def _vid_from_url(u: str) -> str:
            m = re.search(r'/video/(\d+)', str(u))
            if m:
                return m.group(1)
            url_str = str(u or '').strip()
            if url_str:
                return hashlib.md5(url_str.encode('utf-8')).hexdigest()[:10]
            return ''

        def _canon(v: str) -> str:
            s = str(v).strip()
            s = re.sub(r'\.0$', '', s)
            if s.isdigit():
                try:
                    return str(int(s))
                except Exception:
                    return s.lstrip('0') or '0'
            if re.fullmatch(r'[0-9a-fA-F]{10}', s):
                return s.lower()
            return ''

        df_a[url_col] = df_a[url_col].astype(str)
        df_a[label_col] = df_a[label_col].astype(str)
        if video_id_col in df_a.columns:
            df_a[video_id_col] = df_a[video_id_col].astype(str)
        else:
            df_a[video_id_col] = ''

        df_a['__vid'] = df_a.apply(lambda r: _canon(_vid_from_url(r.get(url_col, '')) or str(r.get(video_id_col, '')).strip()), axis=1)

        label_map = dict(zip(df_a['__vid'].astype(str), df_a[label_col].astype(str).str.upper()))
        url_map = dict(zip(df_a['__vid'].astype(str), df_a[url_col].astype(str)))

        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        staging_root = (LAYER1_DATA_DIR / 'layer1_detection_runs' / run_id)
        staging_input = (staging_root / 'input')
        staging_output = (staging_root / 'output')
        staging_data = (staging_output / 'data')
        staging_input.mkdir(parents=True, exist_ok=True)
        staging_output.mkdir(parents=True, exist_ok=True)
        staging_data.mkdir(parents=True, exist_ok=True)

        videos = []
        for folder_key in ['real', 'ai', 'uncertain']:
            folder = (LAYER1_BASE_DIR / folder_key) if folder_key != 'uncertain' else (LAYER1_BASE_DIR / 'not sure')
            if not folder.exists():
                continue
            for video_path in folder.glob('*.mp4'):
                video_id = _canon(video_path.stem.split('_')[-1])
                if not video_id:
                    continue
                dst = staging_input / f'{video_id}.mp4'
                if not dst.exists():
                    shutil.copy2(video_path, dst)
                videos.append(video_id)

        videos = sorted(set(videos))

        if not videos:
            logger.warning('⚠️  沒有可用的視頻做檢測')
            return {'download': download_stats, 'evaluated': 0}

        repo_root = project_root.parent
        env = os.environ.copy()
        env['INPUT_DIR'] = str(staging_input)
        env['OUTPUT_DIR'] = str(staging_output)
        env['DATA_DIR'] = str(staging_data)
        env['MAX_TIME'] = str(env.get('MAX_TIME', '600'))

        result = subprocess.run(
            [sys.executable, str(repo_root / 'autotesting.py')],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=60 * 60
        )

        if result.returncode != 0:
            logger.error('❌ AI檢測執行失敗')
            logger.error(result.stderr[-2000:])
            return {'download': download_stats, 'evaluated': 0, 'error': 'detection_failed'}

        rows = []
        for diag_path in staging_output.glob('diagnostic_*.json'):
            try:
                data = json.loads(diag_path.read_text(encoding='utf-8'))
            except Exception:
                continue

            fp = str(data.get('file_path', ''))
            vid = Path(fp).stem
            ai_p = float(data.get('global_probability', 0.0))
            bitrate = int(data.get('video_characteristics', {}).get('bitrate', 0))
            face = float(data.get('video_characteristics', {}).get('face_presence', 0.0))
            static_ratio = float(data.get('video_characteristics', {}).get('static_ratio', 0.0))

            if ai_p <= 30:
                pred = 'REAL'
            elif ai_p <= 75:
                pred = 'UNCERTAIN'
            else:
                pred = 'AI'

            human = label_map.get(str(vid), '')
            url = url_map.get(str(vid), '')

            module_scores = data.get('module_scores', {})
            top_modules = sorted(
                [(k, float(v)) for k, v in module_scores.items()],
                key=lambda x: x[1],
                reverse=True
            )[:3]

            rows.append({
                'video_id': str(vid),
                'url': url,
                'human_label': human,
                'ai_probability': ai_p,
                'pred_label': pred,
                'bitrate': bitrate,
                'face_presence': face,
                'static_ratio': static_ratio,
                'top_modules': json.dumps(top_modules, ensure_ascii=False)
            })

        if not rows:
            logger.error('❌ AI檢測完成但未生成任何 diagnostic_*.json')
            if result.stdout:
                logger.error(result.stdout[-2000:])
            if result.stderr:
                logger.error(result.stderr[-2000:])
            return {'download': download_stats, 'evaluated': 0, 'error': 'no_diagnostics'}

        df_eval = pd.DataFrame(rows)
        eval_xlsx = LAYER1_DATA_DIR / f'layer1_ai_eval_{run_id}.xlsx'
        df_eval.to_excel(eval_xlsx, index=False)

        def bucket(b: int) -> str:
            if b <= 0:
                return 'unknown'
            if b < 800_000:
                return '<0.8'
            if b < 1_500_000:
                return '0.8-1.5'
            if b < 2_000_000:
                return '1.5-2.0'
            return '>2.0'

        df_eval['bitrate_bucket'] = df_eval['bitrate'].apply(bucket)

        def normalize_label(x: str) -> str:
            x = str(x).strip().upper()
            if x in {'REAL', 'AI', 'UNCERTAIN', 'EXCLUDE'}:
                return x
            if x in {'NOT SURE', 'NOT_SURE'}:
                return 'UNCERTAIN'
            if x in {'MOVIE', 'MOVIE/ANIME', 'MOVIES'}:
                return 'EXCLUDE'
            return x

        df_eval['human_label'] = df_eval['human_label'].apply(normalize_label)

        considered = df_eval[df_eval['human_label'].isin(['REAL', 'AI'])].copy()
        tp = int(((considered['human_label'] == 'AI') & (considered['pred_label'] == 'AI')).sum())
        tn = int(((considered['human_label'] == 'REAL') & (considered['pred_label'] == 'REAL')).sum())
        fp = int(((considered['human_label'] == 'REAL') & (considered['pred_label'] == 'AI')).sum())
        fn = int(((considered['human_label'] == 'AI') & (considered['pred_label'] == 'REAL')).sum())

        accuracy = (tp + tn) / max(len(considered), 1)

        report_lines = []
        report_lines.append('=' * 80)
        report_lines.append('LAYER 1 AI檢測對比報告')
        report_lines.append('=' * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Excel A: {EXCEL_A_PATH}")
        report_lines.append(f"Eval Excel: {eval_xlsx}")
        report_lines.append('')
        report_lines.append(f"Evaluated videos: {len(df_eval)}")
        report_lines.append(f"Considered (REAL/AI): {len(considered)}")
        report_lines.append(f"Accuracy (REAL/AI only): {accuracy*100:.1f}%")
        report_lines.append('')
        report_lines.append('Confusion (REAL/AI only):')
        report_lines.append(f"  TP (AI→AI): {tp}")
        report_lines.append(f"  TN (REAL→REAL): {tn}")
        report_lines.append(f"  FP (REAL→AI): {fp}")
        report_lines.append(f"  FN (AI→REAL): {fn}")
        report_lines.append('')
        report_lines.append('Bitrate buckets (REAL/AI only):')

        for b, g in considered.groupby('bitrate_bucket'):
            g_tp = int(((g['human_label'] == 'AI') & (g['pred_label'] == 'AI')).sum())
            g_tn = int(((g['human_label'] == 'REAL') & (g['pred_label'] == 'REAL')).sum())
            g_fp = int(((g['human_label'] == 'REAL') & (g['pred_label'] == 'AI')).sum())
            g_fn = int(((g['human_label'] == 'AI') & (g['pred_label'] == 'REAL')).sum())
            report_lines.append(f"  {b}: n={len(g)}, FP={g_fp}, FN={g_fn}, TP={g_tp}, TN={g_tn}")

        report_lines.append('')
        report_text = '\n'.join(report_lines)

        report_dir = LAYER1_DATA_DIR / 'report'
        report_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        docx_path = report_dir / f'layer1_ai_eval_{ts}.docx'
        pdf_path = report_dir / f'layer1_ai_eval_{ts}.pdf'

        try:
            from docx import Document
            doc = Document()
            for line in report_text.split('\n'):
                doc.add_paragraph(line)
            doc.save(str(docx_path))
        except Exception as e:
            logger.warning(f"⚠️  Word報告生成失敗: {e}")
            docx_path = None

        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet

            styles = getSampleStyleSheet()
            story = []
            for line in report_text.split('\n'):
                story.append(Paragraph(line.replace('<', '&lt;').replace('>', '&gt;'), styles['Normal']))
                story.append(Spacer(1, 6))
            doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
            doc.build(story)
        except Exception as e:
            logger.warning(f"⚠️  PDF報告生成失敗: {e}")
            pdf_path = None

        return {
            'download': download_stats,
            'evaluated': len(df_eval),
            'eval_excel': str(eval_xlsx),
            'docx': str(docx_path) if docx_path else '',
            'pdf': str(pdf_path) if pdf_path else ''
        }


def main():
    """主程式"""
    parser = argparse.ArgumentParser(description="Layer 1 人工主導標註流水線")

    parser.add_argument(
        '--check-paths',
        action='store_true',
        help='檢查路徑配置'
    )

    parser.add_argument(
        '--download-detect-report',
        action='store_true',
        help='下載 + AI檢測 + 對比報告（Layer 1）'
    )

    parser.add_argument(
        '--download-only',
        action='store_true',
        help='只下載並分類（不做檢測/報告）'
    )

    parser.add_argument(
        '--redo-download',
        action='store_true',
        help='重做下載（備份現有影片後重新下載）'
    )

    parser.add_argument(
        '--redo-backup-only',
        action='store_true',
        help='只做重做下載的備份（不下載）'
    )

    args = parser.parse_args()

    # 創建流水線
    pipeline = Layer1Pipeline()

    if args.check_paths:
        print(f"\n{'='*80}")
        print("路徑配置:")
        print(f"{'='*80}")
        print(f"基礎目錄: {LAYER1_BASE_DIR}")
        print(f"數據目錄: {LAYER1_DATA_DIR}")
        print(f"\nExcel 文件:")
        print(f"  • Excel A: {EXCEL_A_PATH}")
        print(f"  • Excel B: {EXCEL_B_PATH}")
        print(f"  • Excel C: {EXCEL_C_PATH}")
        print(f"\n視頻文件夾:")
        from config import LAYER1_VIDEO_FOLDERS
        for label, folder in LAYER1_VIDEO_FOLDERS.items():
            print(f"  • {label}: {folder}")
        print(f"{'='*80}\n")
        return

    if args.download_detect_report:
        stats = pipeline.run_download_detect_report()
        if stats and not stats.get('error'):
            print("\n✅ Layer 1 下載+檢測+報告完成！")
        elif stats and stats.get('error'):
            print(f"\n❌ Layer 1 下載+檢測+報告失敗: {stats.get('error')}")
        return

    if args.download_only:
        stats = pipeline.run_download_only()
        if stats:
            print("\n✅ Layer 1 下載完成！")
        return

    if args.redo_backup_only:
        stats = pipeline.run_redo_download(download=False)
        if stats:
            print("\n✅ Layer 1 備份完成！")
        return
    
    if args.redo_download:
        stats = pipeline.run_redo_download()
        if stats:
            print("\n✅ Layer 1 重做下載完成！")
        return

    # 執行完整流水線
    stats = pipeline.run_full_pipeline()

    if stats:
        print(f"\n✅ Layer 1 流水線執行完成！")


if __name__ == "__main__":
    main()
