#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
執行第一層下載 - 快捷方式
"""
import subprocess
import sys
from pathlib import Path

# 切換到腳本所在目錄
script_dir = Path(__file__).parent
pipeline_script = script_dir / "tiktok_labeler" / "pipeline" / "layer1_pipeline.py"

print("=" * 80)
print("執行第一層下載")
print("=" * 80)
print()

# 執行 pipeline
subprocess.run([sys.executable, str(pipeline_script)])
