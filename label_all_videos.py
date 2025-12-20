#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Label ALL downloaded videos (not just "not sure" folder)
本地標註所有下載的視頻
"""
import sys
from tiktok_tinder_labeler import TikTokTinderLabeler

# Label all videos from the download folder
video_folder = r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok videos download\tiktok videos download"
output_excel = r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok videos download\data\human_labels_all.xlsx"

print("Labeling ALL downloaded videos...")
labeler = TikTokTinderLabeler(video_folder, output_excel)
labeler.run()
