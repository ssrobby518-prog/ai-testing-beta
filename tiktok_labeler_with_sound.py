#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TikTok Labeler with Sound - 使用系統播放器播放有聲音的視頻
"""
import cv2
import pandas as pd
from pathlib import Path
import subprocess
import time
import shutil
import os

class TikTokLabelerWithSound:
    def __init__(self, video_folder, excel_output):
        self.video_folder = Path(video_folder)
        self.excel_output = Path(excel_output)
        self.videos = list(self.video_folder.glob("*.mp4"))
        self.current_index = 0
        self.labels = []
        self.player_process = None

        # Classification folders
        base_dir = Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok videos download")
        self.classification_folders = {
            'REAL': base_dir / 'real',
            'AI': base_dir / 'ai',
            'NOT_SURE': base_dir / 'not sure',
            'MOVIE': base_dir / '電影動畫'
        }

        # Ensure classification folders exist
        for folder in self.classification_folders.values():
            folder.mkdir(parents=True, exist_ok=True)

        # Load existing labels
        if self.excel_output.exists():
            df = pd.read_excel(self.excel_output)
            self.labels = df.to_dict('records')
            labeled_videos = set(str(r['Video_ID']) for r in self.labels)
            self.videos = [v for v in self.videos if v.stem not in labeled_videos]

        print(f"\n{'='*70}")
        print("TikTok Labeler with Sound - 有聲音標註工具")
        print(f"{'='*70}")
        print(f"Total videos to label: {len(self.videos)}")
        print(f"\n控制說明:")
        print("  影片會用系統播放器打開（有聲音）")
        print("  看完後在控制視窗按鍵：")
        print("    ← LEFT  = REAL (真實)")
        print("    → RIGHT = AI (AI生成)")
        print("    ↑ UP    = NOT SURE (不確定)")
        print("    ↓ DOWN  = MOVIE (電影/動畫)")
        print("    r       = Replay (重播)")
        print("    ESC/q   = Quit (退出)")
        print(f"{'='*70}\n")

    def play_video_with_sound(self, video_path):
        """Play video with system player"""
        # Kill any existing player
        if self.player_process:
            try:
                self.player_process.kill()
            except:
                pass

        # Open video with default player (has sound)
        self.player_process = subprocess.Popen(['start', '', str(video_path)], shell=True)

        # Create control window
        control_img = self._create_control_window(video_path.name)
        window_name = "Labeling Controls - Press Arrow Keys"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 600, 400)

        while True:
            cv2.imshow(window_name, control_img)
            # Check if window was closed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                return 'EXIT'

            key = cv2.waitKey(100)

            if key == -1:
                continue

            print(f"[DEBUG] Key: {key}")

            # ESC or q = EXIT
            if key == 27 or key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                return 'EXIT'

            # r = replay
            elif key == ord('r') or key == ord('R'):
                if self.player_process:
                    try:
                        self.player_process.kill()
                    except:
                        pass
                self.player_process = subprocess.Popen(['start', '', str(video_path)], shell=True)
                continue

            # WASD or Arrow keys
            elif key == ord('a') or key == ord('A'):  # 'a' = REAL
                cv2.destroyAllWindows()
                return 'REAL'
            elif key == ord('d') or key == ord('D'):  # 'd' = AI
                cv2.destroyAllWindows()
                return 'AI'
            elif key == ord('w') or key == ord('W'):  # 'w' = NOT_SURE
                cv2.destroyAllWindows()
                return 'NOT_SURE'
            elif key == ord('s') or key == ord('S'):  # 's' = MOVIE
                cv2.destroyAllWindows()
                return 'MOVIE'

    def _create_control_window(self, filename):
        """Create control instruction window"""
        import numpy as np
        img = np.zeros((400, 600, 3), dtype=np.uint8)

        # Title
        cv2.putText(img, "TikTok Labeler - Controls", (50, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Current video
        cv2.putText(img, f"Video: {filename[:40]}", (50, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(img, f"[{self.current_index + 1}/{len(self.videos)}]", (50, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Instructions
        y = 160
        cv2.putText(img, "Press:", (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        y += 40
        cv2.putText(img, "<- or 'a'  = REAL", (80, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        y += 35
        cv2.putText(img, "-> or 'd'  = AI", (80, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        y += 35
        cv2.putText(img, "^  or 'w'  = NOT SURE", (80, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        y += 35
        cv2.putText(img, "v  or 's'  = MOVIE", (80, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)

        y += 50
        cv2.putText(img, "'r'        = Replay", (80, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        y += 35
        cv2.putText(img, "ESC or 'q' = Quit", (80, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return img

    def save_labels(self):
        """Save labels to Excel"""
        if not self.labels:
            return

        df = pd.DataFrame(self.labels)
        self.excel_output.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(self.excel_output, index=False)

        print(f"\nLabels saved: {len(self.labels)} videos")

        real = sum(1 for l in self.labels if l['Label'] == 'REAL')
        ai = sum(1 for l in self.labels if l['Label'] == 'AI')
        not_sure = sum(1 for l in self.labels if l['Label'] == 'NOT_SURE')
        movie = sum(1 for l in self.labels if l['Label'] == 'MOVIE')

        print(f"  REAL: {real} | AI: {ai} | NOT_SURE: {not_sure} | MOVIE: {movie}")

    def run(self):
        """Main labeling loop"""
        if not self.videos:
            print("No videos to label!")
            return

        try:
            while self.current_index < len(self.videos):
                video = self.videos[self.current_index]
                print(f"\n[{self.current_index + 1}/{len(self.videos)}] {video.name}")

                label = self.play_video_with_sound(video)

                if label == 'EXIT':
                    print("\nExiting...")
                    break

                # Record label
                self.labels.append({
                    'Video_ID': video.stem,
                    'Filename': video.name,
                    'Label': label,
                    'Timestamp': pd.Timestamp.now()
                })

                print(f"  -> Labeled as: {label}")

                # Classify to folder
                dest_folder = self.classification_folders.get(label)
                if dest_folder:
                    dest_path = dest_folder / video.name
                    try:
                        shutil.copy2(video, dest_path)
                        print(f"  -> Copied to: {dest_folder.name}/")
                    except Exception as e:
                        print(f"  [ERROR] Copy failed: {e}")

                # Auto-save every 5 labels
                if len(self.labels) % 5 == 0:
                    self.save_labels()

                self.current_index += 1

        finally:
            # Cleanup
            if self.player_process:
                try:
                    self.player_process.kill()
                except:
                    pass

            cv2.destroyAllWindows()
            self.save_labels()

        print("\nDone!")

if __name__ == "__main__":
    video_folder = r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok videos download\tiktok videos download"
    output_excel = r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok videos download\data\human_labels_all.xlsx"

    labeler = TikTokLabelerWithSound(video_folder, output_excel)
    labeler.run()
