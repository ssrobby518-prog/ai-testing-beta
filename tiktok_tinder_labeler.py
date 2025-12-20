#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Local TikTok Tinder-style Video Labeler
本地TikTok視頻標註工具 - Tinder式上下左右滑動

Controls:
    ← LEFT:  REAL (真實視頻)
    → RIGHT: AI (AI生成)
    ↑ UP:    NOT SURE (不確定)
    ↓ DOWN:  MOVIE (電影/動畫)
    SPACE:   Replay current video
    ESC:     Exit

Design: 第一性原理 - 極簡高效的人工標註界面
"""
import cv2
import pandas as pd
from pathlib import Path
import time
import os
import shutil

class TikTokTinderLabeler:
    def __init__(self, video_folder, excel_output):
        self.video_folder = Path(video_folder)
        self.excel_output = Path(excel_output)
        self.videos = list(self.video_folder.glob("*.mp4"))
        self.current_index = 0
        self.labels = []

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

        # Load existing labels if available
        if self.excel_output.exists():
            df = pd.read_excel(self.excel_output)
            self.labels = df.to_dict('records')
            labeled_videos = set(r['Video_ID'] for r in self.labels)
            self.videos = [v for v in self.videos if v.stem not in labeled_videos]

        print(f"\n{'='*70}")
        print("TikTok Tinder Labeler - Local Video Annotation Tool")
        print(f"{'='*70}")
        print(f"Total videos to label: {len(self.videos)}")
        print(f"\nControls:")
        print("  A or ←  = REAL (真實)")
        print("  D or →  = AI (AI生成)")
        print("  W or ↑  = NOT SURE (不確定)")
        print("  S or ↓  = MOVIE (電影/動畫)")
        print("  SPACE   = Replay")
        print("  ESC/X   = Exit & Save")
        print(f"{'='*70}\n")

    def play_video(self, video_path):
        """Play video in a window"""
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Error: Cannot open {video_path.name}")
            return None

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30
        delay = int(1000 / fps)

        window_name = f"TikTok Labeler - {video_path.name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 405, 720)  # TikTok aspect ratio

        while True:
            ret, frame = cap.read()

            if not ret:
                # Video finished, loop it
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Add overlay text (no sticker during playback)
            self._add_overlay(frame, video_path.name, self.current_index + 1, len(self.videos), label_sticker=None)

            cv2.imshow(window_name, frame)

            # Check if window was closed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                cap.release()
                cv2.destroyAllWindows()
                return 'EXIT'

            # Use waitKeyEx for proper arrow key detection on Windows
            key = cv2.waitKeyEx(delay)

            # ESC or window close
            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                return 'EXIT'

            # Use letters for easier control (arrow keys are problematic on Windows)
            # a=left/REAL, d=right/AI, w=up/NOT_SURE, s=down/MOVIE
            if key == ord('a'):  # 'a' = LEFT = REAL
                label = 'REAL'
                self._show_confirmation(cap, video_path.name, label)
                cap.release()
                cv2.destroyAllWindows()
                return label
            elif key == ord('d'):  # 'd' = RIGHT = AI
                label = 'AI'
                self._show_confirmation(cap, video_path.name, label)
                cap.release()
                cv2.destroyAllWindows()
                return label
            elif key == ord('w'):  # 'w' = UP = NOT_SURE
                label = 'NOT_SURE'
                self._show_confirmation(cap, video_path.name, label)
                cap.release()
                cv2.destroyAllWindows()
                return label
            elif key == ord('s'):  # 's' = DOWN = MOVIE
                label = 'MOVIE'
                self._show_confirmation(cap, video_path.name, label)
                cap.release()
                cv2.destroyAllWindows()
                return label

            # Also support arrow keys (extended codes for Windows)
            elif key == 2424832:  # LEFT arrow
                label = 'REAL'
                self._show_confirmation(cap, video_path.name, label)
                cap.release()
                cv2.destroyAllWindows()
                return label
            elif key == 2555904:  # RIGHT arrow
                label = 'AI'
                self._show_confirmation(cap, video_path.name, label)
                cap.release()
                cv2.destroyAllWindows()
                return label
            elif key == 2490368:  # UP arrow
                label = 'NOT_SURE'
                self._show_confirmation(cap, video_path.name, label)
                cap.release()
                cv2.destroyAllWindows()
                return label
            elif key == 2621440:  # DOWN arrow
                label = 'MOVIE'
                self._show_confirmation(cap, video_path.name, label)
                cap.release()
                cv2.destroyAllWindows()
                return label

            elif key == 32:  # SPACE - replay
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _show_confirmation(self, cap, filename, label):
        """Show confirmation sticker for 1 second"""
        window_name = f"TikTok Labeler - {filename}"

        # Show 3 frames with the sticker
        for _ in range(30):  # ~1 second at 30fps
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

            if ret:
                self._add_overlay(frame, filename, self.current_index + 1, len(self.videos), label_sticker=label)
                cv2.imshow(window_name, frame)
                cv2.waitKey(33)  # ~30fps

    def _add_overlay(self, frame, filename, current, total, label_sticker=None):
        """Add info overlay to frame"""
        h, w = frame.shape[:2]

        # Semi-transparent overlay at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Text
        cv2.putText(frame, f"Video: {current}/{total}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, filename[:40], (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Controls reminder at bottom
        cv2.rectangle(overlay, (0, h-100), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, "A/< REAL  |  W/^ NOT SURE  |  S/v MOVIE  |  D/> AI", (10, h-50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, "SPACE=Replay  ESC/X=Exit", (10, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Label sticker in bottom-left corner
        if label_sticker:
            sticker_colors = {
                'REAL': (0, 255, 0),      # Green
                'AI': (0, 0, 255),        # Red
                'NOT_SURE': (0, 255, 255), # Yellow
                'MOVIE': (255, 0, 255)    # Magenta
            }
            color = sticker_colors.get(label_sticker, (255, 255, 255))

            # Draw sticker background
            sticker_overlay = frame.copy()
            cv2.rectangle(sticker_overlay, (10, h-150), (150, h-110), color, -1)
            cv2.addWeighted(sticker_overlay, 0.8, frame, 0.2, 0, frame)

            # Draw sticker text
            label_text = label_sticker.replace('_', ' ')
            cv2.putText(frame, label_text, (20, h-125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    def save_labels(self):
        """Save labels to Excel"""
        if not self.labels:
            print("No labels to save")
            return

        df = pd.DataFrame(self.labels)
        self.excel_output.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(self.excel_output, index=False)

        print(f"\n{'='*70}")
        print(f"Labels saved to: {self.excel_output}")
        print(f"Total labeled: {len(self.labels)}")

        # Summary
        real = sum(1 for l in self.labels if l['Label'] == 'REAL')
        ai = sum(1 for l in self.labels if l['Label'] == 'AI')
        not_sure = sum(1 for l in self.labels if l['Label'] == 'NOT_SURE')
        movie = sum(1 for l in self.labels if l['Label'] == 'MOVIE')

        print(f"\nSummary:")
        print(f"  REAL:     {real}")
        print(f"  AI:       {ai}")
        print(f"  NOT SURE: {not_sure}")
        print(f"  MOVIE:    {movie}")
        print(f"{'='*70}\n")

    def run(self):
        """Main labeling loop"""
        if not self.videos:
            print("No videos to label!")
            return

        while self.current_index < len(self.videos):
            video = self.videos[self.current_index]

            print(f"\n[{self.current_index + 1}/{len(self.videos)}] Playing: {video.name}")

            label = self.play_video(video)

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

            print(f"  Labeled as: {label}")

            # Classify video to appropriate folder
            dest_folder = self.classification_folders.get(label)
            if dest_folder:
                dest_path = dest_folder / video.name
                try:
                    shutil.copy2(video, dest_path)
                    print(f"  Copied to: {dest_folder.name}/")
                except Exception as e:
                    print(f"  [ERROR] Failed to copy: {e}")

            # Auto-save every 5 labels
            if len(self.labels) % 5 == 0:
                self.save_labels()

            self.current_index += 1

        # Final save
        self.save_labels()
        print("All videos labeled!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TikTok Tinder-style Video Labeler")
    parser.add_argument('--folder', type=str,
                       default=r'C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok videos download\not sure',
                       help='Folder containing videos to label')
    parser.add_argument('--output', type=str,
                       default=r'C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok videos download\data\human_labels.xlsx',
                       help='Output Excel file for labels')

    args = parser.parse_args()

    labeler = TikTokTinderLabeler(args.folder, args.output)
    labeler.run()


if __name__ == "__main__":
    main()
