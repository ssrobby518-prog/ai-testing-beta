#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
圖形界面標註工具 - 最簡單版本
"""
import tkinter as tk
from tkinter import messagebox
import subprocess
import pandas as pd
from pathlib import Path
import shutil

class VideoLabeler:
    def __init__(self):
        self.video_folder = Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok videos download\tiktok videos download")
        self.excel_output = Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok videos download\data\human_labels_all.xlsx")

        base_dir = Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok videos download")
        self.folders = {
            'REAL': base_dir / 'real',
            'AI': base_dir / 'ai',
            'NOT_SURE': base_dir / 'not sure',
            'MOVIE': base_dir / '電影動畫'
        }

        # Create folders
        for f in self.folders.values():
            f.mkdir(parents=True, exist_ok=True)

        # Get videos
        self.videos = list(self.video_folder.glob("*.mp4"))
        self.labels = []
        self.current_index = 0
        self.player_process = None

        # Load existing
        if self.excel_output.exists():
            df = pd.read_excel(self.excel_output)
            self.labels = df.to_dict('records')
            labeled = set(str(r['Video_ID']) for r in self.labels)
            self.videos = [v for v in self.videos if v.stem not in labeled]

        # Create GUI
        self.root = tk.Tk()
        self.root.title("TikTok 標註工具")
        self.root.geometry("500x400")
        self.root.configure(bg='#2b2b2b')

        # Title
        self.title_label = tk.Label(
            self.root,
            text="TikTok 影片標註",
            font=("Arial", 20, "bold"),
            bg='#2b2b2b',
            fg='white'
        )
        self.title_label.pack(pady=20)

        # Video info
        self.info_label = tk.Label(
            self.root,
            text=f"總共 {len(self.videos)} 個影片",
            font=("Arial", 12),
            bg='#2b2b2b',
            fg='#aaa'
        )
        self.info_label.pack(pady=10)

        self.current_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 10),
            bg='#2b2b2b',
            fg='#888'
        )
        self.current_label.pack(pady=5)

        # Buttons frame
        button_frame = tk.Frame(self.root, bg='#2b2b2b')
        button_frame.pack(pady=30)

        # Buttons
        self.btn_real = tk.Button(
            button_frame,
            text="← REAL\n(真實)",
            font=("Arial", 14, "bold"),
            bg='#4CAF50',
            fg='white',
            width=10,
            height=3,
            command=lambda: self.label_video('REAL')
        )
        self.btn_real.grid(row=0, column=0, padx=10)

        self.btn_ai = tk.Button(
            button_frame,
            text="AI →\n(AI生成)",
            font=("Arial", 14, "bold"),
            bg='#f44336',
            fg='white',
            width=10,
            height=3,
            command=lambda: self.label_video('AI')
        )
        self.btn_ai.grid(row=0, column=1, padx=10)

        self.btn_not_sure = tk.Button(
            button_frame,
            text="↑ NOT SURE\n(不確定)",
            font=("Arial", 14, "bold"),
            bg='#FFC107',
            fg='black',
            width=10,
            height=3,
            command=lambda: self.label_video('NOT_SURE')
        )
        self.btn_not_sure.grid(row=1, column=0, padx=10, pady=10)

        self.btn_movie = tk.Button(
            button_frame,
            text="↓ MOVIE\n(電影/動畫)",
            font=("Arial", 14, "bold"),
            bg='#9C27B0',
            fg='white',
            width=10,
            height=3,
            command=lambda: self.label_video('MOVIE')
        )
        self.btn_movie.grid(row=1, column=1, padx=10, pady=10)

        # Keyboard shortcuts
        self.root.bind('a', lambda e: self.label_video('REAL'))
        self.root.bind('d', lambda e: self.label_video('AI'))
        self.root.bind('w', lambda e: self.label_video('NOT_SURE'))
        self.root.bind('s', lambda e: self.label_video('MOVIE'))
        self.root.bind('<Left>', lambda e: self.label_video('REAL'))
        self.root.bind('<Right>', lambda e: self.label_video('AI'))
        self.root.bind('<Up>', lambda e: self.label_video('NOT_SURE'))
        self.root.bind('<Down>', lambda e: self.label_video('MOVIE'))

        # Start first video
        if self.videos:
            self.play_current_video()
        else:
            messagebox.showinfo("完成", "所有影片已標註完畢！")
            self.root.quit()

    def play_current_video(self):
        """Play current video"""
        if self.current_index >= len(self.videos):
            self.save_labels()
            messagebox.showinfo("完成", f"標註完成！共 {len(self.labels)} 個影片")
            self.root.quit()
            return

        video = self.videos[self.current_index]
        self.current_label.config(text=f"[{self.current_index + 1}/{len(self.videos)}] {video.name}")

        # Play video
        if self.player_process:
            try:
                self.player_process.kill()
            except:
                pass

        self.player_process = subprocess.Popen(['start', '', str(video)], shell=True)

    def label_video(self, label):
        """Label current video"""
        if self.current_index >= len(self.videos):
            return

        video = self.videos[self.current_index]

        # Save label
        self.labels.append({
            'Video_ID': video.stem,
            'Filename': video.name,
            'Label': label,
            'Timestamp': pd.Timestamp.now()
        })

        # Copy to folder
        dest = self.folders[label] / video.name
        shutil.copy2(video, dest)

        # Auto-save every 5
        if len(self.labels) % 5 == 0:
            self.save_labels()

        # Next video
        self.current_index += 1
        self.play_current_video()

    def save_labels(self):
        """Save to Excel"""
        if not self.labels:
            return

        df = pd.DataFrame(self.labels)
        self.excel_output.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(self.excel_output, index=False)

    def run(self):
        """Start GUI"""
        self.root.mainloop()

if __name__ == "__main__":
    app = VideoLabeler()
    app.run()
