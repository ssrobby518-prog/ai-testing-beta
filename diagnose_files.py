#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pymediainfo import MediaInfo
import sys

def diagnose(file_path):
    media_info = MediaInfo.parse(file_path)
    for track in media_info.tracks:
        if track.track_type == 'General':
            print(f"Format: {track.format}")
            print(f"File size: {track.file_size}")
            print(f"Duration: {track.duration}")
            print(f"Overall bit rate: {track.overall_bit_rate}")
        if track.track_type == 'Video':
            print(f"Video codec: {track.codec_id}")
            print(f"Bit rate: {track.bit_rate}")
            print(f"Frame rate: {track.frame_rate}")
            print(f"Width: {track.width}")
            print(f"Height: {track.height}")
            print(f"Compression mode: {track.compression_mode}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_files.py <file_path>")
    else:
        diagnose(sys.argv[1])