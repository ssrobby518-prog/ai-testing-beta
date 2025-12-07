#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRIMARY_TIER: Video Preprocessor - 一次性视频数据提取
FR-TSAR: Maximum energy transmission - 只读取一次视频，生成压缩的不可变事件流
FR-RAPTOR: Zero abstraction cost - 直接使用OpenCV，无ORM包装
FR-SPARK-PLUG: Pure I/O module - 不做复杂计算，只做数据提取和压缩
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pymediainfo import MediaInfo

logging.basicConfig(level=logging.INFO)


@dataclass
class VideoMetadata:
    """不可变视频元数据（PRIMARY_TIER输出）"""
    file_path: str
    bitrate: int
    fps: float
    total_frames: int
    width: int
    height: int
    duration: float


@dataclass
class PreprocessedFrame:
    """预处理帧数据（压缩格式）"""
    index: int
    gray: np.ndarray  # 灰度图
    color: np.ndarray  # BGR彩色图
    hsv: np.ndarray    # HSV色彩空间
    faces: List[Tuple[int, int, int, int]]  # 检测到的人脸 [(x, y, w, h), ...]


@dataclass
class VideoFeatures:
    """
    PRIMARY_TIER 输出：压缩的视频特征
    FR-TSAR: 这是"压缩数据"，为 SECONDARY_TIER 提供最大能量
    """
    metadata: VideoMetadata
    frames: List[PreprocessedFrame]  # 采样的帧
    face_presence: float  # [0,1] 有人脸的帧比例
    static_ratio: float   # [0,1] 静态帧比例


class VideoPreprocessor:
    """
    PRIMARY_TIER Service: 视频预处理服务

    FR-TSAR: 一次性读取视频，生成压缩数据，为后续模块提供"核燃料"
    FR-RAPTOR: 单一职责 - 只做视频I/O和基础特征提取，不做AI检测
    """

    def __init__(self, max_frames: int = 100):
        """
        Args:
            max_frames: 最大采样帧数（性能优化）
        """
        self.max_frames = max_frames
        # FR-RAPTOR: 提前加载分类器，避免重复加载
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def _primary_extract_metadata(self, file_path: str) -> VideoMetadata:
        """
        FR-TSAR: PRIMARY阶段 - 提取视频元数据
        FR-SPARK-PLUG: 纯函数逻辑，无副作用
        """
        media_info = MediaInfo.parse(file_path)
        bitrate = 0
        fps = 30.0
        width = 0
        height = 0
        duration = 0.0

        for track in media_info.tracks:
            if track.track_type == 'Video':
                bitrate = track.bit_rate if track.bit_rate else 0
                fps = float(track.frame_rate) if track.frame_rate else 30.0
                width = track.width if track.width else 0
                height = track.height if track.height else 0
                duration = float(track.duration) / 1000.0 if track.duration else 0.0
                break

        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        return VideoMetadata(
            file_path=file_path,
            bitrate=bitrate,
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            duration=duration
        )

    def _primary_decode_frames(
        self,
        file_path: str,
        total_frames: int
    ) -> List[PreprocessedFrame]:
        """
        FR-TSAR: PRIMARY阶段 - 解码并预处理帧
        FR-RAPTOR: 一次性读取，避免重复I/O
        """
        cap = cv2.VideoCapture(file_path)
        frames = []

        # FR-RAPTOR: 智能采样 - 根据视频长度调整采样率
        sample_frames = min(self.max_frames, total_frames)

        for i in range(sample_frames):
            if total_frames > 0:
                frame_pos = int(i * total_frames / sample_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

            ret, frame = cap.read()
            if not ret:
                break

            # FR-RAPTOR: 一次性转换所有需要的色彩空间，避免后续重复转换
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # FR-TSAR: 提前检测人脸，压缩数据传递
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)
            faces_list = [tuple(map(int, face)) for face in faces]

            frames.append(PreprocessedFrame(
                index=i,
                gray=gray,
                color=frame,
                hsv=hsv,
                faces=faces_list
            ))

        cap.release()
        return frames

    def _primary_calculate_face_presence(
        self,
        frames: List[PreprocessedFrame]
    ) -> float:
        """
        FR-SPARK-PLUG: 纯计算函数 - 计算人脸存在比例
        """
        if not frames:
            return 0.0

        face_count = sum(1 for f in frames if len(f.faces) > 0)
        return face_count / len(frames)

    def _primary_calculate_static_ratio(
        self,
        frames: List[PreprocessedFrame]
    ) -> float:
        """
        FR-SPARK-PLUG: 纯计算函数 - 计算静态帧比例
        FR-RAPTOR: 使用向量化操作提升性能
        """
        if len(frames) < 2:
            return 0.0

        diffs = []
        for i in range(1, min(40, len(frames))):
            prev = frames[i-1].gray
            curr = frames[i].gray

            # FR-RAPTOR: 优化 - 缩小尺寸加速计算
            h, w = prev.shape
            scale = 160.0 / max(w, 1)
            prev_resized = cv2.resize(prev, (int(w*scale), int(h*scale)))
            curr_resized = cv2.resize(curr, (int(w*scale), int(h*scale)))

            diff = cv2.absdiff(curr_resized, prev_resized)
            diffs.append(float(diff.mean()))

        if not diffs:
            return 0.0

        # FR-SPARK-PLUG: 向量化操作
        static_count = sum(1.0 for d in diffs if d < 1.5)
        return static_count / len(diffs)

    def preprocess(self, file_path: str) -> VideoFeatures:
        """
        主入口：预处理视频

        FR-TSAR: 一次性提取所有数据，生成压缩的不可变事件流
        FR-RAPTOR: 明确的数据流方向 - PRIMARY -> SECONDARY

        Returns:
            VideoFeatures: 压缩的视频特征，供 SECONDARY_TIER 使用
        """
        logging.info(f"[PRIMARY_TIER] Preprocessing video: {file_path}")

        # 阶段1：提取元数据
        metadata = self._primary_extract_metadata(file_path)
        logging.info(f"[PRIMARY_TIER] Metadata extracted: bitrate={metadata.bitrate}, fps={metadata.fps}")

        # 阶段2：解码并预处理帧
        frames = self._primary_decode_frames(file_path, metadata.total_frames)
        logging.info(f"[PRIMARY_TIER] Decoded {len(frames)} frames")

        # 阶段3：计算全局特征
        face_presence = self._primary_calculate_face_presence(frames)
        static_ratio = self._primary_calculate_static_ratio(frames)
        logging.info(f"[PRIMARY_TIER] Global features: face={face_presence:.2f}, static={static_ratio:.2f}")

        return VideoFeatures(
            metadata=metadata,
            frames=frames,
            face_presence=face_presence,
            static_ratio=static_ratio
        )
