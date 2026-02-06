#!/usr/bin/env python3
"""動画を FastSAM で監視し、時系列ログと新規物体画像を出力する。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import FastSAM
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "ultralytics が必要です。`pip install ultralytics opencv-python numpy` を実行してください。"
    ) from exc


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray  # x1,y1,x2,y2
    area_ratio: float
    created_frame: int
    last_frame: int
    label: str = field(default="unknown")
    image_file: Optional[str] = field(default=None)


class VideoMonitor:
    def __init__(
        self,
        model_path: str,
        conf: float,
        iou: float,
        detect_interval: int,
        diff_threshold: int,
        min_area_ratio: float,
        max_area_ratio: float,
        max_missing_frames: int,
    ) -> None:
        self.model = FastSAM(model_path)
        self.conf = conf
        self.iou = iou
        self.detect_interval = max(1, detect_interval)
        self.diff_threshold = diff_threshold
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.max_missing_frames = max_missing_frames

        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1

    @staticmethod
    def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0.0
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    def _changed_rois(self, prev_gray: np.ndarray, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        diff = cv2.absdiff(prev_gray, gray)
        _, th = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        th = cv2.dilate(th, np.ones((7, 7), np.uint8), iterations=2)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois: List[Tuple[int, int, int, int]] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < 64:
                continue
            rois.append((x, y, x + w, y + h))
        return rois

    def _flow_update_tracks(self, prev_gray: np.ndarray, gray: np.ndarray) -> None:
        if not self.tracks:
            return
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        h, w = gray.shape[:2]
        for tr in self.tracks.values():
            x1, y1, x2, y2 = tr.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            local = flow[y1:y2, x1:x2]
            if local.size == 0:
                continue
            dx = float(np.median(local[..., 0]))
            dy = float(np.median(local[..., 1]))
            tr.bbox = np.array(
                [
                    np.clip(x1 + dx, 0, w - 1),
                    np.clip(y1 + dy, 0, h - 1),
                    np.clip(x2 + dx, 0, w - 1),
                    np.clip(y2 + dy, 0, h - 1),
                ],
                dtype=np.float32,
            )

    def _segment_roi(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> List[np.ndarray]:
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        res = self.model.predict(crop, conf=self.conf, iou=self.iou, retina_masks=True, verbose=False)
        if not res:
            return []
        boxes = getattr(res[0], "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []
        out = []
        for b in boxes.xyxy.cpu().numpy():
            bx1, by1, bx2, by2 = b[:4]
            out.append(np.array([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1], dtype=np.float32))
        return out

    def monitor(self, video_path: Path, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        img_dir = out_dir / "new_objects"
        img_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "monitor_log.jsonl"

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"動画が開けません: {video_path}")

        prev_gray = None
        frame_idx = 0

        with log_path.open("w", encoding="utf-8") as fw:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape[:2]
                frame_area = float(h * w)

                if prev_gray is not None:
                    self._flow_update_tracks(prev_gray, gray)

                need_detect = frame_idx % self.detect_interval == 0
                detections: List[np.ndarray] = []
                if need_detect:
                    if prev_gray is None:
                        rois = [(0, 0, w, h)]
                    else:
                        rois = self._changed_rois(prev_gray, gray)
                        if not rois:
                            rois = []
                    for roi in rois:
                        detections.extend(self._segment_roi(frame, roi))

                    # size filter: 1/100 <= area <= 1/2
                    filtered = []
                    for b in detections:
                        area = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
                        ratio = area / frame_area
                        if self.min_area_ratio <= ratio <= self.max_area_ratio:
                            filtered.append((b, ratio))

                    matched_track_ids = set()
                    for b, ratio in filtered:
                        best_id = None
                        best_iou = 0.0
                        for tid, tr in self.tracks.items():
                            iou = self._bbox_iou(b, tr.bbox)
                            if iou > best_iou:
                                best_iou = iou
                                best_id = tid

                        if best_id is not None and best_iou >= 0.3:
                            tr = self.tracks[best_id]
                            tr.bbox = b
                            tr.area_ratio = ratio
                            tr.last_frame = frame_idx
                            matched_track_ids.add(best_id)
                            record = {
                                "frame": frame_idx,
                                "event": "update",
                                "track_id": best_id,
                                "bbox": [float(x) for x in b],
                                "area_ratio": ratio,
                                "image_file": tr.image_file,
                            }
                            fw.write(json.dumps(record, ensure_ascii=False) + "\n")
                        else:
                            tid = self.next_track_id
                            self.next_track_id += 1
                            label = f"object_{tid}"
                            tr = Track(tid, b, ratio, frame_idx, frame_idx, label=label)

                            x1, y1, x2, y2 = b.astype(int)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w - 1, x2), min(h - 1, y2)
                            obj = frame[y1:y2, x1:x2].copy()
                            img_name = f"{label}_f{frame_idx:06d}.jpg"
                            img_path = img_dir / img_name
                            if obj.size > 0:
                                cv2.imwrite(str(img_path), obj)
                                tr.image_file = str(img_path.relative_to(out_dir))

                            self.tracks[tid] = tr
                            matched_track_ids.add(tid)
                            record = {
                                "frame": frame_idx,
                                "event": "new",
                                "track_id": tid,
                                "label": label,
                                "bbox": [float(x) for x in b],
                                "area_ratio": ratio,
                                "image_file": tr.image_file,
                            }
                            fw.write(json.dumps(record, ensure_ascii=False) + "\n")

                    # lost track logging
                    to_remove = []
                    for tid, tr in self.tracks.items():
                        if frame_idx - tr.last_frame > self.max_missing_frames:
                            fw.write(
                                json.dumps(
                                    {
                                        "frame": frame_idx,
                                        "event": "lost",
                                        "track_id": tid,
                                        "label": tr.label,
                                        "last_bbox": [float(x) for x in tr.bbox],
                                        "image_file": tr.image_file,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            to_remove.append(tid)
                    for tid in to_remove:
                        self.tracks.pop(tid, None)

                prev_gray = gray
                frame_idx += 1

        cap.release()
        return log_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FastSAM による動画モニタリング")
    p.add_argument("video", type=Path, help="入力動画")
    p.add_argument("--output", type=Path, default=Path("monitor_output"), help="出力ディレクトリ")
    p.add_argument("--model", default="FastSAM-s.pt", help="FastSAM の重み")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--iou", type=float, default=0.9)
    p.add_argument("--detect-interval", type=int, default=3, help="セグメンテーション間隔")
    p.add_argument("--diff-threshold", type=int, default=18, help="差分2値化しきい値")
    p.add_argument("--max-missing", type=int, default=20, help="追跡ロスト判定フレーム数")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    monitor = VideoMonitor(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        detect_interval=args.detect_interval,
        diff_threshold=args.diff_threshold,
        min_area_ratio=1 / 100,
        max_area_ratio=1 / 2,
        max_missing_frames=args.max_missing,
    )
    log_path = monitor.monitor(args.video, args.output)
    print(f"monitor log: {log_path}")


if __name__ == "__main__":
    main()
