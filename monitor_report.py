#!/usr/bin/env python3
"""monitor_fastsam.py の出力からサマリー HTML レポートを生成する。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class Event:
    frame: int
    event: str
    bbox: Optional[List[float]]
    label: Optional[str]
    image_file: Optional[str]
    track_id: Optional[int]


@dataclass
class FrameSummary:
    frame: int
    events: List[Event]
    total_area: float
    change_types: List[str]

    @property
    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {"new": 0, "update": 0, "lost": 0}
        for e in self.events:
            out[e.event] = out.get(e.event, 0) + 1
        return out


def load_logs(log_path: Path) -> List[Dict]:
    rows = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def bbox_area(bbox: Optional[List[float]]) -> float:
    if not bbox:
        return 0.0
    x1, y1, x2, y2 = bbox
    return max(0.0, float((x2 - x1) * (y2 - y1)))


def build_frame_summaries(rows: List[Dict]) -> Dict[int, FrameSummary]:
    grouped: Dict[int, List[Event]] = {}
    track_history: Dict[int, List[Tuple[int, List[float]]]] = {}
    for row in rows:
        frame = int(row.get("frame", 0))
        bbox = row.get("bbox") or row.get("last_bbox")
        event = Event(
            frame=frame,
            event=str(row.get("event", "")),
            bbox=bbox,
            label=row.get("label"),
            image_file=row.get("image_file"),
            track_id=row.get("track_id"),
        )
        grouped.setdefault(frame, []).append(event)
        if event.track_id is not None and bbox is not None:
            track_history.setdefault(int(event.track_id), []).append((frame, bbox))

    summaries: Dict[int, FrameSummary] = {}
    for frame, events in grouped.items():
        total_area = sum(bbox_area(e.bbox) for e in events)
        change_types = summarize_change_types(events, track_history)
        summaries[frame] = FrameSummary(
            frame=frame,
            events=events,
            total_area=total_area,
            change_types=change_types,
        )
    return summaries


def summarize_change_types(
    events: List[Event],
    track_history: Dict[int, List[Tuple[int, List[float]]]],
) -> List[str]:
    types: List[str] = []
    has_new = any(e.event == "new" for e in events)
    has_lost = any(e.event == "lost" for e in events)
    if has_new and has_lost:
        types.append("画面の切り替わり")

    moving_tracks = 0
    for e in events:
        if e.event != "update" or e.track_id is None or e.bbox is None:
            continue
        history = track_history.get(int(e.track_id), [])
        prev_bbox = None
        for f, bbox in history:
            if f >= e.frame:
                break
            prev_bbox = bbox
        if prev_bbox is None:
            continue
        if is_moved(prev_bbox, e.bbox):
            moving_tracks += 1
    if moving_tracks > 0:
        types.append("オブジェクト移動")
    if not types:
        types.append("変化")
    return types


def is_moved(prev_bbox: List[float], curr_bbox: List[float], threshold: float = 10.0) -> bool:
    px1, py1, px2, py2 = prev_bbox
    cx1, cy1, cx2, cy2 = curr_bbox
    prev_center = ((px1 + px2) / 2.0, (py1 + py2) / 2.0)
    curr_center = ((cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0)
    dist = ((prev_center[0] - curr_center[0]) ** 2 + (prev_center[1] - curr_center[1]) ** 2) ** 0.5
    return dist >= threshold


def select_major_frames(
    summaries: Sequence[FrameSummary],
    top_k: int,
    min_gap: int,
) -> List[FrameSummary]:
    picked: List[FrameSummary] = []
    for summary in sorted(summaries, key=lambda s: (-s.total_area, s.frame)):
        if len(picked) >= top_k:
            break
        if any(abs(summary.frame - other.frame) < min_gap for other in picked):
            continue
        picked.append(summary)
    return sorted(picked, key=lambda s: s.frame)


def read_frame(cap: cv2.VideoCapture, frame_index: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def draw_events(frame: np.ndarray, events: Sequence[Event]) -> np.ndarray:
    out = frame.copy()
    for e in events:
        if e.bbox is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in e.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(out.shape[1] - 1, x2), min(out.shape[0] - 1, y2)
        label = e.label or e.event
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(
            out,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def build_crop_montage(
    frame: np.ndarray,
    events: Sequence[Event],
    thumb_size: int,
    max_crops: int,
) -> Optional[np.ndarray]:
    crops: List[np.ndarray] = []
    for e in events:
        if e.bbox is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in e.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crops.append(crop)
        if len(crops) >= max_crops:
            break

    if not crops:
        return None

    thumbs = [cv2.resize(c, (thumb_size, thumb_size)) for c in crops]
    cols = min(4, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    montage = np.zeros((rows * thumb_size, cols * thumb_size, 3), dtype=np.uint8)
    montage[:] = 245

    for idx, thumb in enumerate(thumbs):
        r = idx // cols
        c = idx % cols
        y0, y1 = r * thumb_size, (r + 1) * thumb_size
        x0, x1 = c * thumb_size, (c + 1) * thumb_size
        montage[y0:y1, x0:x1] = thumb

    return montage


def frame_to_time(frame_idx: int, fps: float) -> str:
    if fps <= 0:
        return f"{frame_idx}f"
    seconds = frame_idx / fps
    minutes = int(seconds // 60)
    sec = seconds - minutes * 60
    return f"{minutes:02d}:{sec:05.2f}"


def render_html(
    output_dir: Path,
    video_name: str,
    fps: float,
    summaries: List[FrameSummary],
    image_pairs: Dict[int, Tuple[str, Optional[str]]],
) -> Path:
    blocks = []
    for idx, summary in enumerate(summaries, start=1):
        full_img, crop_img = image_pairs.get(summary.frame, ("", None))
        counts = summary.counts
        time = frame_to_time(summary.frame, fps)
        change_desc = " / ".join(summary.change_types)
        crop_html = f"<img src=\"{crop_img}\" class=\"thumb\" />" if crop_img else ""
        blocks.append(
            "<section class=\"change\">"
            f"<h2>大きな変化{idx}</h2>"
            f"<div class=\"meta\">タイミング: {time} / {change_desc} / new={counts.get('new', 0)}, update={counts.get('update', 0)}, lost={counts.get('lost', 0)}</div>"
            f"<div class=\"main-image\"><img src=\"{full_img}\" /></div>"
            f"<div class=\"sub-image\">{crop_html}</div>"
            "</section>"
        )

    html = f"""<!doctype html>
<html lang=\"ja\">
<head>
  <meta charset=\"utf-8\" />
  <title>Monitoring Summary - {video_name}</title>
  <style>
    body {{ font-family: "Noto Sans JP", sans-serif; background: #f5f5f5; margin: 0; padding: 24px; }}
    h1 {{ margin-bottom: 8px; }}
    .summary {{ color: #555; margin-bottom: 24px; }}
    .change {{ background: #fff; padding: 16px 20px; margin-bottom: 20px; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
    .meta {{ color: #666; margin-bottom: 12px; }}
    .main-image img {{ max-width: 100%; border-radius: 8px; border: 1px solid #ddd; }}
    .sub-image {{ margin-top: 12px; }}
    .thumb {{ max-width: 480px; border-radius: 6px; border: 1px solid #ddd; }}
  </style>
</head>
<body>
  <h1>Monitoring Summary</h1>
  <div class=\"summary\">動画: {video_name} / fps: {fps:.2f} / 主要変化数: {len(summaries)}</div>
  {''.join(blocks) if blocks else '<p>大きな変化が検出されませんでした。</p>'}
</body>
</html>
"""
    out_path = output_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="monitor_fastsam.py 出力からサマリー HTML レポートを生成")
    p.add_argument("--input-dir", type=Path, default=Path("monitor_output"), help="monitor_output ディレクトリ")
    p.add_argument("--video", type=Path, default=Path("sample_video.mp4"), help="入力動画")
    p.add_argument("--output-dir", type=Path, default=Path("report_output"), help="レポート出力先")
    p.add_argument("--top-k", type=int, default=5, help="大きな変化として抽出するフレーム数")
    p.add_argument("--min-gap-frames", type=int, default=15, help="大きな変化同士の最小フレーム間隔")
    p.add_argument("--thumb-size", type=int, default=160, help="変化サムネイルの1枚あたりのサイズ")
    p.add_argument("--max-crops", type=int, default=8, help="サムネイルに含める最大オブジェクト数")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_path = args.input_dir / "monitor_log.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"ログが見つかりません: {log_path}")
    if not args.video.exists():
        raise FileNotFoundError(f"動画が見つかりません: {args.video}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    rows = load_logs(log_path)
    summaries_dict = build_frame_summaries(rows)
    summaries = select_major_frames(
        list(summaries_dict.values()),
        top_k=args.top_k,
        min_gap=args.min_gap_frames,
    )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"動画が開けません: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    image_pairs: Dict[int, Tuple[str, Optional[str]]] = {}
    for summary in summaries:
        frame = read_frame(cap, summary.frame)
        if frame is None:
            continue
        labeled = draw_events(frame, summary.events)
        full_name = f"change_f{summary.frame:06d}.jpg"
        full_path = frames_dir / full_name
        cv2.imwrite(str(full_path), labeled)

        crop_img = build_crop_montage(frame, summary.events, args.thumb_size, args.max_crops)
        crop_rel = None
        if crop_img is not None:
            crop_name = f"change_f{summary.frame:06d}_crops.jpg"
            crop_path = frames_dir / crop_name
            cv2.imwrite(str(crop_path), crop_img)
            crop_rel = str(crop_path.relative_to(output_dir))

        image_pairs[summary.frame] = (str(full_path.relative_to(output_dir)), crop_rel)

    cap.release()

    report_path = render_html(output_dir, args.video.name, fps, summaries, image_pairs)
    print(f"report saved: {report_path}")


if __name__ == "__main__":
    main()
