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


@dataclass
class FrameSummary:
    frame: int
    events: List[Event]
    score: float

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


def build_frame_summaries(rows: List[Dict]) -> Dict[int, FrameSummary]:
    grouped: Dict[int, List[Event]] = {}
    for row in rows:
        frame = int(row.get("frame", 0))
        bbox = row.get("bbox") or row.get("last_bbox")
        event = Event(
            frame=frame,
            event=str(row.get("event", "")),
            bbox=bbox,
            label=row.get("label"),
            image_file=row.get("image_file"),
        )
        grouped.setdefault(frame, []).append(event)

    summaries: Dict[int, FrameSummary] = {}
    for frame, events in grouped.items():
        score = 0.0
        for e in events:
            if e.event == "new":
                score += 3.0
            elif e.event == "lost":
                score += 3.0
            else:
                score += 1.0
        summaries[frame] = FrameSummary(frame=frame, events=events, score=score)
    return summaries


def select_major_frames(
    summaries: Sequence[FrameSummary],
    top_k: int,
    min_gap: int,
) -> List[FrameSummary]:
    picked: List[FrameSummary] = []
    for summary in sorted(summaries, key=lambda s: (-s.score, s.frame)):
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
        crop_html = f"<img src=\"{crop_img}\" class=\"thumb\" />" if crop_img else ""
        blocks.append(
            "<section class=\"change\">"
            f"<h2>大きな変化{idx}</h2>"
            f"<div class=\"meta\">タイミング: {time} / new={counts.get('new', 0)}, update={counts.get('update', 0)}, lost={counts.get('lost', 0)}</div>"
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
    p.add_argument("--input-dir", type=Path, required=True, help="monitor_output ディレクトリ")
    p.add_argument("--video", type=Path, required=True, help="入力動画")
    p.add_argument("--output-dir", type=Path, default=None, help="レポート出力先")
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

    output_dir = args.output_dir or (args.input_dir / "report")
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
