#!/usr/bin/env python3
"""monitor_fastsam.py の出力から HTML レポートを生成する。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2


@dataclass
class TrackEvent:
    frame: int
    event: str
    bbox: Optional[List[float]]
    label: Optional[str]
    image_file: Optional[str]


@dataclass
class TrackSummary:
    track_id: int
    events: List[TrackEvent]

    @property
    def first_frame(self) -> int:
        return min(e.frame for e in self.events)

    @property
    def last_frame(self) -> int:
        return max(e.frame for e in self.events)

    @property
    def label(self) -> str:
        for e in self.events:
            if e.label:
                return e.label
        return f"object_{self.track_id}"

    @property
    def sample_image(self) -> Optional[str]:
        for e in self.events:
            if e.image_file:
                return e.image_file
        return None


def load_logs(log_path: Path) -> List[Dict]:
    rows = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_tracks(rows: List[Dict]) -> Dict[int, TrackSummary]:
    tracks: Dict[int, List[TrackEvent]] = {}
    for row in rows:
        tid = row.get("track_id")
        if tid is None:
            continue
        bbox = row.get("bbox") or row.get("last_bbox")
        event = TrackEvent(
            frame=int(row.get("frame", 0)),
            event=str(row.get("event", "")),
            bbox=bbox,
            label=row.get("label"),
            image_file=row.get("image_file"),
        )
        tracks.setdefault(int(tid), []).append(event)
    return {tid: TrackSummary(track_id=tid, events=sorted(ev, key=lambda e: e.frame)) for tid, ev in tracks.items()}


def read_frame(cap: cv2.VideoCapture, frame_index: int) -> Optional[cv2.Mat]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def draw_bbox(frame: cv2.Mat, bbox: Optional[List[float]], label: str) -> cv2.Mat:
    if bbox is None:
        return frame
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
    out = frame.copy()
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


def build_bbox_timeline(events: Iterable[TrackEvent]) -> List[Tuple[int, Optional[List[float]]]]:
    timeline = []
    for e in events:
        timeline.append((e.frame, e.bbox))
    return sorted(timeline, key=lambda t: t[0])


def lookup_bbox(timeline: List[Tuple[int, Optional[List[float]]]], frame_idx: int) -> Optional[List[float]]:
    current = None
    for f, bbox in timeline:
        if f > frame_idx:
            break
        if bbox is not None:
            current = bbox
    return current


def frame_to_time(frame_idx: int, fps: float) -> str:
    if fps <= 0:
        return f"{frame_idx}f"
    seconds = frame_idx / fps
    minutes = int(seconds // 60)
    sec = seconds - minutes * 60
    return f"{minutes:02d}:{sec:05.2f}"


def write_track_clip(
    cap: cv2.VideoCapture,
    summary: TrackSummary,
    fps: float,
    out_path: Path,
    max_frames: int = 160,
) -> Optional[Path]:
    span = summary.last_frame - summary.first_frame
    if span <= 0:
        return None
    step = max(1, span // max_frames)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        return None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps / step if fps > 0 else 10.0, (width, height))
    timeline = build_bbox_timeline(summary.events)

    for frame_idx in range(summary.first_frame, summary.last_frame + 1, step):
        frame = read_frame(cap, frame_idx)
        if frame is None:
            continue
        bbox = lookup_bbox(timeline, frame_idx)
        labeled = draw_bbox(frame, bbox, f"{summary.label} ({summary.track_id})")
        writer.write(labeled)

    writer.release()
    return out_path


def render_html(
    output_dir: Path,
    video_name: str,
    fps: float,
    tracks: Dict[int, TrackSummary],
    event_images: Dict[Tuple[int, int], str],
    clip_paths: Dict[int, str],
) -> Path:
    rows = []
    for tid, summary in sorted(tracks.items(), key=lambda item: item[0]):
        first_time = frame_to_time(summary.first_frame, fps)
        last_time = frame_to_time(summary.last_frame, fps)
        duration = frame_to_time(summary.last_frame - summary.first_frame, fps)
        image_html = ""
        if summary.sample_image:
            image_html = f"<img src=\"{summary.sample_image}\" class=\"thumb\" />"

        event_items = []
        for e in summary.events:
            key = (tid, e.frame)
            img = event_images.get(key)
            label = e.event
            time = frame_to_time(e.frame, fps)
            if img:
                event_items.append(
                    f"<div class=\"event\"><div class=\"event-title\">{label} @ {time}</div><img src=\"{img}\" /></div>"
                )
            else:
                event_items.append(f"<div class=\"event\"><div class=\"event-title\">{label} @ {time}</div></div>")

        clip_html = ""
        clip = clip_paths.get(tid)
        if clip:
            clip_html = (
                "<video controls class=\"clip\">"
                f"<source src=\"{clip}\" type=\"video/mp4\" />"
                "</video>"
            )

        rows.append(
            "<section class=\"track\">"
            f"<h2>Track {tid}: {summary.label}</h2>"
            f"<div class=\"meta\">開始 {first_time} / 終了 {last_time} / 継続 {duration}</div>"
            f"<div class=\"sample\">{image_html}</div>"
            f"<div class=\"clip-wrap\">{clip_html}</div>"
            f"<div class=\"events\">{''.join(event_items)}</div>"
            "</section>"
        )

    html = f"""<!doctype html>
<html lang=\"ja\">
<head>
  <meta charset=\"utf-8\" />
  <title>Monitoring Report - {video_name}</title>
  <style>
    body {{ font-family: "Noto Sans JP", sans-serif; background: #f7f7f7; margin: 0; padding: 24px; }}
    h1 {{ margin-bottom: 8px; }}
    .summary {{ color: #555; margin-bottom: 24px; }}
    .track {{ background: #fff; padding: 16px 20px; margin-bottom: 20px; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
    .meta {{ color: #666; margin-bottom: 12px; }}
    .thumb {{ max-width: 160px; border-radius: 6px; border: 1px solid #ddd; }}
    .events {{ display: flex; flex-wrap: wrap; gap: 12px; }}
    .event {{ background: #fafafa; border: 1px solid #e0e0e0; padding: 8px; border-radius: 6px; width: 260px; }}
    .event img {{ width: 100%; border-radius: 4px; }}
    .event-title {{ font-size: 12px; color: #666; margin-bottom: 6px; }}
    .clip {{ max-width: 100%; width: 640px; border-radius: 8px; border: 1px solid #ddd; }}
  </style>
</head>
<body>
  <h1>Monitoring Report</h1>
  <div class=\"summary\">動画: {video_name} / fps: {fps:.2f} / 対象トラック数: {len(tracks)}</div>
  {''.join(rows) if rows else '<p>イベントがありません。</p>'}
</body>
</html>
"""
    out_path = output_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="monitor_fastsam.py 出力から HTML レポートを生成")
    p.add_argument("--input-dir", type=Path, required=True, help="monitor_output ディレクトリ")
    p.add_argument("--video", type=Path, required=True, help="入力動画")
    p.add_argument("--output-dir", type=Path, default=None, help="レポート出力先")
    p.add_argument("--max-clip-frames", type=int, default=160, help="クリップの最大フレーム数")
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
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    rows = load_logs(log_path)
    tracks = build_tracks(rows)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"動画が開けません: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    event_images: Dict[Tuple[int, int], str] = {}
    for tid, summary in tracks.items():
        for e in summary.events:
            if e.bbox is None:
                continue
            frame = read_frame(cap, e.frame)
            if frame is None:
                continue
            labeled = draw_bbox(frame, e.bbox, f"{summary.label} ({tid})")
            fname = f"track{tid}_f{e.frame:06d}_{e.event}.jpg"
            out_path = frames_dir / fname
            cv2.imwrite(str(out_path), labeled)
            event_images[(tid, e.frame)] = str(out_path.relative_to(output_dir))

    clip_paths: Dict[int, str] = {}
    for tid, summary in tracks.items():
        out_path = clips_dir / f"track{tid:03d}.mp4"
        clip = write_track_clip(cap, summary, fps, out_path, max_frames=args.max_clip_frames)
        if clip:
            clip_paths[tid] = str(clip.relative_to(output_dir))

    cap.release()

    report_path = render_html(output_dir, args.video.name, fps, tracks, event_images, clip_paths)
    print(f"report saved: {report_path}")


if __name__ == "__main__":
    main()
