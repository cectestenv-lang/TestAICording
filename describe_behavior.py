#!/usr/bin/env python3
"""monitor_fastsam.py の出力を要約し、LLM へ説明依頼を投げる。"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import requests


def load_logs(log_path: Path) -> List[Dict]:
    rows = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_prompt(rows: List[Dict], output_dir: Path, video_name: str) -> str:
    events = Counter(r["event"] for r in rows)
    by_track = defaultdict(list)
    for r in rows:
        tid = r.get("track_id")
        if tid is not None:
            by_track[tid].append(r)

    lines = []
    lines.append(f"動画名: {video_name}")
    lines.append("以下は FastSAM モニタリング結果です。")
    lines.append(f"イベント数: new={events.get('new', 0)}, update={events.get('update', 0)}, lost={events.get('lost', 0)}")
    lines.append("")
    lines.append("トラック別サマリ:")

    for tid, hist in sorted(by_track.items(), key=lambda x: x[0]):
        first = min(h.get("frame", 0) for h in hist)
        last = max(h.get("frame", 0) for h in hist)
        new_event = next((h for h in hist if h.get("event") == "new"), {})
        image_file = new_event.get("image_file")
        mean_area = sum(h.get("area_ratio", 0) for h in hist if "area_ratio" in h) / max(
            1, sum(1 for h in hist if "area_ratio" in h)
        )
        lines.append(
            f"- track_id={tid}, 出現={first}f, 最終={last}f, 平均面積比={mean_area:.4f}, 参照画像={image_file or 'なし'}"
        )

    lines.append("")
    lines.append("依頼:")
    lines.append("1) 動画の主要な挙動を時系列で説明してください。")
    lines.append("2) 各物体がどのように出現・移動・消失したかを説明してください。")
    lines.append("3) 異常や注目すべき変化があれば指摘してください。")
    lines.append("4) 必要なら追加で確認すべき映像区間を提案してください。")
    lines.append("")
    lines.append("補足: 参照画像パスは出力ディレクトリ配下の相対パスです。")

    return "\n".join(lines)


def call_openai(prompt: str, model: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が未設定です。")

    resp = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": [
                {
                    "role": "system",
                    "content": "あなたは動画監視ログを解析するアシスタントです。"},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("output_text", "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="モニタリングログから LLM 用説明文を作成")
    p.add_argument("--input-dir", type=Path, required=True, help="monitor_fastsam.py の出力ディレクトリ")
    p.add_argument("--video-name", default="unknown_video")
    p.add_argument("--prompt-out", type=Path, default=None, help="生成プロンプトの保存先")
    p.add_argument("--send", action="store_true", help="LLM へ送信する")
    p.add_argument("--model", default="gpt-4.1-mini")
    p.add_argument("--response-out", type=Path, default=None, help="LLM 応答の保存先")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_path = args.input_dir / "monitor_log.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"ログが見つかりません: {log_path}")

    rows = load_logs(log_path)
    prompt = build_prompt(rows, args.input_dir, args.video_name)

    prompt_out = args.prompt_out or (args.input_dir / "llm_prompt.txt")
    prompt_out.write_text(prompt, encoding="utf-8")
    print(f"prompt saved: {prompt_out}")

    if args.send:
        answer = call_openai(prompt, args.model)
        out = args.response_out or (args.input_dir / "llm_response.txt")
        out.write_text(answer, encoding="utf-8")
        print(f"response saved: {out}")


if __name__ == "__main__":
    main()
