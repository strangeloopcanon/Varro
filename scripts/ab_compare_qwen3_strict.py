#!/usr/bin/env python3
"""
A/B comparison for Qwen3-1.7B strict paragraph models on a fixed headline set.

Usage
  python scripts/ab_compare_qwen3_strict.py \
    --date 20250904 \
    --ckpt_a training/checkpoints/gspo_QWEN3_17B_STRICT_CHAT_ART_RERUN/final_model_20250829 \
    --ckpt_b training/checkpoints/gspo_QWEN3_17B_STRICT_CHAT_ART_RERUN/final_model \
    --run_suffix QWEN3_17B_STRICT_CHAT_ART_RERUN \
    --num_rollouts 4 \
    --sampler_profile tight \
    --limit 20 \
    --with_articles

Produces a concise Markdown report at:
  reports/AB_<SUFFIX>_<DATE>.md
with per‑arm metrics and deltas (B − A).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_collection.timestamped_storage import TimestampedStorage
from data_collection.article_cleaning import build_link_to_excerpt_map, normalize_url
from prediction_generation.adaptive_rollout_generator import AdaptiveRolloutGenerator


LEAK_PATTERNS = [
    r"\bkeep (it|to)\b",
    r"\bdo not\b",
    r"\bavoid\b",
    r"\buse (emojis|markdown|bold|bullet|headers|one paragraph)\b",
    r"\banswer:\b",
    r"\bfinal answer\b",
    r"\bnote:\b",
    r"\bthe following (is|are)\b",
    r"\bstart with\b",
    r"\bmake sure\b",
    r"\bin this forecast\b",
    r"\byou must\b",
    r"\bthe user wants\b",
]


@dataclass
class ArmResult:
    name: str
    q_avg: float
    leak_share: float
    avg_words: float
    zeros: int
    rollouts: int


def load_headlines(date: str, limit: int | None = None, with_articles: bool = False, run_suffix: str | None = None) -> List[Dict[str, Any]]:
    """Load base headlines and optionally attach cleaned article excerpts for the date."""
    storage = TimestampedStorage()
    j = storage.load_data("headlines", date)
    if not j:
        raise SystemExit(f"No base headlines found for {date}")
    heads = list(j.get("headlines", []))
    if limit is not None:
        heads = heads[: max(1, int(limit))]
    if with_articles:
        # Build link→excerpt map using the run-specific storage (if provided)
        if run_suffix:
            os.environ["VARRO_RUN_DIR_SUFFIX"] = run_suffix
        link_map = build_link_to_excerpt_map(date, TimestampedStorage())
        for h in heads:
            link = h.get("link") or h.get("url") or h.get("href")
            if not link:
                continue
            ex = link_map.get(link)
            if not ex:
                try:
                    ex = link_map.get(normalize_url(link))
                except Exception:
                    ex = None
            if ex:
                h["article_excerpt"] = ex
    return heads


def gen_predictions(ckpt: str, headlines: List[Dict[str, Any]], num_rollouts: int, sampler_profile: str) -> List[Dict[str, Any]]:
    # Match strict chat recipe
    os.environ["VARRO_MODEL_NAME"] = "Qwen/Qwen3-1.7B"
    os.environ["VARRO_USE_CHAT"] = "1"
    os.environ["VARRO_USE_HF_CHAT_TEMPLATE"] = "1"
    os.environ["VARRO_CHAT_THINKING"] = "0"
    os.environ["VARRO_FORMAT_STRICT"] = "1"
    os.environ["VARRO_PARAGRAPH_MAX_TOKENS"] = "180"

    gen = AdaptiveRolloutGenerator(
        checkpoint_path=ckpt, sampler_profile=sampler_profile, output_format="paragraph"
    )
    # ensure sampler matches requested profile
    gen.sampler_profile = sampler_profile
    gen.sampler = gen._create_sampler(sampler_profile)

    preds = gen.generate_daily_predictions(headlines, num_rollouts=num_rollouts)
    return preds


def compute_metrics(preds: List[Dict[str, Any]]) -> Tuple[float, float, float, int, int]:
    rx = re.compile("|".join(LEAK_PATTERNS), re.I)
    words: List[int] = []
    qvals: List[float] = []
    leaks = 0
    zeros = 0
    rolls = 0
    for p in preds:
        for r in p.get("rollouts", []):
            txt = (r.get("prediction") or "").strip()
            v = r.get("immediate_reward")
            if txt:
                words.append(len(txt.split()))
                if rx.search(txt):
                    leaks += 1
            if isinstance(v, (int, float)):
                qvals.append(float(v))
                if float(v) == 0.0:
                    zeros += 1
            rolls += 1
    q_avg = round(sum(qvals) / len(qvals), 3) if qvals else 0.0
    leak_share = round(leaks / max(1, rolls), 3)
    avg_words = round(mean(words), 1) if words else 0.0
    return q_avg, leak_share, avg_words, zeros, rolls


def write_report(path: Path, date: str, suffix: str, a: ArmResult, b: ArmResult):
    lines: List[str] = []
    lines.append(f"### A/B — {suffix} on {date}")
    lines.append("")
    lines.append(f"- A: {a.name}")
    lines.append(f"- B: {b.name}")
    lines.append("")
    lines.append("#### Metrics (paragraph mode, tight sampler)")
    lines.append(f"- q_avg: A={a.q_avg:.3f}, B={b.q_avg:.3f}, delta={b.q_avg - a.q_avg:+.3f}")
    lines.append(f"- leak_share: A={a.leak_share:.3f}, B={b.leak_share:.3f}, delta={b.leak_share - a.leak_share:+.3f}")
    lines.append(f"- avg_words: A={a.avg_words:.1f}, B={b.avg_words:.1f}, delta={b.avg_words - a.avg_words:+.1f}")
    lines.append(f"- zeros: A={a.zeros} /{a.rollouts}, B={b.zeros} /{b.rollouts}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(str(path))


def main() -> int:
    ap = argparse.ArgumentParser(description="A/B compare two checkpoints on the same headlines")
    ap.add_argument("--date", required=True, help="YYYYMMDD date for headline set")
    ap.add_argument("--ckpt_a", required=True, help="Path to checkpoint A (early)")
    ap.add_argument("--ckpt_b", required=True, help="Path to checkpoint B (late/final)")
    ap.add_argument("--run_suffix", default="QWEN3_17B_STRICT_CHAT_ART_RERUN")
    ap.add_argument("--num_rollouts", type=int, default=4)
    ap.add_argument("--sampler_profile", default="tight", choices=["tight","default","loose"])
    ap.add_argument("--limit", type=int, default=20, help="Limit number of headlines")
    ap.add_argument("--with_articles", action="store_true")
    args = ap.parse_args()

    # Load headlines and optional excerpts (pinned to provided run suffix)
    heads = load_headlines(args.date, limit=args.limit, with_articles=bool(args.with_articles), run_suffix=args.run_suffix)

    preds_a = gen_predictions(args.ckpt_a, heads, num_rollouts=args.num_rollouts, sampler_profile=args.sampler_profile)
    qa, la, wa, za, ra = compute_metrics(preds_a)
    arm_a = ArmResult(Path(args.ckpt_a).name, qa, la, wa, za, ra)

    preds_b = gen_predictions(args.ckpt_b, heads, num_rollouts=args.num_rollouts, sampler_profile=args.sampler_profile)
    qb, lb, wb, zb, rb = compute_metrics(preds_b)
    arm_b = ArmResult(Path(args.ckpt_b).name, qb, lb, wb, zb, rb)

    out = Path("reports") / f"AB_{args.run_suffix}_{args.date}.md"
    write_report(out, args.date, args.run_suffix, arm_a, arm_b)

    # Also print concise summary
    print(f"A: q_avg={arm_a.q_avg:.3f}, leak={arm_a.leak_share:.3f}, words={arm_a.avg_words:.1f}")
    print(f"B: q_avg={arm_b.q_avg:.3f}, leak={arm_b.leak_share:.3f}, words={arm_b.avg_words:.1f}")
    print(f"Delta (B-A): q_avg={arm_b.q_avg - arm_a.q_avg:+.3f}, leak={arm_b.leak_share - arm_a.leak_share:+.3f}, words={arm_b.avg_words - arm_a.avg_words:+.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
