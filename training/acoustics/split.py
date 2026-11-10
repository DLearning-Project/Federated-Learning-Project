"""Leakage-safe train / validation / test split of QiandaoEar22.

    .venv/Scripts/python.exe split.py

Reads results/audit/inventory.csv (from audit.py) and writes
splits/split_manifest.csv (one row per 3-second example), its SHA-256 and
splits/split_summary.json.

Rules (recorded in TRAINING.md, Phase 2):

- Only canonical (de-duplicated) files that read correctly are used.
- Label 1 "vessel": every ship clip, including clips where all vessels are far
  and weak (flagged `all_far_weak` so results can be reported separately).
- Label 0 "no vessel": 3-second windows of the 5-minute background recordings.
- Excluded: clips whose only sources are not watercraft (qiandao.NON_VESSEL_SOURCES).
- Grouping: a whole session goes to one split. A session is a run of
  back-to-back recordings (ship clips) or consecutively numbered files on the
  same card (noise), because neighbouring recordings hear the same vessels and
  conditions. Each class is split separately so both classes reach every split.
- Blocks: a ship session longer than BLOCK_RECORDINGS recordings (30 min) is
  cut into blocks, with one 5-minute recording held back between blocks as a
  gap. A gap joins its neighbours when both land in the same split and is
  dropped otherwise, so no two splits ever hold recordings less than 5 minutes
  apart. Without this, two sessions holding 71% of the near-vessel clips are
  each larger than a whole 15% split.
- Balance: among block assignments, choose the one closest to the target
  fractions and, for vessel clips, to the overall mix of closest-vessel
  distance (near / medium / far) in every split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

import qiandao

HERE = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
DEFAULT_FRACTIONS = (0.70, 0.15, 0.15)
SEARCH_RESTARTS = 300
MIX_WEIGHT = 1.0  # weight of the distance-mix mismatch against the size mismatch
DISTANCES = ("N", "M", "F")
BLOCK_RECORDINGS = 6  # 6 back-to-back 5-minute recordings = 30 minutes

MANIFEST_FIELDS = [
    "clip_id", "sha256", "offset_s", "duration_s", "label", "label_name", "group",
    "recording", "split", "closest_distance", "all_far_weak", "multi", "sources",
]


def _true(value: str) -> bool:
    return value == "True"


def closest_distance(sources: str) -> str:
    """N, M or F for the nearest source in an inventory `sources` string ("?" if none is given)."""
    found = {s.split(":")[1] for s in sources.split("&") if s.count(":") == 2}
    return next((d for d in DISTANCES if d in found), "?")


def build_examples(inventory: list[dict]) -> tuple[list[dict], Counter]:
    """3-second examples with label and group, plus counts of excluded files by reason."""
    examples: list[dict] = []
    excluded: Counter = Counter()
    for row in inventory:
        if not _true(row["canonical"]):
            continue
        if not _true(row["readable"]):
            excluded["unreadable"] += 1
            continue
        if row["kind"] == "ship":
            if _true(row["non_vessel_only"]):
                excluded["non_vessel_only"] += 1
                continue
            examples.append({
                "clip_id": row["path"], "sha256": row["sha256"], "offset_s": 0.0,
                "duration_s": qiandao.CLIP_SECONDS, "label": 1, "label_name": "vessel",
                "group": f"ship/{row['session']}", "recording": row["recording_ts"],
                "closest_distance": closest_distance(row["sources"]),
                "all_far_weak": row["all_far_weak"], "multi": row["multi"], "sources": row["sources"],
            })
        elif row["kind"] == "noise":
            windows = int(float(row["duration_s"]) // qiandao.CLIP_SECONDS)
            for w in range(windows):
                examples.append({
                    "clip_id": row["path"], "sha256": row["sha256"], "offset_s": w * qiandao.CLIP_SECONDS,
                    "duration_s": qiandao.CLIP_SECONDS, "label": 0, "label_name": "no_vessel",
                    "group": f"noise/{row['session']}", "recording": f"{row['card']}/DATA{int(row['noise_num']):04d}",
                    "closest_distance": "", "all_far_weak": "", "multi": "", "sources": "",
                })
        else:
            excluded["unknown_name"] += 1
    return examples, excluded


def split_score(totals: np.ndarray, fractions: np.ndarray, mix_weight: float = MIX_WEIGHT) -> float:
    """Mismatch of one assignment; `totals` is splits x categories (example counts).

    Size term: sum over splits of |share of examples - target fraction|.
    Mix term: sum over splits of the total variation distance between the
    split's category mix and the overall mix (0 = identical, 1 = disjoint).
    """
    sizes = totals.sum(axis=1)
    if (sizes == 0).any():
        return float("inf")
    overall = totals.sum(axis=0) / sizes.sum()
    size_term = np.abs(sizes / sizes.sum() - fractions).sum()
    mix_term = 0.5 * np.abs(totals / sizes[:, None] - overall).sum()
    return float(size_term + mix_weight * mix_term)


def assign_groups(
    counts: dict[str, Counter],
    fractions: tuple[float, ...],
    seed: int,
    restarts: int = SEARCH_RESTARTS,
    mix_weight: float = MIX_WEIGHT,
) -> dict[str, str]:
    """Group -> split with the lowest `split_score`, keeping every group whole.

    `counts[group]` counts the group's examples per category (for example
    closest-vessel distance). Each restart deals the shuffled groups to the
    split furthest below its target size, then improves the assignment by
    moving one group or swapping two until no move lowers the score. The best
    of `restarts` seeded restarts wins.
    """
    if len(counts) < len(SPLITS):
        raise ValueError(f"need at least {len(SPLITS)} groups to split, got {len(counts)}")
    names = sorted(counts)
    categories = sorted({c for counter in counts.values() for c in counter})
    matrix = np.array([[counts[g][c] for c in categories] for g in names], dtype=float)
    sizes = matrix.sum(axis=1)
    target = np.asarray(fractions, dtype=float)
    k = len(SPLITS)
    rng = random.Random(seed)

    def totals_of(assign: np.ndarray) -> np.ndarray:
        return np.stack([matrix[assign == s].sum(axis=0) for s in range(k)])

    best_score, best_assign = float("inf"), None
    for _ in range(restarts):
        order = list(range(len(names)))
        rng.shuffle(order)
        assign = np.zeros(len(names), dtype=int)
        filled = np.zeros(k)
        for placed, g in enumerate(order):
            empty = [s for s in range(k) if filled[s] == 0]
            if empty and len(order) - placed <= len(empty):
                s = empty[0]  # the remaining groups must fill the empty splits
            else:
                s = int(np.argmax(target * sizes.sum() - filled))
            assign[g] = s
            filled[s] += sizes[g]

        totals = totals_of(assign)
        score = split_score(totals, target, mix_weight)
        while True:
            best_move = None
            for g in range(len(names)):
                for s in range(k):
                    if s == assign[g]:
                        continue
                    trial = totals.copy()
                    trial[assign[g]] -= matrix[g]
                    trial[s] += matrix[g]
                    new = split_score(trial, target, mix_weight)
                    if new < score - 1e-12 and (best_move is None or new < best_move[0]):
                        best_move = (new, [(g, s)])
                for h in range(g + 1, len(names)):
                    if assign[h] == assign[g]:
                        continue
                    trial = totals.copy()
                    trial[assign[g]] += matrix[h] - matrix[g]
                    trial[assign[h]] += matrix[g] - matrix[h]
                    new = split_score(trial, target, mix_weight)
                    if new < score - 1e-12 and (best_move is None or new < best_move[0]):
                        best_move = (new, [(g, assign[h]), (h, assign[g])])
            if best_move is None:
                break
            for g, s in best_move[1]:
                assign[g] = s
            totals = totals_of(assign)
            score = best_move[0]
        if score < best_score - 1e-12:
            best_score, best_assign = score, assign.copy()
    if best_assign is None:
        raise ValueError("no assignment filled every split")
    return {names[i]: SPLITS[s] for i, s in enumerate(best_assign)}


def make_blocks(examples: list[dict], block_recordings: int = BLOCK_RECORDINGS) -> dict[str, tuple[str, str]]:
    """Cut long ship sessions into blocks, rewriting each vessel example's group.

    Returns gap group -> (block before, block after). A leftover of a single
    recording joins the last block instead of becoming a block of its own.
    """
    recordings: dict[str, set[str]] = defaultdict(set)
    for e in examples:
        if e["label"] == 1:
            recordings[e["group"]].add(e["recording"])
    group_of_recording: dict[str, str] = {}
    gaps: dict[str, tuple[str, str]] = {}
    for session, recs in recordings.items():
        order = sorted(recs)
        part, i = 0, 0
        while i < len(order):
            end = i + block_recordings
            if len(order) - end <= 1:
                end = len(order)
            block = f"{session}#{part}" if end - i < len(order) else session
            for r in order[i:end]:
                group_of_recording[r] = block
            i = end
            if i < len(order):
                gap = f"{session}#gap{part}"
                group_of_recording[order[i]] = gap
                gaps[gap] = (block, f"{session}#{part + 1}")
                i += 1
            part += 1
    for e in examples:
        if e["label"] == 1:
            e["group"] = group_of_recording[e["recording"]]
    return gaps


def make_split(examples: list[dict], fractions: tuple[float, ...], seed: int) -> dict[str, str | None]:
    """Split each class's groups separately; returns group -> split (None = gap dropped).

    Vessel blocks are balanced on closest-vessel distance as well as size;
    noise windows have no distance, so only size counts for them.
    """
    gaps = make_blocks(examples)
    group_of_split: dict[str, str | None] = {}
    for label in sorted({e["label"] for e in examples}):
        counts: dict[str, Counter] = defaultdict(Counter)
        for e in examples:
            if e["label"] == label and e["group"] not in gaps:
                counts[e["group"]][e["closest_distance"] or "none"] += 1
        group_of_split.update(assign_groups(dict(counts), fractions, seed + label))
    for gap, (before, after) in gaps.items():
        same = group_of_split[before] == group_of_split[after]
        group_of_split[gap] = group_of_split[before] if same else None
    return group_of_split


def check_no_leakage(examples: list[dict]) -> None:
    """Every group, recording and audio file content must sit in exactly one split,
    and back-to-back ship recordings (no gap between them) must share a split."""
    for key in ("group", "recording", "sha256"):
        splits_of: dict[str, set[str]] = defaultdict(set)
        for e in examples:
            splits_of[e[key]].add(e["split"])
        leaked = [k for k, s in splits_of.items() if len(s) > 1]
        if leaked:
            raise AssertionError(f"{len(leaked)} {key} values appear in more than one split, e.g. {leaked[:3]}")
    split_of_recording = {e["recording"]: e["split"] for e in examples if e.get("label") in (1, "1")}
    order = sorted(split_of_recording)
    for a, b in zip(order, order[1:]):
        apart = (datetime.strptime(b, "%Y%m%d%H%M%S") - datetime.strptime(a, "%Y%m%d%H%M%S")).total_seconds()
        if apart <= qiandao.SESSION_GAP_S and split_of_recording[a] != split_of_recording[b]:
            raise AssertionError(f"back-to-back recordings {a} and {b} are in different splits")


def summarize(examples: list[dict], excluded: Counter, dropped: list[dict], fractions, seed) -> dict:
    per_split = {}
    for s in SPLITS:
        rows = [e for e in examples if e["split"] == s]
        ships = [e for e in rows if e["label"] == 1]
        per_split[s] = {
            "examples": len(rows),
            "vessel": len(ships),
            "no_vessel": len(rows) - len(ships),
            "share_of_vessel": round(len(ships) / max(1, sum(e["label"] == 1 for e in examples)), 4),
            "share_of_no_vessel": round((len(rows) - len(ships)) / max(1, sum(e["label"] == 0 for e in examples)), 4),
            "vessel_all_far_weak": sum(e["all_far_weak"] == "True" for e in ships),
            "vessel_closest_distance": {
                d: round(sum(e["closest_distance"] == d for e in ships) / max(1, len(ships)), 4) for d in DISTANCES
            },
            "groups": sorted({e["group"] for e in rows}),
            "recordings": len({e["recording"] for e in rows}),
            "days": dict(sorted(Counter(e["recording"][:8] for e in ships).items())),
            "clips_per_source_name": dict(Counter(
                s.split(":")[0] for e in ships for s in e["sources"].split("&")
            ).most_common()),
        }
    return {
        "seed": seed,
        "target_fractions": dict(zip(SPLITS, fractions)),
        "grouping": "session (back-to-back ship recordings; consecutively numbered noise files per card); "
                    f"ship sessions over {BLOCK_RECORDINGS} recordings cut into blocks with a 1-recording gap",
        "balance": f"size vs. target fractions + {MIX_WEIGHT} x closest-vessel distance mix (vessel clips); "
                   f"best of {SEARCH_RESTARTS} seeded restarts with move/swap improvement",
        "label_rule": "vessel = ship clip (all_far_weak kept, flagged); no_vessel = 3 s windows of background files",
        "excluded_files": dict(excluded),
        "gap_clips_dropped": len(dropped),
        "gap_recordings_dropped": sorted({e["recording"] for e in dropped}),
        "per_split": per_split,
    }


def run_split(inventory_path: Path, out_dir: Path, fractions: tuple[float, ...], seed: int) -> dict:
    with inventory_path.open(newline="", encoding="utf-8") as fh:
        inventory = list(csv.DictReader(fh))
    examples, excluded = build_examples(inventory)
    group_split = make_split(examples, fractions, seed)
    for e in examples:
        e["split"] = group_split[e["group"]]
    dropped = [e for e in examples if e["split"] is None]
    examples = [e for e in examples if e["split"] is not None]
    check_no_leakage(examples)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "split_manifest.csv"
    examples.sort(key=lambda e: (e["clip_id"], e["offset_s"]))
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, lineterminator="\n")  # LF: hashes match on every OS
        writer.writeheader()
        for e in examples:
            writer.writerow({f: e[f] for f in MANIFEST_FIELDS})
    with manifest.open("rb") as fh:
        digest = hashlib.file_digest(fh, "sha256").hexdigest()
    (out_dir / "split_manifest.csv.sha256").write_text(f"{digest}  split_manifest.csv\n", encoding="utf-8", newline="\n")

    summary = summarize(examples, excluded, dropped, fractions, seed)
    with inventory_path.open("rb") as fh:
        summary["inventory_sha256"] = hashlib.file_digest(fh, "sha256").hexdigest()
    summary["manifest_sha256"] = digest
    (out_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inventory", type=Path, default=HERE / "results" / "audit" / "inventory.csv")
    parser.add_argument("--out", type=Path, default=HERE / "splits")
    parser.add_argument("--fractions", type=float, nargs=3, default=DEFAULT_FRACTIONS, metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()
    if abs(sum(args.fractions) - 1) > 1e-9:
        parser.error("fractions must add up to 1")
    summary = run_split(args.inventory, args.out, tuple(args.fractions), args.seed)
    for s, info in summary["per_split"].items():
        print(f"{s:5}: {info['vessel']:6} vessel ({info['share_of_vessel']:.1%}), "
              f"{info['no_vessel']:6} no-vessel ({info['share_of_no_vessel']:.1%}), {len(info['groups'])} groups, "
              "near/medium/far " + " / ".join(f"{v:.0%}" for v in info["vessel_closest_distance"].values()))
    print(f"Excluded files: {summary['excluded_files']}; gap clips dropped: {summary['gap_clips_dropped']}")
    print(f"Manifest SHA-256: {summary['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
