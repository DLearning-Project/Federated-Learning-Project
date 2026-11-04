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
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import qiandao

HERE = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
DEFAULT_FRACTIONS = (0.70, 0.15, 0.15)
SEARCH_TRIALS = 20000

MANIFEST_FIELDS = [
    "clip_id", "sha256", "offset_s", "duration_s", "label", "label_name", "group",
    "recording", "split", "all_far_weak", "multi", "sources",
]


def _true(value: str) -> bool:
    return value == "True"


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
                "all_far_weak": row["all_far_weak"], "multi": row["multi"], "sources": row["sources"],
            })
        elif row["kind"] == "noise":
            windows = int(float(row["duration_s"]) // qiandao.CLIP_SECONDS)
            for w in range(windows):
                examples.append({
                    "clip_id": row["path"], "sha256": row["sha256"], "offset_s": w * qiandao.CLIP_SECONDS,
                    "duration_s": qiandao.CLIP_SECONDS, "label": 0, "label_name": "no_vessel",
                    "group": f"noise/{row['session']}", "recording": f"{row['card']}/DATA{int(row['noise_num']):04d}",
                    "all_far_weak": "", "multi": "", "sources": "",
                })
        else:
            excluded["unknown_name"] += 1
    return examples, excluded


def _score(totals: dict[str, int], all_count: int, fractions: tuple[float, ...]) -> float:
    return sum(abs(totals[s] / all_count - f) for s, f in zip(SPLITS, fractions))


def assign_groups(sizes: dict[str, int], fractions: tuple[float, ...], seed: int, trials: int = SEARCH_TRIALS) -> dict[str, str]:
    """Group -> split, as close to `fractions` (by example count) as whole groups allow.

    Each trial shuffles the groups and gives each one to the split furthest
    below its target; the best of `trials` seeded trials wins. Every split
    gets at least one group when there are enough groups.
    """
    if len(sizes) < len(SPLITS):
        raise ValueError(f"need at least {len(SPLITS)} groups to split, got {len(sizes)}")
    rng = random.Random(seed)
    names = sorted(sizes)
    total = sum(sizes.values())
    best: tuple[float, dict[str, str]] | None = None
    for _ in range(trials):
        rng.shuffle(names)
        totals = dict.fromkeys(SPLITS, 0)
        choice: dict[str, str] = {}
        for g in names:
            empty = [s for s in SPLITS if totals[s] == 0]
            if empty and len(names) - len(choice) <= len(empty):
                target = empty[0]  # the remaining groups must fill the empty splits
            else:
                target = max(SPLITS, key=lambda s: fractions[SPLITS.index(s)] * total - totals[s])
            choice[g] = target
            totals[target] += sizes[g]
        if any(v == 0 for v in totals.values()):
            continue
        score = _score(totals, total, fractions)
        if best is None or score < best[0]:
            best = (score, dict(choice))
    if best is None:
        raise ValueError("no assignment filled every split")
    return best[1]


def make_split(examples: list[dict], fractions: tuple[float, ...], seed: int) -> dict[str, str]:
    """Split each class's groups separately; returns group -> split."""
    group_of_split: dict[str, str] = {}
    for label in sorted({e["label"] for e in examples}):
        sizes = Counter(e["group"] for e in examples if e["label"] == label)
        group_of_split.update(assign_groups(dict(sizes), fractions, seed + label))
    return group_of_split


def check_no_leakage(examples: list[dict]) -> None:
    """Every group, recording and audio file content must sit in exactly one split."""
    for key in ("group", "recording", "sha256"):
        splits_of: dict[str, set[str]] = defaultdict(set)
        for e in examples:
            splits_of[e[key]].add(e["split"])
        leaked = [k for k, s in splits_of.items() if len(s) > 1]
        if leaked:
            raise AssertionError(f"{len(leaked)} {key} values appear in more than one split, e.g. {leaked[:3]}")


def summarize(examples: list[dict], excluded: Counter, fractions, seed) -> dict:
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
        "grouping": "session (back-to-back ship recordings; consecutively numbered noise files per card)",
        "label_rule": "vessel = ship clip (all_far_weak kept, flagged); no_vessel = 3 s windows of background files",
        "excluded_files": dict(excluded),
        "per_split": per_split,
    }


def run_split(inventory_path: Path, out_dir: Path, fractions: tuple[float, ...], seed: int) -> dict:
    with inventory_path.open(newline="", encoding="utf-8") as fh:
        inventory = list(csv.DictReader(fh))
    examples, excluded = build_examples(inventory)
    group_split = make_split(examples, fractions, seed)
    for e in examples:
        e["split"] = group_split[e["group"]]
    check_no_leakage(examples)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "split_manifest.csv"
    examples.sort(key=lambda e: (e["clip_id"], e["offset_s"]))
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for e in examples:
            writer.writerow({f: e[f] for f in MANIFEST_FIELDS})
    with manifest.open("rb") as fh:
        digest = hashlib.file_digest(fh, "sha256").hexdigest()
    (out_dir / "split_manifest.csv.sha256").write_text(f"{digest}  split_manifest.csv\n", encoding="utf-8")

    summary = summarize(examples, excluded, fractions, seed)
    with inventory_path.open("rb") as fh:
        summary["inventory_sha256"] = hashlib.file_digest(fh, "sha256").hexdigest()
    summary["manifest_sha256"] = digest
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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
              f"{info['no_vessel']:6} no-vessel ({info['share_of_no_vessel']:.1%}), {len(info['groups'])} groups")
    print(f"Excluded files: {summary['excluded_files']}")
    print(f"Manifest SHA-256: {summary['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
