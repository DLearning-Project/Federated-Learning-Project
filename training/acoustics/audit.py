"""Inventory every QiandaoEar22 file: format, hash, parsed name, duplicates.

    .venv/Scripts/python.exe audit.py

Reads data/qiandaoear22_full/ (made by extract.py) and writes
results/audit/inventory.csv (one row per file) and results/audit/summary.json.
No audio is copied into the results, only names, formats and hashes.

The same 3-second clip is stored in several folders (each folder is one
"target vs. other" task), so every file is hashed and each distinct content
gets one canonical file. split.py uses only canonical files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

import qiandao
from envinfo import git_info, package_versions

HERE = Path(__file__).resolve().parent

# From QiandaoEar22_documentations.pdf, for comparison only.
DOCUMENTED = {
    "ship_records": 10611,
    "ship_recordings": 143,
    "noise_records_3s": 25900,
    "noise_hours": 21 + 58 / 60,
    "sample_rate_hz": 52734,
    "bits": 16,
}

FIELDS = [
    "path", "folder", "subset", "kind", "bytes", "sha256", "readable", "error",
    "sample_rate", "channels", "subtype", "frames", "duration_s",
    "is_copy", "base_name", "recording_ts", "session", "multi", "k", "n", "idx",
    "sources", "n_sources", "all_far_weak", "non_vessel_only", "unparsed_sources",
    "card", "noise_num", "canonical", "duplicate_of",
]


def sha256_file(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def read_format(path: Path) -> dict:
    """Header facts, plus a read of the last frames to catch truncated files."""
    try:
        info = sf.info(str(path))
        with sf.SoundFile(str(path)) as fh:
            if info.frames:
                fh.seek(max(0, info.frames - 16))
                fh.read(16)
    except Exception as exc:  # any decode failure is recorded, not raised
        return {"readable": False, "error": repr(exc)}
    return {
        "readable": True,
        "error": "",
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "subtype": info.subtype,
        "frames": info.frames,
        "duration_s": round(info.frames / info.samplerate, 4),
    }


def describe_file(root: Path, path: Path) -> dict:
    rel = path.relative_to(root).as_posix()
    parts = rel.split("/")
    base_stem, is_copy = qiandao.split_copy(path.stem)
    row: dict = {
        "path": rel,
        "folder": parts[0],
        "subset": parts[1] if len(parts) > 2 else "",
        "bytes": path.stat().st_size,
        "is_copy": is_copy,
        "base_name": base_stem + path.suffix,
    }
    ship = qiandao.parse_ship_stem(path.stem)
    noise_num = qiandao.parse_noise_stem(path.stem)
    if ship:
        row.update(
            kind="ship",
            recording_ts=ship["recording_ts"],
            multi=ship["multi"],
            k=ship["k"],
            n=ship["n"],
            idx=ship["idx"],
            sources="&".join(f"{s['name']}:{s['distance'] or ''}:{s['audibility'] or ''}" for s in ship["sources"]),
            n_sources=len(ship["sources"]),
            all_far_weak=ship["all_far_weak"],
            non_vessel_only=ship["non_vessel_only"],
            unparsed_sources="&".join(ship["unparsed_sources"]),
        )
    elif noise_num is not None:
        row.update(kind="noise", card=parts[-2], noise_num=noise_num)
    else:
        row["kind"] = "unknown"
    return row


def mark_duplicates(rows: list[dict]) -> None:
    """One canonical file per content hash: prefer a name without the copy suffix."""
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_hash[row["sha256"]].append(row)
    for group in by_hash.values():
        group.sort(key=lambda r: (r["is_copy"], r["path"]))
        keep = group[0]
        keep["canonical"], keep["duplicate_of"] = True, ""
        for row in group[1:]:
            row["canonical"], row["duplicate_of"] = False, keep["path"]


def add_sessions(rows: list[dict]) -> None:
    ship_session = qiandao.ship_sessions([r["recording_ts"] for r in rows if r["kind"] == "ship"])
    noise_session = qiandao.noise_sessions([(r["card"], r["noise_num"]) for r in rows if r["kind"] == "noise"])
    for row in rows:
        if row["kind"] == "ship":
            row["session"] = ship_session[row["recording_ts"]]
        elif row["kind"] == "noise":
            row["session"] = noise_session[(row["card"], row["noise_num"])]


def _shown(path: Path) -> str:
    """Path relative to this folder when possible, so records hold no user names."""
    try:
        return path.resolve().relative_to(HERE).as_posix()
    except ValueError:
        return path.as_posix()


def summarize(rows: list[dict], root: Path, archive_record: Path) -> dict:
    canon = [r for r in rows if r["canonical"]]
    ships = [r for r in canon if r["kind"] == "ship" and r["readable"]]
    noise = [r for r in canon if r["kind"] == "noise" and r["readable"]]

    # Copies: is the copy byte-identical to a file with its original name anywhere?
    hashes_by_name: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if not r["is_copy"]:
            hashes_by_name[r["base_name"]].add(r["sha256"])
    copies = [r for r in rows if r["is_copy"]]
    copy_status = Counter(
        "no_original" if r["base_name"] not in hashes_by_name
        else "identical_to_original" if r["sha256"] in hashes_by_name[r["base_name"]]
        else "differs_from_original"
        for r in copies
    )
    # Same name (copy suffix removed) but different content.
    contents_by_name: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        contents_by_name[r["base_name"]].add(r["sha256"])
    name_conflicts = sorted(n for n, h in contents_by_name.items() if len(h) > 1)

    formats = Counter(
        f"{r['kind']}: {r.get('sample_rate')} Hz, {r.get('channels')} ch, {r.get('subtype')}, {r.get('duration_s')} s"
        for r in canon
    )
    source_names = Counter(s.split(":")[0] for r in ships for s in r["sources"].split("&"))
    noise_seconds = sum(r.get("duration_s") or 0 for r in noise)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_root": _shown(root),
        "archive_record": {
            "path": _shown(archive_record),
            "sha256": sha256_file(archive_record) if archive_record.is_file() else None,
        },
        "git": git_info(HERE),
        "packages": {k: v for k, v in package_versions().items() if k in ("soundfile", "numpy")},
        "files": {
            "total": len(rows),
            "canonical": len(canon),
            "duplicates": len(rows) - len(canon),
            "by_folder": {
                f"{folder}/{subset}": {"files": n, "canonical": sum(1 for r in canon if (r["folder"], r["subset"]) == (folder, subset))}
                for (folder, subset), n in sorted(Counter((r["folder"], r["subset"]) for r in rows).items())
            },
            "unreadable": [r["path"] for r in rows if not r["readable"]],
            "unknown_names": [r["path"] for r in rows if r["kind"] == "unknown"][:50],
            "canonical_formats": dict(formats.most_common()),
        },
        "copies": {
            "files_with_copy_suffix": len(copies),
            **dict(copy_status),
            "names_with_differing_content": len(name_conflicts),
            "examples_of_differing_content": name_conflicts[:10],
        },
        "ship": {
            "canonical_clips": len(ships),
            "hours": round(len(ships) * qiandao.CLIP_SECONDS / 3600, 2),
            "recordings": len({r["recording_ts"] for r in ships}),
            "sessions": len({r["session"] for r in ships}),
            "clips_per_session": dict(sorted(Counter(r["session"] for r in ships).items())),
            "days": dict(sorted(Counter(r["recording_ts"][:8] for r in ships).items())),
            "single_vs_multiple": dict(Counter(r["multi"] for r in ships)),
            "all_far_weak": sum(r["all_far_weak"] for r in ships),
            "non_vessel_only": sum(r["non_vessel_only"] for r in ships),
            "non_vessel_only_sources": dict(Counter(r["sources"] for r in ships if r["non_vessel_only"]).most_common(10)),
            "sources_without_distance": dict(Counter(
                t for r in ships for t in r["unparsed_sources"].split("&") if r["unparsed_sources"]
            ).most_common()),
            "clips_per_source_name": dict(source_names.most_common()),
        },
        "noise": {
            "canonical_files": len(noise),
            "hours": round(noise_seconds / 3600, 2),
            "sessions": dict(sorted(Counter(r["session"] for r in noise).items())),
            "duplicate_files": sorted(r["path"] for r in rows if r["kind"] == "noise" and not r["canonical"]),
        },
        "documented": DOCUMENTED,
    }


def run_audit(root: Path, out_dir: Path, archive_record: Path) -> dict:
    paths = sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"), key=lambda p: p.as_posix())
    rows = []
    for i, path in enumerate(paths, 1):
        row = describe_file(root, path)
        row["sha256"] = sha256_file(path)
        row.update(read_format(path))
        rows.append(row)
        if i % 2000 == 0:
            print(f"  {i}/{len(paths)} files", flush=True)
    mark_duplicates(rows)
    add_sessions(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "inventory.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in FIELDS})
    summary = summarize(rows, root, archive_record)
    summary["inventory_sha256"] = sha256_file(out_dir / "inventory.csv")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=HERE / "data" / "qiandaoear22_full")
    parser.add_argument("--out", type=Path, default=HERE / "results" / "audit")
    parser.add_argument("--archive-record", type=Path, default=HERE / "results" / "audit" / "archives.csv")
    args = parser.parse_args()
    if not args.data_root.is_dir():
        print(f"{args.data_root} not found; run extract.py first", file=sys.stderr)
        return 1
    summary = run_audit(args.data_root, args.out, args.archive_record)
    print(json.dumps({k: summary[k] for k in ("files", "copies")}, indent=2, ensure_ascii=False)[:3000])
    return 1 if summary["files"]["unreadable"] else 0


if __name__ == "__main__":
    sys.exit(main())
