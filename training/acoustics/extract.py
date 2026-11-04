"""Hash the downloaded QiandaoEar22 archives and extract them into one tree.

    .venv/Scripts/python.exe extract.py

Each zip in data/archives/ is hashed (SHA-256) and extracted into
data/qiandaoear22_full/<folder>/ following qiandao.ARCHIVE_FOLDERS. Zip
members are CRC-checked as they are read. Re-running is safe: a file that is
already there is compared by size and CRC-32 and left alone; a different file
with the same name is reported as a conflict and never overwritten.

The archive record (results/audit/archives.csv) is what proves which download
the audit and splits were built from. Keep the zips unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
import zipfile
import zlib
from pathlib import Path

import qiandao

HERE = Path(__file__).resolve().parent
CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def crc32_file(path: Path) -> int:
    crc = 0
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            crc = zlib.crc32(chunk, crc)
    return crc


def member_target(folder: str, member: str) -> str:
    """Path of a zip member under the data root.

    Most zips wrap their files in a folder named like the last part of
    `folder` ("target/x.wav" -> "KaiYuan/target/x.wav"); the SpeedBoat/other
    parts have no wrapper, and the noise zips keep their card folder
    ("DATA0001/DATA0032.WAV").
    """
    parts = member.replace("\\", "/").split("/")
    if len(parts) > 1 and parts[0].lower() == folder.rsplit("/", 1)[-1].lower():
        parts = parts[1:]
    if any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"unsafe member path: {member!r}")
    return f"{folder}/{'/'.join(parts)}"


def extract_archive(archive: Path, out_root: Path) -> dict:
    folder = qiandao.archive_folder(archive.name)
    if folder is None:
        raise ValueError(f"no folder rule for {archive.name}; add one to qiandao.ARCHIVE_FOLDERS")
    row = {
        "archive": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
        "folder": folder,
        "members": 0,
        "extracted": 0,
        "already_present": 0,
        "conflicts": 0,
    }
    conflicts = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            row["members"] += 1
            target = out_root / member_target(folder, info.filename)
            if target.exists():
                if target.stat().st_size == info.file_size and crc32_file(target) == info.CRC:
                    row["already_present"] += 1
                else:
                    row["conflicts"] += 1
                    conflicts.append(target.relative_to(out_root).as_posix())
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            with zf.open(info) as src, partial.open("wb") as dst:  # zipfile checks the CRC at EOF
                shutil.copyfileobj(src, dst, CHUNK)
            partial.replace(target)
            row["extracted"] += 1
    row["conflict_examples"] = ";".join(conflicts[:5])
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archives", type=Path, default=HERE / "data" / "archives")
    parser.add_argument("--out", type=Path, default=HERE / "data" / "qiandaoear22_full")
    parser.add_argument("--record", type=Path, default=HERE / "results" / "audit" / "archives.csv")
    args = parser.parse_args()

    archives = sorted(args.archives.glob("*.zip"), key=lambda p: p.name.lower())
    if not archives:
        print(f"No zip files in {args.archives}", file=sys.stderr)
        return 1
    rows = []
    for archive in archives:
        row = extract_archive(archive, args.out)
        rows.append(row)
        print(
            f"{row['archive']}: {row['members']} files, {row['extracted']} extracted, "
            f"{row['already_present']} already present, {row['conflicts']} conflicts"
        )
    args.record.parent.mkdir(parents=True, exist_ok=True)
    with args.record.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Archive record: {args.record}")
    return 1 if any(r["conflicts"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
