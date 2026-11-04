"""Catalog of locally downloaded QiandaoEar22 recordings.

The audio is NOT part of the repository (see training/acoustics/README.md). The
catalog scans a local folder with this layout and works without it:

    <data_dir>/speedboat_target/*.wav     3 s clips, a speedboat is among the vessels
    <data_dir>/speedboat_other/*.wav      3 s clips, ships but no speedboat
    <data_dir>/noise/<group>/DATA####.WAV 5-minute background recordings

A clip is addressed by its path relative to the data folder. Requests never
supply a filesystem path directly: `resolve_clip` rejects anything that is not
a .wav file inside the data folder.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ship clip names: <YYYYMMDDHHMMSS>_<M|S>_<vessel>&<vessel>..._<n>_label__<k>__<i>.wav
# where <vessel> is <Name>_<distance N/M/F>_<audibility S/M/W>. The 14-digit
# timestamp is the start of the original 5-minute recording (meaning inferred
# from file names and times, to be confirmed in the dataset audit).
_SHIP_NAME = re.compile(r"^(?P<ts>\d{14})_(?P<multi>[MS])_(?P<vessels>.+?)_(?P<n>\d+)_label__(?P<label>\d+)__(?P<idx>\d+)$")
_DISTANCE = {"N": "near", "M": "medium", "F": "far"}
_AUDIBILITY = {"S": "strong", "M": "middle", "W": "weak"}

NOISE_WINDOW_OFFSET_S = 60.0  # background files are 5 minutes: skip the first minute
_CATEGORIES = ("speedboat_target", "speedboat_other", "noise")
# Windows names a copied file "<name> - 副本" ("copy"). In QiandaoEar22 most copies
# have no original in the same folder, so a copy is only skipped when it is
# byte-identical to "<name>" next to it.
_COPY_SUFFIX = " - 副本"


@dataclass(frozen=True)
class Clip:
    id: str  # path relative to the data folder, with forward slashes
    path: Path
    category: str  # speedboat_target | speedboat_other | noise
    label: str
    recording_id: str
    offset_s: float


def _vessels(text: str) -> list[str]:
    out = []
    for token in text.split("&"):
        parts = token.rsplit("_", 2)
        if len(parts) == 3 and parts[1] in _DISTANCE and parts[2] in _AUDIBILITY:
            out.append(f"{parts[0]} ({_DISTANCE[parts[1]]}, {_AUDIBILITY[parts[2]]})")
        else:
            out.append(token)
    return out


def describe(category: str, stem: str) -> tuple[str, str]:
    """(human-readable label, recording id) from a category and file stem."""
    stem = stem.removesuffix(_COPY_SUFFIX)
    if category == "noise":
        return f"Background noise, recording {stem}", stem
    match = _SHIP_NAME.match(stem)
    if not match:
        return stem, stem
    vessels = ", ".join(_vessels(match["vessels"]))
    return f"{vessels}", match["ts"]


def _sha256(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def _is_identical_copy(path: Path) -> bool:
    if not path.stem.endswith(_COPY_SUFFIX):
        return False
    original = path.with_name(path.stem.removesuffix(_COPY_SUFFIX) + path.suffix)
    return (
        original.is_file()
        and original.stat().st_size == path.stat().st_size
        and _sha256(original) == _sha256(path)
    )


def _is_wav(path: Path) -> bool:
    return path.suffix.lower() == ".wav" and not _is_identical_copy(path)


def _make_clip(root: Path, path: Path, category: str) -> Clip:
    label, recording_id = describe(category, path.stem)
    offset = NOISE_WINDOW_OFFSET_S if category == "noise" else 0.0
    return Clip(path.relative_to(root).as_posix(), path, category, label, recording_id, offset)


def scan_clips(root: Path, per_category: int = 12) -> list[Clip]:
    """Up to `per_category` clips per category, spread evenly over the sorted files."""
    if not root.is_dir():
        return []
    clips: list[Clip] = []
    for category in _CATEGORIES:
        folder = root / category
        if not folder.is_dir():
            continue
        files = sorted(p for p in folder.rglob("*") if p.is_file() and _is_wav(p))
        if len(files) > per_category:
            picks = np.linspace(0, len(files) - 1, per_category).round().astype(int)
            files = [files[i] for i in sorted(set(picks.tolist()))]
        clips.extend(_make_clip(root, p, category) for p in files)
    return clips


class ClipNotFound(LookupError):
    """The requested clip id is not a usable recording in the data folder."""


def resolve_clip(root: Path, clip_id: str) -> Clip:
    """Map a clip id to a file, refusing anything outside the data folder."""
    if not root.is_dir():
        raise ClipNotFound("The QiandaoEar22 data folder was not found on the server.")
    root_resolved = root.resolve()
    path = (root_resolved / clip_id).resolve()
    category = clip_id.replace("\\", "/").split("/", 1)[0]
    if (
        category not in _CATEGORIES
        or not path.is_relative_to(root_resolved)
        or not path.is_file()
        or not _is_wav(path)
    ):
        raise ClipNotFound("Unknown clip. Choose one from the clip list.")
    return _make_clip(root_resolved, path, category)
