"""What the QiandaoEar22 file names and folders mean.

Shared by extract.py, audit.py and split.py. The rules here were worked out
from the author-hosted OneDrive copy (2026-11-03); see TRAINING.md, Phase 2.

Ship clip names:

    <YYYYMMDDHHMMSS>_<S|M>_<source>&<source>..._<n>_label__<k>__<idx>[ - 副本].wav

- The 14-digit timestamp is the start of a 5-minute recording; recordings
  follow each other with 300-302 s gaps, so consecutive ones form a session.
- <source> is <Name>_<distance N/M/F>_<audibility S/M/W>, or a bare name
  (for example "ArtificialSignals") without distance and audibility.
- <k> repeats the S/M flag (0 = single source, 1 = multiple); it is not a class.
- <idx> numbers the 3-second clips within a recording and source list.
- " - 副本" is the Windows "copy" suffix.

Background noise files are 5-minute recordings named DATA####.WAV inside a
card folder (DATA0001, DATA0002); consecutive numbers are consecutive files.
"""

from __future__ import annotations

import re
from datetime import datetime

COPY_SUFFIX = " - 副本"
CLIP_SECONDS = 3.0
SESSION_GAP_S = 330  # recordings start 300-302 s apart; a longer gap starts a new session

_SHIP_STEM = re.compile(
    r"^(?P<ts>\d{14})_(?P<multi>[MS])_(?P<sources>.+?)_(?P<n>\d+)_label__(?P<k>\d+)__(?P<idx>\d+)$"
)
_NOISE_STEM = re.compile(r"^DATA(?P<num>\d{4})$", re.IGNORECASE)
_DISTANCES = {"N", "M", "F"}
_AUDIBILITIES = {"S", "M", "W"}

# Fig. 3 of the documentation lists these among the "ship targets", but they are
# not watercraft. A clip whose only sources are these is not a vessel example.
NON_VESSEL_SOURCES = {"ArtificialSignals", "Car", "Helicopter"}

# Download archive name -> folder it extracts into (relative to the data root).
# OneDrive could not zip SpeedBoat/other in one piece, so it came in 9 parts.
ARCHIVE_FOLDERS = [
    (re.compile(r"^KaiYuan_(target|other)\.zip$", re.I), "KaiYuan/{0}"),
    (re.compile(r"^UUV_(target|other)\.zip$", re.I), "UUV/{0}"),
    (re.compile(r"^SpeedBoat_target\.zip$", re.I), "SpeedBoat/target"),
    (re.compile(r"^Speedboat_Other_\d+\.zip$", re.I), "SpeedBoat/other"),
    (re.compile(r"^NoiseTarget_target\.zip$", re.I), "noise_target/target"),
    (re.compile(r"^NoiseTarget_Other_(DATA\d{4})\.zip$", re.I), "noise_target/noise"),
]


def archive_folder(archive_name: str) -> str | None:
    """Folder (relative to the data root) that an archive extracts into."""
    for pattern, folder in ARCHIVE_FOLDERS:
        match = pattern.match(archive_name)
        if match:
            return folder.format(*(g.lower() for g in match.groups()))
    return None


def split_copy(stem: str) -> tuple[str, bool]:
    """(stem without the copy suffix, whether it had one)."""
    if stem.endswith(COPY_SUFFIX):
        return stem.removesuffix(COPY_SUFFIX), True
    return stem, False


def parse_sources(text: str) -> list[dict]:
    out = []
    for token in text.split("&"):
        parts = token.rsplit("_", 2)
        if len(parts) == 3 and parts[1] in _DISTANCES and parts[2] in _AUDIBILITIES:
            out.append({"name": parts[0], "distance": parts[1], "audibility": parts[2]})
        else:
            out.append({"name": token, "distance": None, "audibility": None})
    return out


def parse_ship_stem(stem: str) -> dict | None:
    """Fields of a ship clip name, or None if it does not follow the pattern."""
    base, _ = split_copy(stem)
    match = _SHIP_STEM.match(base)
    if not match:
        return None
    sources = parse_sources(match["sources"])
    names = [s["name"] for s in sources]
    return {
        "recording_ts": match["ts"],
        "multi": match["multi"],
        "sources": sources,
        "n": int(match["n"]),
        "k": int(match["k"]),
        "idx": int(match["idx"]),
        # every source is far and weak: the hardest "vessel" examples
        "all_far_weak": all(s["distance"] == "F" and s["audibility"] == "W" for s in sources),
        "non_vessel_only": all(name in NON_VESSEL_SOURCES for name in names),
        "unparsed_sources": [s["name"] for s in sources if s["distance"] is None],
    }


def parse_noise_stem(stem: str) -> int | None:
    match = _NOISE_STEM.match(stem)
    return int(match["num"]) if match else None


def ship_sessions(timestamps: list[str]) -> dict[str, str]:
    """Map each recording timestamp to its session id (the session's first timestamp).

    A session is a run of recordings that start at most SESSION_GAP_S apart:
    the hydrophone recorded continuously, so a vessel heard at the end of one
    recording is still heard at the start of the next.
    """
    out: dict[str, str] = {}
    previous: datetime | None = None
    session = ""
    for ts in sorted(set(timestamps)):
        start = datetime.strptime(ts, "%Y%m%d%H%M%S")
        if previous is None or (start - previous).total_seconds() > SESSION_GAP_S:
            session = ts
        out[ts] = session
        previous = start
    return out


def noise_sessions(files: list[tuple[str, int]]) -> dict[tuple[str, int], str]:
    """Map (card folder, file number) to a session id: runs of consecutive numbers per card."""
    out: dict[tuple[str, int], str] = {}
    for card in sorted({c for c, _ in files}):
        previous = None
        session = ""
        for num in sorted(n for c, n in files if c == card):
            if previous is None or num != previous + 1:
                session = f"{card}/DATA{num:04d}"
            out[(card, num)] = session
            previous = num
    return out
