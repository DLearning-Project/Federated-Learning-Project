"""Name parsing, extraction, audit and split rules, on tiny generated files.

Run from training/acoustics: .venv/Scripts/python.exe -m pytest
"""

import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import audit
import extract
import qiandao
import split

RATE = 8000


def write_wav(path: Path, seconds: float, seed: int, subtype: str = "PCM_32") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    noise = 0.1 * np.random.default_rng(seed).standard_normal(int(RATE * seconds))
    sf.write(str(path), noise, RATE, subtype=subtype)


def ship(ts: str, idx: int, sources: str = "SpeedBoat_N_S&QianDao_M_M", copy: bool = False) -> str:
    multi = "M" if "&" in sources else "S"
    return f"{ts}_{multi}_{sources}_0_label__{int(multi == 'M')}__{idx}{qiandao.COPY_SUFFIX if copy else ''}.wav"


# --- qiandao: names and sessions ------------------------------------------------------


def test_parse_ship_stem_fields_and_copy_suffix():
    stem = Path(ship("20220628101217", 5, "QianDao_N_S&QianDao_F_W", copy=True)).stem
    fields = qiandao.parse_ship_stem(stem)
    assert fields["recording_ts"] == "20220628101217"
    assert fields["multi"] == "M" and fields["k"] == 1 and fields["idx"] == 5
    assert [s["name"] for s in fields["sources"]] == ["QianDao", "QianDao"]
    assert fields["all_far_weak"] is False and fields["non_vessel_only"] is False


def test_flags_far_weak_and_non_vessel_and_bare_sources():
    weak = qiandao.parse_ship_stem("20220628152817_M_QianDao_F_W&KaiYuan_F_W_0_label__1__57")
    assert weak["all_far_weak"] is True
    car = qiandao.parse_ship_stem("20220625121837_S_Car_F_W_0_label__0__73")
    assert car["non_vessel_only"] is True
    mixed = qiandao.parse_ship_stem("20220624163955_M_UUV_M_M&ArtificialSignals_0_label__1__81")
    assert mixed["non_vessel_only"] is False
    assert mixed["unparsed_sources"] == ["ArtificialSignals"]


def test_non_ship_names_are_not_parsed():
    assert qiandao.parse_ship_stem("DATA0033") is None
    assert qiandao.parse_noise_stem("DATA0033") == 33
    assert qiandao.parse_noise_stem("notes") is None


def test_ship_sessions_chain_back_to_back_recordings():
    sessions = qiandao.ship_sessions(["20220628100715", "20220628101217", "20220628101717", "20220628110227"])
    assert sessions == {
        "20220628100715": "20220628100715",
        "20220628101217": "20220628100715",  # 302 s later: same session
        "20220628101717": "20220628100715",
        "20220628110227": "20220628110227",  # 45 min gap: new session
    }


def test_noise_sessions_are_per_card_runs():
    sessions = qiandao.noise_sessions([("DATA0001", 32), ("DATA0001", 33), ("DATA0001", 39), ("DATA0002", 33)])
    assert sessions[("DATA0001", 33)] == "DATA0001/DATA0032"
    assert sessions[("DATA0001", 39)] == "DATA0001/DATA0039"
    assert sessions[("DATA0002", 33)] == "DATA0002/DATA0033"  # same number, other card: separate


# --- extract ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "archive, folder",
    [
        ("KaiYuan_other.zip", "KaiYuan/other"),
        ("UUV_target.zip", "UUV/target"),
        ("Speedboat_Other_9.zip", "SpeedBoat/other"),
        ("NoiseTarget_Other_DATA0002.zip", "noise_target/noise"),
        ("NoiseTarget_target.zip", "noise_target/target"),
        ("something_else.zip", None),
    ],
)
def test_archive_folder(archive, folder):
    assert qiandao.archive_folder(archive) == folder


def test_member_target_strips_wrapper_and_keeps_card():
    assert extract.member_target("KaiYuan/target", "target/a.wav") == "KaiYuan/target/a.wav"
    assert extract.member_target("SpeedBoat/other", "a.wav") == "SpeedBoat/other/a.wav"
    assert extract.member_target("noise_target/noise", "DATA0001/DATA0032.WAV") == "noise_target/noise/DATA0001/DATA0032.WAV"
    with pytest.raises(ValueError):
        extract.member_target("UUV/other", "other/../../evil.wav")


def test_extract_is_idempotent_and_reports_conflicts(tmp_path):
    src = tmp_path / "src.wav"
    write_wav(src, 0.1, seed=1)
    archive = tmp_path / "archives" / "UUV_other.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(src, "other/a.wav")
    out = tmp_path / "full"
    first = extract.extract_archive(archive, out)
    assert (first["extracted"], first["already_present"]) == (1, 0)
    assert (out / "UUV/other/a.wav").read_bytes() == src.read_bytes()
    again = extract.extract_archive(archive, out)
    assert (again["extracted"], again["already_present"], again["conflicts"]) == (0, 1, 0)
    write_wav(out / "UUV/other/a.wav", 0.1, seed=2)  # same name, different audio
    clash = extract.extract_archive(archive, out)
    assert clash["conflicts"] == 1


# --- audit and split on a small fake dataset ---------------------------------------------------


@pytest.fixture
def full_tree(tmp_path: Path) -> Path:
    """Ship clips in three sessions (one also stored as a copy elsewhere), noise on two cards."""
    root = tmp_path / "full"
    sessions = ["20220624101843", "20220626100227", "20220628094713", "20220628160323"]
    seed = 0
    for s_i, ts in enumerate(sessions):
        for idx in range(1, 4 + s_i):
            seed += 1
            write_wav(root / "SpeedBoat/target" / ship(ts, idx), 3.0, seed)
    # The same clip, copied into another folder under a "副本" name.
    original = root / "SpeedBoat/target" / ship(sessions[0], 1)
    copy = root / "KaiYuan/other" / ship(sessions[0], 1, copy=True)
    copy.parent.mkdir(parents=True)
    copy.write_bytes(original.read_bytes())
    # A clip whose only source is not a vessel.
    write_wav(root / "SpeedBoat/other" / ship("20220625121837", 73, "Car_F_W"), 3.0, 99)
    # Noise: card DATA0001 has two runs; DATA0002 repeats one file of DATA0001 byte for byte.
    for card, num in [("DATA0001", 32), ("DATA0001", 33), ("DATA0001", 125), ("DATA0002", 145)]:
        write_wav(root / f"noise_target/noise/{card}/DATA{num:04d}.WAV", 9.5, num, subtype="PCM_24")
    dup = root / "noise_target/noise/DATA0002/DATA0032.WAV"
    dup.write_bytes((root / "noise_target/noise/DATA0001/DATA0032.WAV").read_bytes())
    # An unreadable file.
    (root / "UUV/other").mkdir(parents=True)
    (root / "UUV/other" / ship("20220628094713", 50)).write_bytes(b"not a wav")
    return root


def test_audit_inventory_and_summary(full_tree, tmp_path):
    out = tmp_path / "audit"
    summary = audit.run_audit(full_tree, out, tmp_path / "missing_archives.csv")
    with (out / "inventory.csv").open(newline="", encoding="utf-8") as fh:
        rows = {r["path"]: r for r in csv.DictReader(fh)}

    copy_path = f"KaiYuan/other/{ship('20220624101843', 1, copy=True)}"
    assert rows[copy_path]["canonical"] == "False"
    assert rows[copy_path]["duplicate_of"] == f"SpeedBoat/target/{ship('20220624101843', 1)}"
    assert rows[copy_path]["recording_ts"] == "20220624101843"
    assert summary["copies"]["identical_to_original"] == 1

    assert summary["files"]["unreadable"] == [f"UUV/other/{ship('20220628094713', 50)}"]
    assert summary["noise"]["duplicate_files"] == ["noise_target/noise/DATA0002/DATA0032.WAV"]
    assert summary["ship"]["sessions"] == 5  # 4 sessions + the Car clip's recording
    assert summary["ship"]["canonical_clips"] == 3 + 4 + 5 + 6 + 1  # the copy and the unreadable file are not counted
    assert summary["ship"]["non_vessel_only"] == 1
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["inventory_sha256"] == summary["inventory_sha256"]


def test_split_keeps_groups_whole_and_is_reproducible(full_tree, tmp_path):
    audit.run_audit(full_tree, tmp_path / "audit", tmp_path / "none.csv")
    inventory = tmp_path / "audit" / "inventory.csv"
    first = split.run_split(inventory, tmp_path / "s1", (0.5, 0.25, 0.25), seed=7)
    second = split.run_split(inventory, tmp_path / "s2", (0.5, 0.25, 0.25), seed=7)
    assert first["manifest_sha256"] == second["manifest_sha256"]

    with (tmp_path / "s1" / "split_manifest.csv").open(newline="", encoding="utf-8") as fh:
        manifest = list(csv.DictReader(fh))
    split.check_no_leakage(manifest)  # raises if a group, recording or file spans two splits
    assert first["excluded_files"] == {"unreadable": 1, "non_vessel_only": 1}
    assert not any("Car" in r["sources"] for r in manifest)
    assert not any("KaiYuan/other" in r["clip_id"] for r in manifest)  # the copy is not used
    noise = [r for r in manifest if r["label"] == "0"]
    assert len(noise) == 4 * 3  # 4 distinct 9.5 s files -> three 3 s windows each
    for s in split.SPLITS:
        assert first["per_split"][s]["vessel"] > 0 and first["per_split"][s]["no_vessel"] > 0


def test_leakage_check_catches_a_shared_recording():
    rows = [
        {"group": "a", "recording": "r1", "sha256": "x", "split": "train"},
        {"group": "b", "recording": "r1", "sha256": "y", "split": "test"},
    ]
    with pytest.raises(AssertionError, match="recording"):
        split.check_no_leakage(rows)


def _ship_examples(session: str, starts: list[str]) -> list[dict]:
    return [{"label": 1, "group": f"ship/{session}", "recording": ts, "closest_distance": "N"} for ts in starts]


def recording_times(first: str, count: int) -> list[str]:
    from datetime import datetime, timedelta
    start = datetime.strptime(first, "%Y%m%d%H%M%S")
    return [(start + timedelta(seconds=301 * i)).strftime("%Y%m%d%H%M%S") for i in range(count)]


def test_make_blocks_cuts_long_sessions_with_gaps():
    times = recording_times("20220628093209", 14)
    examples = _ship_examples("20220628093209", times)
    gaps = split.make_blocks(examples, block_recordings=6)
    groups = [e["group"] for e in examples]
    assert groups[:6] == ["ship/20220628093209#0"] * 6
    assert groups[6] == "ship/20220628093209#gap0"
    assert groups[7:] == ["ship/20220628093209#1"] * 7  # a single leftover joins the last block
    assert gaps == {"ship/20220628093209#gap0": ("ship/20220628093209#0", "ship/20220628093209#1")}


def test_short_sessions_stay_whole():
    examples = _ship_examples("20220624101843", recording_times("20220624101843", 6))
    assert split.make_blocks(examples, block_recordings=6) == {}
    assert {e["group"] for e in examples} == {"ship/20220624101843"}


def test_back_to_back_recordings_in_different_splits_are_refused():
    a, b, c = recording_times("20220628093209", 3)
    ok = [{"label": 1, "recording": a, "group": "x", "sha256": "1", "split": "train"},
          {"label": 1, "recording": c, "group": "y", "sha256": "3", "split": "test"}]  # b dropped as a gap
    split.check_no_leakage(ok)
    bad = ok[:1] + [{"label": 1, "recording": b, "group": "y", "sha256": "2", "split": "test"}]
    with pytest.raises(AssertionError, match="back-to-back"):
        split.check_no_leakage(bad)


def test_assign_groups_balances_distance_mix():
    # Two near-heavy and two far-heavy groups: each split pair should get one of each kind.
    counts = {
        "n1": Counter(N=90, F=10), "n2": Counter(N=90, F=10),
        "f1": Counter(N=10, F=90), "f2": Counter(N=10, F=90),
    }
    result = split.assign_groups(counts, (0.5, 0.25, 0.25), seed=1, restarts=20)
    train = [g for g, s in result.items() if s == "train"]
    assert sorted(g[0] for g in train) == ["f", "n"]


def test_assign_groups_needs_three_groups():
    with pytest.raises(ValueError):
        split.assign_groups({"a": Counter(N=5), "b": Counter(N=3)}, split.DEFAULT_FRACTIONS, seed=0)
