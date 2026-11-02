"""Checks that paper_facts.json is complete and well formed."""

import json
from collections import Counter
from pathlib import Path

import pytest

FACTS_PATH = Path(__file__).resolve().parents[1] / "app" / "modules" / "language_models" / "data" / "paper_facts.json"
FACTS = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
REQUIRED = ("id", "label", "short", "long", "equation", "reference", "levels")
LEVELS = ("beginner", "student", "researcher")


@pytest.mark.parametrize("term", FACTS["terms"], ids=lambda t: t.get("id", "?"))
def test_term_has_required_fields_and_levels(term):
    for field in REQUIRED:
        assert term.get(field), f"{term.get('id')}: missing {field}"
    for level in LEVELS:
        assert term["levels"].get(level, "").strip(), f"{term['id']}: missing level text {level}"


def test_term_ids_unique():
    ids = [t["id"] for t in FACTS["terms"]]
    assert len(ids) == len(set(ids))


def test_table2_complete_and_unique():
    rows = FACTS["reported_results"]["table2"]["rows"]
    assert len(rows) == 72  # 2 models x 2 client configs x 6 methods x 3 datasets
    keys = {(r["model"], r["client_config"], r["method"], r["dataset"]) for r in rows}
    assert len(keys) == 72
    per_model_config = Counter((r["model"], r["client_config"]) for r in rows)
    assert set(per_model_config.values()) == {18}
    for r in rows:
        assert 0 < r["accuracy_pct"] < 100 and r["efficiency_e-4"] > 0


def test_tables_3_and_4_present_with_configuration():
    t3 = FACTS["reported_results"]["table3"]["rows"]
    t4 = FACTS["reported_results"]["table4"]["rows"]
    assert len(t3) == 7 and len(t4) == 5
    for r in t3 + t4:
        assert r["model"] == "TinyLlama" and r["dataset"] == "Wizard"
    for table in ("table2", "table3", "table4"):
        assert FACTS["reported_results"][table]["reference"]


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def test_glossary_math_markup_is_balanced():
    """Glossary text marks math with $…$ for KaTeX; an odd count breaks rendering."""
    facts = json.loads(FACTS_PATH.read_text())
    for term in facts["terms"]:
        for text in _strings({k: v for k, v in term.items() if k not in ("equation", "equation_plain")}):
            assert text.count("$") % 2 == 0, f"{term['id']}: unbalanced $ in {text!r}"
