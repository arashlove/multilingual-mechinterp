"""Tests for culture MCQ loader."""

from pathlib import Path

from multilingual_mechinterp.data import culture_prompt_pairs, load_culture_questions


def test_load_culture_questions_if_present():
    path = Path(__file__).resolve().parents[1] / "data" / "all200questions_persianMiddleEastCulture.json"
    if not path.exists():
        return
    items = load_culture_questions(path, limit=3)
    assert len(items) == 3
    assert "english" in items[0].languages
    assert "persian" in items[0].languages
    prompt = items[0].mcq_prompt("english")
    assert "Answer:" in prompt
    pairs = culture_prompt_pairs(items, limit=2)
    assert len(pairs) == 2
    assert "source_prompt" in pairs[0]
