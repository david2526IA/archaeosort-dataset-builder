"""Torch-free tests for api/zero_shot.py: DEFAULT_PROMPTS shape and the
ARCHAEOSORT_ZEROSHOT_LABELS parser. load_zero_shot_model/classify_zero_shot
are torch-dependent and are instead exercised (monkeypatched) via the HTTP
contract tests in test_api_app.py.
"""

from __future__ import annotations

import pytest

from archaeosort_dataset_builder.api.zero_shot import (
    DEFAULT_PROMPTS,
    ZeroShotError,
    parse_label_prompts,
)


def test_default_prompts_cover_all_six_categories():
    assert set(DEFAULT_PROMPTS) == {
        "natural_rock",
        "stone_artefact",
        "natural_bone",
        "bone_artefact",
        "natural_wood",
        "wood_artefact",
    }
    assert all(isinstance(prompt, str) and prompt for prompt in DEFAULT_PROMPTS.values())


def test_parse_label_prompts_parses_semicolon_separated_pairs():
    result = parse_label_prompts("a=a photo of a;b=a photo of b")
    assert result == {"a": "a photo of a", "b": "a photo of b"}


def test_parse_label_prompts_strips_whitespace():
    result = parse_label_prompts(" a = a photo of a ; b = a photo of b ")
    assert result == {"a": "a photo of a", "b": "a photo of b"}


def test_parse_label_prompts_rejects_missing_equals():
    with pytest.raises(ZeroShotError):
        parse_label_prompts("not-a-valid-entry")


def test_parse_label_prompts_rejects_empty_label_or_prompt():
    with pytest.raises(ZeroShotError):
        parse_label_prompts("=a photo of nothing")
    with pytest.raises(ZeroShotError):
        parse_label_prompts("label=")


def test_parse_label_prompts_rejects_blank_input():
    with pytest.raises(ZeroShotError):
        parse_label_prompts("   ;  ")
