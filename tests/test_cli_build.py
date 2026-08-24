from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from archaeosort_dataset_builder.cli import app
from archaeosort_dataset_builder.config.settings import settings

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_global_settings():
    # `settings` is a module-level singleton shared by the whole CLI (see
    # config/settings.py). Without resetting it, a --dataset passed in one
    # test/invocation leaks into the next one via settings.dataset, since
    # the top-level callback only *sets* it when --dataset is given and
    # never clears it. This is pre-existing behavior, not introduced by the
    # build command; flagged for a future hardening pass (see ROADMAP).
    settings.dataset = None
    yield
    settings.dataset = None


def _make_source_dataset(root: Path) -> Path:
    source = root / "source"
    counts = {"natural_rock": 20, "stone_artefact": 14}
    for class_name, count in counts.items():
        class_dir = source / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            (class_dir / f"img_{index:02d}.jpg").write_bytes(b"fake-image-bytes")
    return source


def test_build_help_lists_expected_options():
    result = runner.invoke(app, ["build", "--help"])
    assert result.exit_code == 0
    for flag in ("--output", "--train", "--val", "--test", "--seed", "--copy", "--force"):
        assert flag in result.stdout


def test_cli_build_invokes_engine_and_writes_outputs(tmp_path):
    source = _make_source_dataset(tmp_path)
    output = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "--dataset",
            str(source),
            "build",
            "--output",
            str(output),
            "--seed",
            "7",
            "--copy",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (output / "split_report.json").exists()
    assert (output / "provenance.json").exists()
    assert (output / "manifest.json").exists()

    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["seed"] == 7


def test_cli_build_requires_dataset_option():
    # No --dataset was passed at the top-level callback, so require_dataset()
    # should raise and the CLI should exit non-zero instead of crashing silently.
    result = runner.invoke(app, ["build", "--output", "/tmp/should-not-be-created"])
    assert result.exit_code != 0
