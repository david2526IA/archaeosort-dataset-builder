"""CLI-level tests for 'archaeosort serve' that do not require torch or
uvicorn actually binding a port: --help only exercises Typer's own option
parsing/help rendering. An actual server start is verified manually against
the project's real environment, not here.
"""

from __future__ import annotations

import os

from typer.testing import CliRunner

from archaeosort_dataset_builder.cli import app

os.environ["COLUMNS"] = "200"

runner = CliRunner()


def test_serve_help_lists_expected_options():
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    rendered = " ".join(result.stdout.split())
    for flag in (
        "--model",
        "--embeddings-dir",
        "--host",
        "--port",
        "--classes",
        "--device",
        "--reload",
        "--zero-shot",
        "--zero-shot-model",
        "--zero-shot-labels",
        "--prototypes",
    ):
        assert flag in rendered


def test_serve_requires_model_or_embeddings_dir_or_zero_shot():
    result = runner.invoke(app, ["serve"])
    assert result.exit_code != 0


def test_build_prototypes_help_lists_expected_options():
    result = runner.invoke(app, ["build-prototypes", "--help"])
    assert result.exit_code == 0
    rendered = " ".join(result.stdout.split())
    for flag in (
        "--output",
        "--references-dir",
        "--natural-rock-dir",
        "--stone-artefact-dir",
        "--per-class",
        "--force",
        "--seed",
    ):
        assert flag in rendered


def test_build_prototypes_requires_output():
    result = runner.invoke(app, ["build-prototypes"])
    assert result.exit_code != 0
