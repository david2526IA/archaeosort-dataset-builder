"""CLI-level test for 'archaeosort train-baseline' that does not require
torch: --help only exercises Typer's own option parsing/help rendering, not
the lazy-imported train_baseline() function itself. An actual training run
through the CLI needs torch/torchvision/scikit-learn and a real dataset —
that must be verified with the project's real environment, not here.
"""

from __future__ import annotations

import os

from typer.testing import CliRunner

from archaeosort_dataset_builder.cli import app

# Rich (Typer's help renderer) wraps long option names to the terminal
# width; a narrow default width can split a flag like "--freeze-backbone"
# across lines and break a naive substring check. Force a wide terminal for
# this test so --help renders each option on one line.
os.environ["COLUMNS"] = "200"

runner = CliRunner()


def test_train_baseline_help_lists_expected_options():
    result = runner.invoke(app, ["train-baseline", "--help"])
    assert result.exit_code == 0
    rendered = " ".join(result.stdout.split())
    for flag in (
        "--data",
        "--output",
        "--epochs",
        "--batch-size",
        "--lr",
        "--freeze-backbone",
        "--no-freeze-backbone",
        "--patience",
        "--seed",
    ):
        assert flag in rendered


def test_train_baseline_requires_data_and_output():
    result = runner.invoke(app, ["train-baseline"])
    assert result.exit_code != 0
