import pytest

from archaeosort_dataset_builder.splitter.dataset_splitter import validate_ratios


def test_valid_ratios():
    validate_ratios(0.7, 0.15, 0.15)


def test_ratios_must_sum_one():
    with pytest.raises(ValueError):
        validate_ratios(0.7, 0.2, 0.2)


def test_ratios_cannot_be_negative():
    with pytest.raises(ValueError):
        validate_ratios(0.8, 0.3, -0.1)
