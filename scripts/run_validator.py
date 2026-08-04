from pathlib import Path

from archaeosort_dataset_builder.validator.dataset_validator import validate_dataset
from archaeosort_dataset_builder.validator.report_generator import save_report

dataset = Path("data/staging/archaeomind_kaggle/ArchaeoMind_dataset")

issues = validate_dataset(dataset)

output = Path("outputs/validation/validation_report.json")

save_report(issues, output)

print("=" * 60)
print("ARCHAEOSORT DATASET VALIDATOR")
print("=" * 60)
print(f"Issues found: {len(issues)}")
print(f"Report: {output.resolve()}")
print("=" * 60)
