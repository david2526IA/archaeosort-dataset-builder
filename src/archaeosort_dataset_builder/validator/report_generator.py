import json
from pathlib import Path


def save_report(
    issues: list[str],
    output_file: Path,
) -> None:

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "total_issues": len(issues),
        "issues": issues,
    }

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )
