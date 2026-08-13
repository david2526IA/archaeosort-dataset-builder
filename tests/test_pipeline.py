from archaeosort_dataset_builder.pipeline import pipeline


def test_pipeline_executes_all_steps_in_order(monkeypatch):
    executed = []

    names = [
        "verify",
        "analyze",
        "duplicates",
        "scan_quality",
        "resolution",
        "aspect_ratio",
        "class_balance",
        "quality",
        "statistics",
        "report",
    ]

    for name in names:
        monkeypatch.setattr(
            pipeline,
            name,
            lambda name=name: executed.append(name),
        )

    pipeline.run_pipeline()

    assert executed == names
