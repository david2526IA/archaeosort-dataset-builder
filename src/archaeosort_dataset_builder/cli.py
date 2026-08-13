import typer

from archaeosort_dataset_builder.analyzer.analyzer import analyze
from archaeosort_dataset_builder.duplicates.duplicates import duplicates
from archaeosort_dataset_builder.verify.verify import verify

app = typer.Typer(no_args_is_help=True, help="ArchaeoSort Dataset Builder")


@app.command()
def verify_cmd():
    verify()


@app.command("analyze")
def analyze_cmd():
    analyze()


@app.command("duplicates")
def duplicates_cmd():
    duplicates()


if __name__ == "__main__":
    app()

