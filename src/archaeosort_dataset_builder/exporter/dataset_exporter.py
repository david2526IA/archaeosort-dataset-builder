from pathlib import Path
import shutil
import argparse


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    shutil.copytree(input_dir, output_dir)

    print("=" * 60)
    print("ARCHAEOSORT DATASET EXPORTER")
    print("=" * 60)
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print("Export completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
