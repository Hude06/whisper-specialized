import argparse

from download import WHISPER_MODEL_NAMES, download_all_whisper_models, download_whisper_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Whisper.cpp GGML model files."
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model name to download. Repeat to download multiple models.",
    )
    parser.add_argument(
        "--out-dir",
        default="whisper_models",
        help="Directory where GGML model files should be stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = args.models

    if not models:
        download_all_whisper_models(out_dir=args.out_dir)
        return

    for model_name in models:
        download_whisper_model(model_name, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
