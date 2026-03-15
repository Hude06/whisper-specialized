import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from datasets import Audio, load_dataset

from config import (
    DEFAULT_LANGUAGES,
    DEFAULT_MAX_SECONDS,
    DEFAULT_MIN_SECONDS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SAMPLES_ROOT,
    DEFAULT_SPLIT,
    DOWNLOAD_METADATA_FILENAME,
)


def _duration_seconds(row: dict, sampling_rate: int) -> float:
    return row["num_samples"] / float(sampling_rate)


def _manifest_path(language: str, split: str, samples_root: str) -> Path:
    return Path(samples_root) / language / split / "manifest.csv"


def _metadata_path(language: str, split: str, samples_root: str) -> Path:
    return Path(samples_root) / language / split / DOWNLOAD_METADATA_FILENAME


def _expected_metadata(
    language: str,
    split: str,
    sample_count: int,
    min_seconds: float,
    max_seconds: float,
    seed: int,
) -> Dict[str, object]:
    return {
        "dataset": "google/fleurs",
        "language": language,
        "split": split,
        "sample_count": sample_count,
        "min_seconds": min_seconds,
        "max_seconds": max_seconds,
        "seed": seed,
    }


def _read_manifest_rows(manifest_path: Path) -> List[Dict[str, str]]:
    with open(manifest_path, newline="", encoding="utf-8") as manifest_file:
        return list(csv.DictReader(manifest_file))


def _legacy_manifest_matches(
    manifest_path: Path,
    sample_count: int,
    min_seconds: float,
    max_seconds: float,
) -> bool:
    rows = _read_manifest_rows(manifest_path)
    if len(rows) != sample_count:
        return False
    for row in rows:
        duration_sec = float(row["duration_sec"])
        if duration_sec < min_seconds or duration_sec > max_seconds:
            return False
    return True


def _existing_download_matches(
    language: str,
    split: str,
    sample_count: int,
    min_seconds: float,
    max_seconds: float,
    seed: int,
    samples_root: str,
) -> bool:
    manifest_path = _manifest_path(language, split, samples_root)
    metadata_path = _metadata_path(language, split, samples_root)
    if not manifest_path.exists():
        return False

    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        if metadata != _expected_metadata(
            language=language,
            split=split,
            sample_count=sample_count,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            seed=seed,
        ):
            return False
        rows = _read_manifest_rows(manifest_path)
        return len(rows) == sample_count

    return _legacy_manifest_matches(
        manifest_path=manifest_path,
        sample_count=sample_count,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
    )


def _clear_existing_audio_files(target_dir: Path) -> None:
    for wav_file in target_dir.glob("*.wav"):
        wav_file.unlink()


def download_fleurs_samples(
    language: str,
    split: str = DEFAULT_SPLIT,
    min_seconds: float = DEFAULT_MIN_SECONDS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    samples_root: str = DEFAULT_SAMPLES_ROOT,
    seed: int = DEFAULT_RANDOM_SEED,
    force_redownload: bool = False,
) -> str:
    target_dir = Path(samples_root) / language / split
    manifest_path = _manifest_path(language, split, samples_root)
    metadata_path = _metadata_path(language, split, samples_root)

    if not force_redownload and _existing_download_matches(
        language=language,
        split=split,
        sample_count=sample_count,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        seed=seed,
        samples_root=samples_root,
    ):
        print(f"Using existing samples for {language} {split} at {manifest_path}")
        return str(manifest_path)

    print(
        f"Preparing samples for {language} {split} "
        f"({sample_count} clips, {min_seconds}-{max_seconds}s)..."
    )

    try:
        dataset = load_dataset(
            "google/fleurs",
            language,
            split=split,
            trust_remote_code=True,
        )
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" in str(exc):
            raise RuntimeError(
                "google/fleurs currently requires a pre-4.x `datasets` package. "
                "Install compatible dependencies with "
                "`python3 -m pip install --user 'datasets<4'`."
            ) from exc
        raise
    except ValueError as exc:
        if "trust_remote_code=True" in str(exc):
            raise RuntimeError(
                "google/fleurs requires remote dataset code. Make sure you are "
                "running the updated scripts from this directory."
            ) from exc
        raise

    sampling_rate = dataset.features["audio"].sampling_rate or 16000
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate, decode=False))
    dataset = dataset.filter(
        lambda row: min_seconds <= _duration_seconds(row, sampling_rate) <= max_seconds
    )

    if len(dataset) < sample_count:
        raise ValueError(
            f"Only found {len(dataset)} clips in {language} {split} between "
            f"{min_seconds} and {max_seconds} seconds."
        )

    dataset = dataset.shuffle(seed=seed).select(range(sample_count))

    target_dir.mkdir(parents=True, exist_ok=True)
    _clear_existing_audio_files(target_dir)

    with open(manifest_path, "w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(
            [
                "sample_index",
                "fleurs_id",
                "duration_sec",
                "audio_path",
                "transcription",
                "raw_transcription",
                "language",
            ]
        )

        for sample_index, row in enumerate(dataset):
            audio_filename = f"{sample_index:03d}_{row['id']}.wav"
            audio_path = target_dir / audio_filename
            audio_bytes = row["audio"]["bytes"]
            if not audio_bytes:
                raise RuntimeError(
                    f"Missing embedded audio bytes for sample {row['id']}."
                )

            with open(audio_path, "wb") as audio_file:
                audio_file.write(audio_bytes)

            writer.writerow(
                [
                    sample_index,
                    row["id"],
                    _duration_seconds(row, sampling_rate),
                    audio_filename,
                    row["transcription"],
                    row["raw_transcription"],
                    row["language"],
                ]
            )

    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(
            _expected_metadata(
                language=language,
                split=split,
                sample_count=sample_count,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
                seed=seed,
            ),
            metadata_file,
            indent=2,
        )

    print(f"Wrote manifest to {manifest_path}")
    return str(manifest_path)


def download_fleurs_samples_for_languages(
    languages: Iterable[str],
    split: str = DEFAULT_SPLIT,
    min_seconds: float = DEFAULT_MIN_SECONDS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    samples_root: str = DEFAULT_SAMPLES_ROOT,
    seed: int = DEFAULT_RANDOM_SEED,
    force_redownload: bool = False,
) -> Dict[str, str]:
    results: Dict[str, str] = {}
    for language in languages:
        try:
            results[language] = download_fleurs_samples(
                language=language,
                split=split,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
                sample_count=sample_count,
                samples_root=samples_root,
                seed=seed,
                force_redownload=force_redownload,
            )
        except ValueError as exc:
            print(f"Skipping {language}: {exc}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download FLEURS samples that match the configured duration range."
    )
    parser.add_argument(
        "--language",
        action="append",
        dest="languages",
        help="Language code to download. Repeat to download multiple languages.",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Dataset split to download. Default: {DEFAULT_SPLIT}.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help=f"Clips to download per language. Default: {DEFAULT_SAMPLE_COUNT}.",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=DEFAULT_MIN_SECONDS,
        help=f"Minimum clip length. Default: {DEFAULT_MIN_SECONDS}.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_MAX_SECONDS,
        help=f"Maximum clip length. Default: {DEFAULT_MAX_SECONDS}.",
    )
    parser.add_argument(
        "--samples-root",
        default=DEFAULT_SAMPLES_ROOT,
        help=f"Directory for downloaded samples. Default: {DEFAULT_SAMPLES_ROOT}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed used for sample selection. Default: {DEFAULT_RANDOM_SEED}.",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Redownload samples even if the existing manifest matches the request.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    languages = args.languages or list(DEFAULT_LANGUAGES)
    if args.min_seconds > args.max_seconds:
        raise ValueError("--min-seconds cannot be greater than --max-seconds.")
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be greater than zero.")

    download_fleurs_samples_for_languages(
        languages=languages,
        split=args.split,
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        sample_count=args.sample_count,
        samples_root=args.samples_root,
        seed=args.seed,
        force_redownload=args.force_redownload,
    )


if __name__ == "__main__":
    main()
