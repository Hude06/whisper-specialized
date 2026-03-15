from datasets import Audio, load_dataset
import os
import csv
import urllib.request
import urllib.error
import shutil
import time

WHISPER_MODEL_NAMES = (
    "tiny",
    "large",
)

GGML_MODEL_URLS = {
    "tiny": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
    "large": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
}


def _duration_seconds(row: dict, sampling_rate: int) -> float:
    return row["num_samples"] / float(sampling_rate)


def _download_file(url: str, dest_path: str, timeout: int = 60, retries: int = 3):
    """Download a file from URL to destination with progress and retries."""
    print(f"Downloading {url}...")
    print(f"Destination: {dest_path}")

    temp_path = f"{dest_path}.part"
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                total_size = int(response.headers.get("Content-Length", "0"))
                downloaded = 0
                with open(temp_path, "wb") as output_file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output_file.write(chunk)
                        downloaded += len(chunk)
                        percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
                        print(f"\rProgress: {percent:.1f}%", end="", flush=True)
            os.replace(temp_path, dest_path)
            print("\nDownload complete!")
            return
        except (urllib.error.URLError, TimeoutError) as exc:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if attempt == retries:
                raise RuntimeError(
                    f"Failed to download {url} after {retries} attempts: {exc}"
                ) from exc
            wait_seconds = attempt * 2
            print(f"\nDownload attempt {attempt} failed: {exc}. Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)


def download_whisper_model(model_name: str, out_dir: str = "whisper_models"):
    """Download a GGML Whisper model for use with whisper.cpp."""
    if model_name not in WHISPER_MODEL_NAMES:
        supported = ", ".join(WHISPER_MODEL_NAMES)
        raise ValueError(
            f"Unsupported Whisper model '{model_name}'. Supported models: {supported}."
        )

    os.makedirs(out_dir, exist_ok=True)
    model_filename = os.path.basename(GGML_MODEL_URLS[model_name])
    model_path = os.path.join(out_dir, model_filename)

    if os.path.exists(model_path):
        print(f"Model {model_name} already exists at {model_path}")
        return model_path

    url = GGML_MODEL_URLS[model_name]
    _download_file(url, model_path)
    return model_path


def download_whisper_tiny(out_dir: str = "whisper_models"):
    return download_whisper_model("tiny", out_dir=out_dir)


def download_whisper_large(out_dir: str = "whisper_models"):
    return download_whisper_model("large", out_dir=out_dir)


def download_all_whisper_models(out_dir: str = "whisper_models"):
    """Download all supported GGML Whisper models."""
    downloads = {}
    for model_name in WHISPER_MODEL_NAMES:
        model_path = download_whisper_model(model_name, out_dir=out_dir)
        downloads[model_name] = model_path
    return downloads


def cleanup_old_whisper_models(out_dir: str = "whisper_models"):
    """Remove old PyTorch model files and directories."""
    if not os.path.exists(out_dir):
        return

    print(f"Cleaning up old models in {out_dir}...")
    for item in os.listdir(out_dir):
        item_path = os.path.join(out_dir, item)
        # Remove directories (old PyTorch model folders)
        if os.path.isdir(item_path):
            print(f"Removing directory: {item_path}")
            shutil.rmtree(item_path)
        # Remove .pt files (old PyTorch models)
        elif item.endswith(".pt"):
            print(f"Removing file: {item_path}")
            os.remove(item_path)
    print("Cleanup complete!")


TOP_10_SPOKEN_LANGUAGES = (
    "cmn_hans_cn",  # Mandarin Chinese (~1.1 billion)
    "es_419",  # Spanish (~500 million)
    "en_us",  # English (~375 million)
    "hi_in",  # Hindi (~350 million)
    "ar_eg",  # Arabic (~310 million)
    "pt_br",  # Portuguese (~230 million)
    "bn_in",  # Bengali (~230 million)
    "ru_ru",  # Russian (~155 million)
    "ja_jp",  # Japanese (~125 million)
    "pa_in",  # Punjabi (~95 million)
)


def download_fleurs_samples(
    lang_code: str,
    split: str = "test",
    min_seconds: float = 3.0,
    max_seconds: float = 6.0,
    n_samples: int = 30,
    out_dir: str = "fleurs_samples",
    seed: int = 42,
    force_redownload: bool = False,
):
    lang_dir = os.path.join(out_dir, lang_code, split)
    manifest_path = os.path.join(lang_dir, "manifest.csv")
    if not force_redownload and os.path.exists(manifest_path):
        with open(manifest_path, newline="", encoding="utf-8") as existing_file:
            row_count = sum(1 for _ in csv.DictReader(existing_file))
        if row_count >= n_samples:
            print(f"Using existing samples for {lang_code} {split} at {manifest_path}")
            return None, manifest_path

    print(f"Preparing samples for {lang_code} {split}...")

    # Load one FLEURS language config
    try:
        ds = load_dataset(
            "google/fleurs",
            lang_code,
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
                "google/fleurs requires remote dataset code. This script now passes "
                "`trust_remote_code=True`; if you still see this error, make sure "
                "you are running the updated `main.py` from this directory."
            ) from exc
        raise

    sampling_rate = ds.features["audio"].sampling_rate or 16000
    ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate, decode=False))

    # Filter by duration using metadata only so we don't decode audio for every row.
    ds = ds.filter(
        lambda x: min_seconds <= _duration_seconds(x, sampling_rate) <= max_seconds
    )

    if len(ds) < n_samples:
        raise ValueError(
            f"Only found {len(ds)} clips in {lang_code} {split} "
            f"between {min_seconds} and {max_seconds} seconds."
        )

    # Random but reproducible selection
    ds = ds.shuffle(seed=seed).select(range(n_samples))

    # Save files
    os.makedirs(lang_dir, exist_ok=True)

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
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

        for i, row in enumerate(ds):
            wav_path = os.path.join(lang_dir, f"{i:03d}_{row['id']}.wav")
            audio_bytes = row["audio"]["bytes"]
            if not audio_bytes:
                raise RuntimeError(
                    f"Missing embedded audio bytes for sample {row['id']}."
                )

            with open(wav_path, "wb") as wav_file:
                wav_file.write(audio_bytes)

            writer.writerow(
                [
                    i,
                    row["id"],
                    _duration_seconds(row, sampling_rate),
                    wav_path,
                    row["transcription"],
                    row["raw_transcription"],
                    row["language"],
                ]
            )

    return ds, manifest_path


def download_all_fleurs_samples(
    split: str = "test",
    min_seconds: float = 3.0,
    max_seconds: float = 6.0,
    n_samples: int = 30,
    out_dir: str = "fleurs_samples",
    seed: int = 42,
    force_redownload: bool = False,
):
    """Download FLEURS samples for all available languages."""
    results = {}
    for lang_code in TOP_10_SPOKEN_LANGUAGES:
        try:
            ds, manifest_path = download_fleurs_samples(
                lang_code=lang_code,
                split=split,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
                n_samples=n_samples,
                out_dir=out_dir,
                seed=seed,
                force_redownload=force_redownload,
            )
            results[lang_code] = (ds, manifest_path)
        except ValueError as e:
            print(f"Skipping {lang_code}: {e}")
            continue
    return results


if __name__ == "__main__":
    download_all_fleurs_samples(
        split="test",
        min_seconds=2,
        max_seconds=10,
        n_samples=30,
    )
