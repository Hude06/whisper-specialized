import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict

from config import DEFAULT_MODELS_ROOT


WHISPER_CPP_MODEL_FILENAMES: Dict[str, str] = {
    "tiny": "ggml-tiny.bin",
    "large": "ggml-large-v3.bin",
}

WHISPER_MODEL_NAMES = tuple(WHISPER_CPP_MODEL_FILENAMES.keys())

_WHISPER_CPP_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"


def _download_file(url: str, dest_path: Path, timeout: int = 60, retries: int = 3) -> None:
    print(f"Downloading {url}...")
    print(f"Destination: {dest_path}")

    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
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
                        percent = (
                            min(downloaded * 100 / total_size, 100)
                            if total_size > 0
                            else 0
                        )
                        print(f"\rProgress: {percent:.1f}%", end="", flush=True)
            os.replace(temp_path, dest_path)
            print("\nDownload complete!")
            return
        except (urllib.error.URLError, TimeoutError) as exc:
            if temp_path.exists():
                temp_path.unlink()
            if attempt == retries:
                raise RuntimeError(
                    f"Failed to download {url} after {retries} attempts: {exc}"
                ) from exc
            wait_seconds = attempt * 2
            print(
                f"\nDownload attempt {attempt} failed: {exc}. "
                f"Retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)


def model_filename(model_name: str) -> str:
    if model_name not in WHISPER_CPP_MODEL_FILENAMES:
        supported = ", ".join(WHISPER_MODEL_NAMES)
        raise ValueError(
            f"Unsupported Whisper model '{model_name}'. Supported models: {supported}."
        )
    return WHISPER_CPP_MODEL_FILENAMES[model_name]


def model_url(model_name: str) -> str:
    return f"{_WHISPER_CPP_BASE_URL}/{model_filename(model_name)}"


def model_path(model_name: str, models_root: str = DEFAULT_MODELS_ROOT) -> Path:
    return Path(models_root) / model_filename(model_name)


def download_whisper_model(
    model_name: str,
    out_dir: str = DEFAULT_MODELS_ROOT,
) -> str:
    destination = model_path(model_name, out_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        print(f"Model {model_name} already exists at {destination}")
        return str(destination)

    _download_file(model_url(model_name), destination)
    return str(destination)


def download_all_whisper_models(out_dir: str = DEFAULT_MODELS_ROOT) -> Dict[str, str]:
    return {
        model_name: download_whisper_model(model_name, out_dir=out_dir)
        for model_name in WHISPER_MODEL_NAMES
    }


def ensure_whisper_model(
    model_name: str,
    models_root: str = DEFAULT_MODELS_ROOT,
) -> str:
    destination = model_path(model_name, models_root)
    if destination.exists():
        return str(destination)
    return download_whisper_model(model_name, out_dir=models_root)
