# FLEURS Benchmark Sampler

This script downloads a small set of FLEURS audio clips, writes them as `.wav` files, and creates a `manifest.csv`.
It also includes helpers for downloading Whisper.cpp model weights (GGML format) for fast inference on Apple Silicon.
It now includes a benchmark runner for measuring Whisper accuracy, speed, memory, and failures using whisper.cpp with Core ML acceleration.

## Why the earlier error happened

`google/fleurs` still relies on a dataset loading script (`fleurs.py`), but `datasets` 4.x removed support for script-backed datasets. It also requires `trust_remote_code=True` when loading. If you have `datasets>=4`, `python3 main.py` fails with:

`RuntimeError: Dataset scripts are no longer supported, but found fleurs.py`

## Install

### Prerequisites

You must have **whisper.cpp** installed. On macOS with Homebrew:

```bash
brew install whisper-cpp
```

### Python Dependencies

Use the compatible dependency set:

```bash
python3 -m pip install --user -r requirements.txt
```

If you already installed the wrong version, force the downgrade:

```bash
python3 -m pip install --user --upgrade --force-reinstall "datasets<4"
```

## Run

From this directory:

```bash
python3 download.py
```

That downloads 30 samples for the top 10 spoken languages from the `test` split into:

```text
fleurs_samples/
  cmn_hans_cn/test/
  es_es/test/
  en_us/test/
  hi_in/test/
  ar_eg/test/
  pt_br/test/
  bn_in/test/
  ru_ru/test/
  ja_jp/test/
  pa_in/test/
```

Each folder will contain `.wav` files plus `manifest.csv`.

## Download Whisper Models

This project uses **whisper.cpp** with GGML format models for fast inference on Apple Silicon (with Core ML acceleration).

Download the GGML models:

```bash
python3 -c "from download import download_whisper_tiny; download_whisper_tiny()"
python3 -c "from download import download_whisper_large; download_whisper_large()"
```

To download all supported models at once:

```bash
python3 -c "from download import download_all_whisper_models; download_all_whisper_models()"
```

Each model is stored as a `.bin` file under:

```text
whisper_models/
  ggml-model-whisper-tiny.bin
  ggml-model-whisper-large.bin
```

**Note:** Old PyTorch models (`.pt` files) are automatically cleaned up when you run the new download functions.

## Run Benchmarks

Benchmark one model:

```bash
python3 benchmark.py --model tiny
```

Benchmark multiple models:

```bash
python3 benchmark.py --model tiny --model large
```

By default, the benchmark scans every `manifest.csv` under:

```text
fleurs_samples/
```

If you want to benchmark only one manifest, pass it explicitly:

```bash
python3 benchmark.py --model tiny --manifest fleurs_samples/en_us/test/manifest.csv
```

Outputs are written to:

```text
benchmark_results/
```

The benchmark produces:

- `benchmark_results.csv`
- `summary.json`
- `wer_vs_rtf.png`
- `memory_vs_rtf.png`
- `accuracy_and_time_by_audio_length.png`
- `accuracy_ranking.png`
- `success_vs_failure.png`
- `inference_time_by_duration_bucket.png`

Tracked metrics:

- Inference time per sample
- Real-time factor (RTF)
- WER against the manifest transcription
- Efficiency score: `(1 - WER) / (1 + RTF)` so higher means a better speed/accuracy tradeoff
- Peak memory usage (RSS)
- Failure count for exceptions and empty outputs

Benchmark notes:

- `benchmark.py` uses the same GGML model paths as [`download.py`](/Users/jude/xp/vorn/benchmarks/download.py).
- If a requested model file is missing, the benchmark resolves it through the downloader path instead of using a separate hard-coded mapping.
- `whisper-cli` from `whisper.cpp` must be installed and available on `PATH`.

## Features

- **Apple Silicon Acceleration**: Automatically uses Core ML for 3-4x faster inference on M1/M2/M3/M4 Macs
- **Auto Language Detection**: whisper.cpp automatically detects the spoken language for each sample
- **GGML Format**: Uses optimized binary models instead of PyTorch checkpoints
- **Top 10 Languages**: Benchmarks the 10 most spoken languages in the world

## Notes

- Importing `download.py` does not download anything. The download only runs when the file is executed directly.
- Clip duration filtering uses each audio item's declared sample rate instead of assuming `16000`.
- The script explicitly enables `trust_remote_code=True` for `google/fleurs`, because that dataset is distributed with a loader script on Hugging Face.
- Audio files are written from the dataset-provided WAV bytes directly, so no extra audio decoding libraries are required.
- Supported models: `tiny`, `large` (whisper.cpp GGML format)
