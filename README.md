# FLEURS + whisper.cpp Benchmarking

This repo has three clean entry points:

- `python3 download.py` downloads FLEURS clips and writes `manifest.csv` files.
- `python3 download_models.py --model tiny` downloads `whisper.cpp` model binaries.
- `python3 benchmark.py --model tiny` benchmarks a model with `whisper-cli`.

## Defaults You Can Edit In Code

The default dataset shape and folder layout live in `config.py`.

That is the file to change when you want a different standard setup for:

- languages to download
- split
- minimum clip length
- maximum clip length
- clips per language
- sample root directory
- model directory
- benchmark output root

Current defaults are set through these constants:

- `DEFAULT_LANGUAGES`
- `DEFAULT_SPLIT`
- `DEFAULT_MIN_SECONDS`
- `DEFAULT_MAX_SECONDS`
- `DEFAULT_SAMPLE_COUNT`
- `DEFAULT_SAMPLES_ROOT`
- `DEFAULT_MODELS_ROOT`
- `DEFAULT_BENCHMARK_ROOT`

## Install

### whisper.cpp

`benchmark.py` uses `whisper-cli`, so `whisper.cpp` must be installed and available on `PATH`.

On macOS with Homebrew:

```bash
brew install whisper-cpp
```

### Python Dependencies

```bash
python3 -m pip install --user -r requirements.txt
```

If `google/fleurs` fails because of the `datasets` version:

```bash
python3 -m pip install --user --upgrade --force-reinstall "datasets<4"
```

## Download FLEURS Samples

Use the defaults from `config.py`:

```bash
python3 download.py
```

Override them from the CLI when needed:

```bash
python3 download.py --language en_us --language ja_jp --sample-count 20 --min-seconds 3 --max-seconds 8
```

The downloader stores data under:

```text
fleurs_samples/<language>/<split>/
```

Each folder contains:

- `manifest.csv`
- `download_config.json`
- downloaded `.wav` files

The downloader reuses an existing language/split only when the saved download settings still match the requested settings. If you change clip count or duration bounds, it refreshes that manifest automatically.

Use `--force-redownload` to rebuild a language/split even when the config matches.

## Download whisper.cpp Models

Download one model:

```bash
python3 download_models.py --model tiny
```

Download multiple:

```bash
python3 download_models.py --model tiny --model large
```

If you run `download_models.py` with no `--model`, it downloads every supported model.

Supported model names are defined in `whisper_cpp.py`.

## Fine-Tune A Whisper Model

Fine-tuning is not a `whisper.cpp` operation. The usual flow is:

1. Fine-tune a Hugging Face Whisper checkpoint with PyTorch.
2. Evaluate the resulting checkpoint.
3. Convert that trained checkpoint later if you want to run it through `whisper.cpp`.

This repo now includes a simple training script:

```bash
python3 finetune_whisper.py \
  --train-manifest fleurs_samples/en_us/test/manifest.csv \
  --base-model openai/whisper-tiny \
  --language English \
  --output-dir finetuned_models/en-us-tiny
```

If you do not pass `--eval-manifest`, the script automatically holds out part of the training set for evaluation.

Training dependencies are kept separate from the benchmark dependencies:

```bash
python3 -m pip install --user -r requirements-finetune.txt
```

The fine-tune script reads the same `manifest.csv` files created by `download.py`, so you can:

1. Change defaults in `config.py` if you want a different clip count or duration range.
2. Run `python3 download.py` to refresh your dataset.
3. Run `python3 finetune_whisper.py ...` to train a Hugging Face Whisper checkpoint.
4. Run `python3 benchmark.py --model ...` for the stock `whisper.cpp` models already supported here.

## Run Benchmarks

Benchmark one model:

```bash
python3 benchmark.py --model tiny
```

Benchmark a specific manifest only:

```bash
python3 benchmark.py --model tiny --manifest fleurs_samples/en_us/test/manifest.csv
```

Benchmark multiple models:

```bash
python3 benchmark.py --model tiny --model large
```

By default, `benchmark.py` scans every `manifest.csv` under `fleurs_samples/`.

During the run, the terminal prints:

- model start and finish
- manifest progress
- per-sample progress

## Benchmark Outputs

Benchmark outputs are now written per model under the benchmark root:

```text
benchmark_results/
  tiny/
    benchmark_results.csv
    summary.json
    wer_vs_rtf.png
    memory_vs_rtf.png
    accuracy_and_time_by_audio_length.png
    accuracy_ranking.png
    success_vs_failure.png
    inference_time_by_duration_bucket.png
  large/
    ...
```

If you pass `--output-dir custom_results`, the model outputs go to:

```text
custom_results/<model_name>/
```

## Workflow

1. Edit `config.py` if you want a new default sample count or clip duration range.
2. Run `python3 download.py` to fetch or refresh manifests that match those settings.
3. Run `python3 download_models.py --model <model>` to fetch the `whisper.cpp` binary model.
4. Run `python3 benchmark.py --model <model>` to benchmark that model and write results into its own folder.
