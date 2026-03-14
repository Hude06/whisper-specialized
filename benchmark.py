import argparse
import csv
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from jiwer import wer
from download import WHISPER_MODEL_NAMES, download_whisper_model


DURATION_BUCKETS = [
    ("short", 2.0, 5.0),
    ("medium", 5.0, 8.0),
    ("long", 8.0, 10.0),
]

SUPPORTED_MODELS = WHISPER_MODEL_NAMES


@dataclass
class BenchmarkResult:
    model_name: str
    language: str
    split: str
    manifest_path: str
    sample_index: int
    fleurs_id: str
    audio_path: str
    audio_duration: float
    duration_bucket: str
    inference_time: Optional[float]
    rtf: Optional[float]
    wer_value: Optional[float]
    efficiency_score: Optional[float]
    peak_memory_mb: Optional[float]
    success: bool
    failure_reason: str
    prediction: str
    reference: str


def _ensure_whisper_cli() -> None:
    if shutil.which("whisper-cli") is None:
        raise RuntimeError(
            "`whisper-cli` is not installed or not on PATH. Install whisper.cpp "
            "and make sure `whisper-cli` is runnable before benchmarking."
        )


def _get_whisper_cpp_model_path(model_name: str, models_dir: str = "whisper_models") -> str:
    """Resolve a GGML Whisper model path using the downloader as the source of truth."""
    if model_name not in SUPPORTED_MODELS:
        supported = ", ".join(SUPPORTED_MODELS)
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models: {supported}."
        )

    model_path = Path(download_whisper_model(model_name, out_dir=models_dir))
    return str(model_path)


def _duration_bucket(duration_sec: float) -> str:
    for label, lower, upper in DURATION_BUCKETS:
        if lower <= duration_sec < upper:
            return label
    if duration_sec >= DURATION_BUCKETS[-1][1]:
        return DURATION_BUCKETS[-1][0]
    return DURATION_BUCKETS[0][0]


def _read_manifest(manifest_path: str) -> List[Dict[str, str]]:
    with open(manifest_path, newline="", encoding="utf-8") as manifest_file:
        return list(csv.DictReader(manifest_file))


def _discover_manifest_paths(samples_root: str) -> List[Path]:
    root = Path(samples_root)
    return sorted(root.rglob("manifest.csv"))


def _resolve_audio_path(base_dir: Path, audio_path: str) -> Path:
    audio_file = Path(audio_path)
    if audio_file.is_absolute():
        return audio_file
    return (base_dir / audio_file).resolve()


def _monitor_peak_memory(
    stop_event: threading.Event, interval_sec: float = 0.01
) -> List[int]:
    process = psutil.Process(os.getpid())
    samples = []
    while not stop_event.is_set():
        try:
            samples.append(process.memory_info().rss)
        except psutil.Error:
            break
        time.sleep(interval_sec)
    return samples


def _transcribe_with_metrics(model_path: str, audio_path: str) -> Dict[str, object]:
    """Transcribe audio using whisper-cli with timing and memory monitoring."""
    import tempfile

    _ensure_whisper_cli()
    stop_event = threading.Event()
    memory_samples: List[int] = []

    def sampler():
        memory_samples.extend(_monitor_peak_memory(stop_event))

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()

    # Create a temporary file for JSON output
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        start = time.perf_counter()

        # Build whisper-cli command with auto language detection
        cmd = [
            "whisper-cli",
            "-m",
            model_path,
            "-f",
            audio_path,
            "-oj",  # Output JSON
            "-np",  # No prints (suppress progress)
            "-l",
            "auto",  # Auto language detection
            "-of",
            tmp_path.replace(".json", ""),  # Output file prefix (whisper adds .json)
        ]

        # Run whisper-cli
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        inference_time = time.perf_counter() - start
        stop_event.set()
        sampler_thread.join()

        # Read the JSON output file (whisper-cli adds .json extension)
        json_output_path = tmp_path.replace(".json", ".json")
        with open(json_output_path, "r") as f:
            output_data = json.load(f)

        # Extract transcription text from JSON output
        text_parts = []
        if "transcription" in output_data:
            for segment in output_data["transcription"]:
                if "text" in segment:
                    text_parts.append(segment["text"].strip())

        transcription_text = " ".join(text_parts).strip()

        peak_memory_mb = max(
            memory_samples, default=psutil.Process(os.getpid()).memory_info().rss
        ) / (1024 * 1024)

        return {
            "text": transcription_text,
            "inference_time": inference_time,
            "peak_memory_mb": peak_memory_mb,
        }

    except subprocess.CalledProcessError as e:
        stop_event.set()
        sampler_thread.join()
        raise RuntimeError(f"whisper-cli failed: {e.stderr}") from e
    except json.JSONDecodeError as e:
        stop_event.set()
        sampler_thread.join()
        raise RuntimeError(f"Failed to parse whisper-cli JSON output: {e}") from e
    finally:
        # Clean up temporary files
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        json_output_path = tmp_path.replace(".json", ".json")
        if os.path.exists(json_output_path):
            os.remove(json_output_path)


def _benchmark_manifest(
    model_name: str, model_path: str, manifest_file: Path
) -> List[BenchmarkResult]:
    project_dir = manifest_file.parents[3]
    rows = _read_manifest(str(manifest_file))
    split = manifest_file.parent.name
    language = manifest_file.parent.parent.name

    results: List[BenchmarkResult] = []
    for row in rows:
        audio_duration = float(row["duration_sec"])
        bucket = _duration_bucket(audio_duration)
        audio_path = str(_resolve_audio_path(project_dir, row["audio_path"]))
        reference = row["transcription"].strip()

        try:
            metrics = _transcribe_with_metrics(model_path, audio_path)
            prediction = str(metrics["text"]).strip()
            inference_time = float(metrics["inference_time"])
            peak_memory_mb = float(metrics["peak_memory_mb"])
            rtf = inference_time / audio_duration if audio_duration > 0 else None
            success = bool(prediction)
            failure_reason = "" if success else "empty_output"
            wer_value = wer(reference, prediction) if success else None
            efficiency_score = _efficiency_score(wer_value, rtf) if success else None
        except Exception as exc:
            prediction = ""
            inference_time = None
            peak_memory_mb = None
            rtf = None
            success = False
            failure_reason = str(exc)
            wer_value = None
            efficiency_score = None

        results.append(
            BenchmarkResult(
                model_name=model_name,
                language=language,
                split=split,
                manifest_path=str(manifest_file),
                sample_index=int(row["sample_index"]),
                fleurs_id=row["fleurs_id"],
                audio_path=audio_path,
                audio_duration=audio_duration,
                duration_bucket=bucket,
                inference_time=inference_time,
                rtf=rtf,
                wer_value=wer_value,
                efficiency_score=efficiency_score,
                peak_memory_mb=peak_memory_mb,
                success=success,
                failure_reason=failure_reason,
                prediction=prediction,
                reference=reference,
            )
        )

    return results


def benchmark_model(
    model_name: str,
    manifest_paths: List[str],
    models_dir: str = "whisper_models",
) -> List[BenchmarkResult]:
    if model_name not in SUPPORTED_MODELS:
        supported = ", ".join(SUPPORTED_MODELS)
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models: {supported}."
        )

    model_path = _get_whisper_cpp_model_path(model_name, models_dir)

    results: List[BenchmarkResult] = []
    for manifest_path in manifest_paths:
        manifest_file = Path(manifest_path).resolve()
        results.extend(_benchmark_manifest(model_name, model_path, manifest_file))
    return results


def _results_to_dataframe(results: List[BenchmarkResult]) -> pd.DataFrame:
    return pd.DataFrame([result.__dict__ for result in results])


def _efficiency_score(
    wer_value: Optional[float], rtf: Optional[float]
) -> Optional[float]:
    if wer_value is None or rtf is None:
        return None
    accuracy = max(0.0, 1.0 - wer_value)
    return float(accuracy / (1.0 + rtf))


def _write_summary(summary_path: Path, results_df: pd.DataFrame) -> None:
    summary = {}
    for model_name, group in results_df.groupby("model_name"):
        successes = int(group["success"].sum())
        failures = int((~group["success"]).sum())
        summary[model_name] = {
            "samples": int(len(group)),
            "successes": successes,
            "failures": failures,
            "mean_inference_time_sec": _safe_mean(group["inference_time"]),
            "mean_rtf": _safe_mean(group["rtf"]),
            "mean_wer": _safe_mean(group["wer_value"]),
            "mean_efficiency_score": _safe_mean(group["efficiency_score"]),
            "best_efficiency_score": _safe_max(group["efficiency_score"]),
            "peak_memory_mb": _safe_max(group["peak_memory_mb"]),
        }

    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)


def _safe_mean(series: pd.Series) -> Optional[float]:
    values = series.dropna()
    if values.empty:
        return None
    return float(values.mean())


def _safe_max(series: pd.Series) -> Optional[float]:
    values = series.dropna()
    if values.empty:
        return None
    return float(values.max())


def _save_plots(results_df: pd.DataFrame, output_dir: Path) -> None:
    successful = results_df[results_df["success"]].copy()
    failures = results_df.copy()

    fig, ax = plt.subplots(figsize=(9, 6))
    if successful.empty:
        ax.text(0.5, 0.5, "No successful samples to plot", ha="center", va="center")
        ax.set_axis_off()
    else:
        for model_name, group in successful.groupby("model_name"):
            ax.scatter(group["rtf"], group["wer_value"], label=model_name, alpha=0.8)
        ax.set_xlabel("Real-Time Factor (RTF)")
        ax.set_ylabel("WER")
        ax.set_title("WER vs RTF")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "wer_vs_rtf.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    if successful.empty:
        ax.text(0.5, 0.5, "No successful samples to plot", ha="center", va="center")
        ax.set_axis_off()
    else:
        for model_name, group in successful.groupby("model_name"):
            ax.scatter(
                group["rtf"], group["peak_memory_mb"], label=model_name, alpha=0.8
            )
        ax.set_xlabel("Real-Time Factor (RTF)")
        ax.set_ylabel("Peak Memory (MB)")
        ax.set_title("Memory vs RTF")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "memory_vs_rtf.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 10))
    if successful.empty:
        ax.text(0.5, 0.5, "No successful samples to plot", ha="center", va="center")
        ax.set_axis_off()
    else:
        successful = successful.copy()
        successful["accuracy"] = 1.0 - successful["wer_value"]
        norm = Normalize(vmin=0.0, vmax=1.0)
        scatter = ax.scatter(
            successful["audio_duration"],
            successful["inference_time"],
            c=successful["accuracy"],
            cmap="turbo",
            norm=norm,
            s=130,
            alpha=0.95,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.grid(alpha=0.2)

        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.03)
        colorbar.set_label("Accuracy (1 - WER)")
        ax.set_title(
            "Transcription Time by Audio Length, Colored by Accuracy", fontsize=16
        )
        ax.set_xlabel("Audio Duration (s)", fontsize=13)
        ax.set_ylabel("Inference Time (s)", fontsize=13)

        time_values = successful["inference_time"].dropna()
        if not time_values.empty:
            ymin = max(0.0, float(time_values.min()) - 0.05)
            ymax = float(time_values.max()) + 0.08
            if ymax > ymin:
                ax.set_ylim(ymin, ymax)

        worst_points = successful.nsmallest(min(5, len(successful)), "accuracy")
        best_points = successful.nlargest(min(5, len(successful)), "accuracy")
        labeled = pd.concat([best_points, worst_points]).drop_duplicates(
            subset=["model_name", "sample_index"]
        )
        for _, row in labeled.iterrows():
            ax.annotate(
                f"WER={row['wer_value']:.3f}",
                (row["audio_duration"], row["inference_time"]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "fc": "white",
                    "alpha": 0.75,
                    "ec": "none",
                },
            )
    fig.tight_layout(rect=[0, 0, 0.94, 1])
    fig.savefig(output_dir / "accuracy_and_time_by_audio_length.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 7))
    if successful.empty:
        ax.text(0.5, 0.5, "No successful samples to plot", ha="center", va="center")
        ax.set_axis_off()
    else:
        detailed = successful.sort_values(["accuracy", "inference_time"]).reset_index(
            drop=True
        )
        y_positions = np.arange(len(detailed))
        norm = Normalize(vmin=0.0, vmax=1.0)
        bars = ax.barh(
            y_positions,
            detailed["accuracy"],
            color=plt.cm.turbo(norm(detailed["accuracy"].to_numpy())),
            edgecolor="black",
            linewidth=0.3,
        )
        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            [
                f"{row.model_name}:{int(row.sample_index)} ({row.audio_duration:.1f}s)"
                for row in detailed.itertuples()
            ],
            fontsize=8,
        )
        ax.set_xlabel("Accuracy (1 - WER)")
        ax.set_ylabel("Sample")
        ax.set_title("Per-Sample Accuracy Ranking")
        ax.set_xlim(0.0, 1.0)
        for bar, row in zip(bars, detailed.itertuples()):
            ax.text(
                min(bar.get_width() + 0.01, 0.99),
                bar.get_y() + bar.get_height() / 2,
                f"{row.inference_time:.2f}s",
                va="center",
                fontsize=8,
            )
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_ranking.png")
    plt.close(fig)

    bucket_order = [bucket for bucket, _, _ in DURATION_BUCKETS]
    boxplot_data = [
        successful.loc[successful["duration_bucket"] == bucket, "inference_time"]
        .dropna()
        .tolist()
        for bucket in bucket_order
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    if not any(boxplot_data):
        ax.text(0.5, 0.5, "No successful samples to plot", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.boxplot(boxplot_data, labels=bucket_order)
        ax.set_xlabel("Audio Duration Bucket")
        ax.set_ylabel("Inference Time (s)")
        ax.set_title("Inference Time by Duration Bucket")
    fig.tight_layout()
    fig.savefig(output_dir / "inference_time_by_duration_bucket.png")
    plt.close(fig)

    counts = (
        failures.groupby("model_name")["success"].agg(["sum", "count"]).reset_index()
    )
    counts["failures"] = counts["count"] - counts["sum"]
    x = np.arange(len(counts))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, counts["sum"], width, label="success")
    ax.bar(x + width / 2, counts["failures"], width, label="failure")
    ax.set_xticks(x)
    ax.set_xticklabels(counts["model_name"])
    ax.set_ylabel("Sample Count")
    ax.set_title("Success vs Failure Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "success_vs_failure.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Whisper models on downloaded FLEURS samples."
    )
    parser.add_argument(
        "--manifest",
        action="append",
        dest="manifests",
        help="Path to a specific FLEURS manifest.csv file. Repeat to benchmark multiple manifests.",
    )
    parser.add_argument(
        "--samples-root",
        default="fleurs_samples",
        help="Root directory to scan for manifest.csv files when --manifest is not provided.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Whisper model to benchmark. Repeat to benchmark multiple models.",
    )
    parser.add_argument(
        "--models-dir",
        default="whisper_models",
        help="Directory containing per-model Whisper download roots.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory for benchmark CSV, summary JSON, and plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = args.models or ["tiny"]
    manifest_paths = args.manifests
    if manifest_paths is None:
        manifest_paths = [
            str(path) for path in _discover_manifest_paths(args.samples_root)
        ]
    if not manifest_paths:
        raise FileNotFoundError(
            f"No manifest.csv files found. Checked --samples-root={args.samples_root!r}."
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[BenchmarkResult] = []
    for model_name in models:
        all_results.extend(
            benchmark_model(
                model_name=model_name,
                manifest_paths=manifest_paths,
                models_dir=args.models_dir,
            )
        )

    results_df = _results_to_dataframe(all_results)
    results_csv_path = output_dir / "benchmark_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    _write_summary(output_dir / "summary.json", results_df)
    _save_plots(results_df, output_dir)

    print(f"Wrote benchmark results to {results_csv_path}")
    print(f"Wrote summary to {output_dir / 'summary.json'}")
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
