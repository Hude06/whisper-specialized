from typing import Tuple


# Edit these defaults in code if you want a different standard dataset shape.
DEFAULT_LANGUAGES: Tuple[str, ...] = (
    "cmn_hans_cn",
    "es_419",
    "en_us",
    "hi_in",
    "ar_eg",
    "pt_br",
    "bn_in",
    "ru_ru",
    "ja_jp",
    "pa_in",
)

DEFAULT_SPLIT = "test"
DEFAULT_MIN_SECONDS = 2.0
DEFAULT_MAX_SECONDS = 10.0
DEFAULT_SAMPLE_COUNT = 30
DEFAULT_RANDOM_SEED = 42

DEFAULT_SAMPLES_ROOT = "fleurs_samples"
DEFAULT_MODELS_ROOT = "whisper_models"
DEFAULT_BENCHMARK_ROOT = "benchmark_results"

DOWNLOAD_METADATA_FILENAME = "download_config.json"
