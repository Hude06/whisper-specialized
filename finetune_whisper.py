import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import evaluate
import torch
from datasets import Audio, Dataset, concatenate_datasets
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)


def _resolve_audio_path(manifest_path: Path, audio_path: str) -> str:
    audio_file = Path(audio_path)
    if audio_file.is_absolute():
        return str(audio_file)

    manifest_relative = (manifest_path.parent / audio_file).resolve()
    if manifest_relative.exists():
        return str(manifest_relative)

    repo_relative = audio_file.resolve()
    if repo_relative.exists():
        return str(repo_relative)

    return str(manifest_relative)


def _read_manifest_dataset(manifest_path: str) -> Dataset:
    manifest_file = Path(manifest_path).resolve()
    rows: List[Dict[str, str]] = []
    with open(manifest_file, newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            rows.append(
                {
                    "audio": _resolve_audio_path(manifest_file, row["audio_path"]),
                    "transcription": row["transcription"].strip(),
                    "manifest_path": str(manifest_file),
                }
            )

    if not rows:
        raise ValueError(f"Manifest has no rows: {manifest_file}")

    return Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=16000))


def _load_manifest_group(manifest_paths: List[str]) -> Dataset:
    datasets = [_read_manifest_dataset(path) for path in manifest_paths]
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100,
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Hugging Face Whisper checkpoint from local manifest.csv files."
    )
    parser.add_argument(
        "--train-manifest",
        action="append",
        dest="train_manifests",
        required=True,
        help="Training manifest.csv path. Repeat to include multiple manifests.",
    )
    parser.add_argument(
        "--eval-manifest",
        action="append",
        dest="eval_manifests",
        help="Evaluation manifest.csv path. Repeat to include multiple manifests.",
    )
    parser.add_argument(
        "--base-model",
        default="openai/whisper-tiny",
        help="Base Hugging Face Whisper checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--language",
        help=(
            "Language token for multilingual Whisper models, for example 'English' or "
            "'Hindi'. Omit this for English-only checkpoints like *.en models."
        ),
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Whisper task token to use during training and generation.",
    )
    parser.add_argument(
        "--output-dir",
        default="finetuned_models/whisper-run",
        help="Directory to write checkpoints and the final fine-tuned model.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Number of optimizer steps to train.",
    )
    parser.add_argument(
        "--eval-split-size",
        type=float,
        default=0.1,
        help="Validation fraction to hold out if --eval-manifest is omitted.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for fine-tuning.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=50,
        help="Warmup steps for the scheduler.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=8,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=8,
        help="Per-device eval batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=2,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Logging interval in steps.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Checkpoint save interval in steps.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Evaluation interval in steps when validation data is available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/eval splitting and training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.max_steps <= 0:
        raise ValueError("--max-steps must be greater than zero.")
    if not 0.0 < args.eval_split_size < 1.0:
        raise ValueError("--eval-split-size must be between 0 and 1.")

    train_dataset = _load_manifest_group(args.train_manifests)
    eval_dataset: Optional[Dataset] = None

    if args.eval_manifests:
        eval_dataset = _load_manifest_group(args.eval_manifests)
    elif len(train_dataset) > 1:
        split_dataset = train_dataset.train_test_split(
            test_size=args.eval_split_size,
            seed=args.seed,
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.base_model)
    processor_kwargs = {"task": args.task}
    tokenizer_kwargs = {"task": args.task}
    if args.language:
        processor_kwargs["language"] = args.language
        tokenizer_kwargs["language"] = args.language
    tokenizer = WhisperTokenizer.from_pretrained(args.base_model, **tokenizer_kwargs)
    processor = WhisperProcessor.from_pretrained(args.base_model, **processor_kwargs)

    def prepare_dataset(batch: Dict[str, object]) -> Dict[str, object]:
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]
        batch["labels"] = tokenizer(batch["transcription"]).input_ids
        return batch

    train_dataset = train_dataset.map(
        prepare_dataset,
        remove_columns=train_dataset.column_names,
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            prepare_dataset,
            remove_columns=eval_dataset.column_names,
        )

    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model.config.use_cache = False
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    if args.language:
        model.generation_config.language = args.language

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    wer_metric = evaluate.load("wer")

    def compute_metrics(prediction_output) -> Dict[str, float]:
        predicted_ids = prediction_output.predictions
        label_ids = prediction_output.label_ids.copy()
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        predicted_text = tokenizer.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )
        label_text = tokenizer.batch_decode(
            label_ids,
            skip_special_tokens=True,
        )
        return {"wer": 100.0 * wer_metric.compute(predictions=predicted_text, references=label_text)}

    has_eval = eval_dataset is not None
    report_to = ["tensorboard"]
    use_fp16 = torch.cuda.is_available()

    training_args_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=use_fp16,
        evaluation_strategy="steps" if has_eval else "no",
        save_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=report_to,
        remove_unused_columns=False,
        label_names=["labels"],
        seed=args.seed,
    )
    if has_eval:
        training_args_kwargs["eval_steps"] = args.eval_steps
        training_args_kwargs["load_best_model_at_end"] = True
        training_args_kwargs["metric_for_best_model"] = "wer"
        training_args_kwargs["greater_is_better"] = False

    training_args = Seq2SeqTrainingArguments(
        **training_args_kwargs,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics if has_eval else None,
        tokenizer=processor.feature_extractor,
    )

    print(f"Training examples: {len(train_dataset)}")
    print(f"Evaluation examples: {len(eval_dataset) if eval_dataset is not None else 0}")
    print(f"Base model: {args.base_model}")
    if args.language:
        print(f"Language token: {args.language}")

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    if has_eval:
        metrics = trainer.evaluate()
        print(f"Final eval WER: {metrics['eval_wer']:.2f}")

    print(f"Saved fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()
