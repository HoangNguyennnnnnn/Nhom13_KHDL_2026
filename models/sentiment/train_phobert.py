"""Fine-tune PhoBERT for 3-class Vietnamese sentiment (the production model).

Unlike a frozen feature extractor, here PhoBERT's weights ARE updated, so the
model learns Vietnamese negation/context directly from the labelled reviews —
fixing cases like "máy không tốt" that the TF-IDF + rule pipeline could not.

Input : data-project/processed/reviews_labeled.csv (column Review_Text_Segmented,
        produced by preprocessing/prepare_sentiment.py — cleaned + word-segmented,
        WITHOUT the negation tags, because PhoBERT handles negation itself).
Output: models/sentiment/phobert_finetuned/  (HF model + tokenizer)
        models/sentiment/label_names.json
        models/sentiment/phobert_metrics.json (held-out test metrics)

Runs on Apple MPS / CUDA automatically, else CPU. ~10-20 min on MPS.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Let unsupported ops fall back to CPU on Apple Silicon instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

SEED = 42
MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 256
DATA_PATH = Path("data-project/processed/reviews_labeled.csv")
OUT_DIR = Path("models/sentiment/phobert_finetuned")
TEXT_COL = "Review_Text_Segmented"
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run preprocessing/prepare_sentiment.py first."
        )
    df = pd.read_csv(DATA_PATH).dropna(subset=[TEXT_COL, "label"]).reset_index(drop=True)
    df = df[df[TEXT_COL].astype(str).str.strip().astype(bool)].reset_index(drop=True)
    texts = df[TEXT_COL].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.15, stratify=labels, random_state=SEED
    )
    print(f"[train] train={len(X_train)}  test={len(X_test)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def encode(batch):
        return tokenizer(batch, truncation=True, padding="max_length", max_length=MAX_LEN)

    train_ds = ReviewDataset(encode(X_train), y_train)
    test_ds = ReviewDataset(encode(X_test), y_test)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=LABEL_NAMES,
        label2id={v: k for k, v in LABEL_NAMES.items()},
    )

    args = TrainingArguments(
        output_dir="models/sentiment/_phobert_ckpt",
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=20,
        seed=SEED,
        fp16=False,  # MPS does not support fp16 training
        report_to="none",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    print(f"[train] device: {trainer.args.device}")
    trainer.train()

    # Final held-out evaluation report.
    preds = np.argmax(trainer.predict(test_ds).predictions, axis=-1)
    print("\n[train] held-out test report:")
    print(
        classification_report(
            y_test, preds, target_names=list(LABEL_NAMES.values()), zero_division=0
        )
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "n_test": len(y_test),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    (Path("models/sentiment") / "label_names.json").write_text(
        json.dumps(LABEL_NAMES, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (Path("models/sentiment") / "phobert_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[train] saved fine-tuned model -> {OUT_DIR}")
    print(f"[train] test metrics: {metrics}")


if __name__ == "__main__":
    main()
