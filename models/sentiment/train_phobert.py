"""Fine-tune vinai/phobert-base for Vietnamese sentiment classification.

This script is optional for GPU environments. The TF-IDF Random Forest baseline
is the recommended CPU path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


SEED = 42
MODEL_NAME = "vinai/phobert-base"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
    }


def train(input_path: Path, output_dir: Path, epochs: int, batch_size: int) -> None:
    df = pd.read_csv(input_path).dropna(subset=["Review_Text_Segmented", "label"])
    df["label"] = df["label"].astype(int)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    metrics = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(df["Review_Text_Segmented"], df["label"]), start=1):
        train_ds = Dataset.from_pandas(
            df.iloc[train_idx][["Review_Text_Segmented", "label"]].rename(columns={"Review_Text_Segmented": "text"})
        ).map(tokenize, batched=True)
        valid_ds = Dataset.from_pandas(
            df.iloc[valid_idx][["Review_Text_Segmented", "label"]].rename(columns={"Review_Text_Segmented": "text"})
        ).map(tokenize, batched=True)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        args = TrainingArguments(
            output_dir=str(output_dir / f"fold_{fold}"),
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            evaluation_strategy="epoch",
            save_strategy="no",
            seed=SEED,
            report_to=[],
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
        )
        trainer.train()
        row = trainer.evaluate()
        row["fold"] = fold
        metrics.append(row)

    final_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    full_ds = Dataset.from_pandas(
        df[["Review_Text_Segmented", "label"]].rename(columns={"Review_Text_Segmented": "text"})
    ).map(tokenize, batched=True)
    final_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy="epoch",
        seed=SEED,
        report_to=[],
    )
    Trainer(
        model=final_model,
        args=final_args,
        train_dataset=full_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    ).train()
    final_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    pd.DataFrame(metrics).to_csv(output_dir / "phobert_cv_metrics.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data-project/processed/reviews_clean.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/sentiment/phobert_model"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    train(args.input, args.output, args.epochs, args.batch_size)
