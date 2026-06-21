"""Shared inference for the fine-tuned PhoBERT sentiment model.

Both the FastAPI backend and the Streamlit app import :func:`predict_sentiment`
so cleaning + model are defined in exactly one place. The model is loaded lazily
and cached, then reused across calls.

Cleaning matches training: strip "||" system noise -> clean_text -> word segment.
No negation tagging here — the fine-tuned transformer handles negation itself.
"""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Reuse the exact cleaning helpers used to build the training data.
_PREP_DIR = Path(__file__).resolve().parent.parent.parent / "preprocessing"
sys.path.insert(0, str(_PREP_DIR))
from preprocess import clean_text, segment_vi  # noqa: E402
from prepare_sentiment import strip_system_noise  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent / "phobert_finetuned"
LABEL_PATH = Path(__file__).resolve().parent / "label_names.json"
TEENCODE_PATH = Path("data-project/teencode_dict.json")
MAX_LEN = 256


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _load():
    """Load tokenizer, model, labels and teencode once (cached)."""
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"{MODEL_DIR} not found. Run models/sentiment/train_phobert.py first."
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device = _device()
    model.to(device).eval()
    labels = (
        {int(k): v for k, v in json.loads(LABEL_PATH.read_text(encoding="utf-8")).items()}
        if LABEL_PATH.exists()
        else {0: "negative", 1: "neutral", 2: "positive"}
    )
    teencode = (
        json.loads(TEENCODE_PATH.read_text(encoding="utf-8"))
        if TEENCODE_PATH.exists()
        else {}
    )
    return tokenizer, model, device, labels, teencode


def clean_for_model(text: str) -> str:
    """Strip system noise, clean, and word-segment a raw review."""
    _, _, _, _, teencode = _load()
    return segment_vi(clean_text(strip_system_noise(text), teencode))


@torch.no_grad()
def predict_sentiment(text: str) -> dict:
    """Return the predicted 3-class sentiment for a raw review string."""
    tokenizer, model, device, labels, _ = _load()
    cleaned = clean_for_model(text)
    enc = tokenizer(
        cleaned,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(device)
    logits = model(**enc).logits
    probs = F.softmax(logits, dim=-1)[0].cpu().tolist()
    best = int(max(range(len(probs)), key=lambda i: probs[i]))
    return {
        "label": best,
        "label_name": labels.get(best, str(best)),
        "confidence": float(probs[best]),
        "probabilities": {labels.get(i, str(i)): float(p) for i, p in enumerate(probs)},
        "cleaned_text": cleaned,
    }
