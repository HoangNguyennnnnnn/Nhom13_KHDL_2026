"""PhoBERT frozen feature extractor (data-mining stage [5] feature engineering).

PhoBERT is used here as a *frozen* sentence encoder: its 135M weights are never
trained. We only run a forward pass to turn each (word-segmented) Vietnamese
review into a 768-dim contextual vector. A small classifier — the model the
student actually owns — is then trained on top of these vectors.

Why this beats TF-IDF: TF-IDF is bag-of-words and is blind to word order, so it
cannot tell "tốt" from "không tốt". PhoBERT reads the whole sentence and encodes
negation / contrast into the vector, while still costing 0 minutes of training.

Important: PhoBERT expects **word-segmented** input ("màn_hình" not "màn hình").
Feed it the ``Review_Text_Segmented`` column produced by prepare_sentiment.py.
"""

from __future__ import annotations

import os

# Let unsupported ops fall back to CPU instead of crashing on Apple Silicon (MPS).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from typing import Iterable

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 256


def pick_device() -> str:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PhoBERTEmbedder:
    """Wraps PhoBERT to produce mean-pooled sentence embeddings.

    The model is loaded lazily and kept in eval mode with gradients disabled so
    it behaves as a pure feature extractor.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.model_name = model_name
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Average token vectors, ignoring padding (mask-weighted mean)."""
        mask = mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @torch.no_grad()
    def encode(
        self,
        texts: Iterable[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Return an (n, 768) float32 array of sentence embeddings."""
        texts = [str(t) if t is not None else "" for t in texts]
        vectors: list[np.ndarray] = []
        total = len(texts)
        for start in range(0, total, batch_size):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc).last_hidden_state
            pooled = self._mean_pool(out, enc["attention_mask"])
            vectors.append(pooled.cpu().numpy().astype(np.float32))
            if show_progress:
                done = min(start + batch_size, total)
                print(f"\r[embed] {done}/{total}", end="", flush=True)
        if show_progress:
            print()
        return np.vstack(vectors) if vectors else np.zeros((0, 768), dtype=np.float32)


# Module-level singleton so the API loads PhoBERT only once.
_EMBEDDER: PhoBERTEmbedder | None = None


def get_embedder() -> PhoBERTEmbedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = PhoBERTEmbedder()
    return _EMBEDDER
