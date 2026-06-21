"""Negation-focused data augmentation (data-mining: targeted data synthesis).

EDA on the labelled reviews showed negation phrases are severely under-represented
(e.g. "không tốt" appears in only 9 reviews, "không khoẻ" in 0), so no model can
learn them. We close that gap by synthesising negation examples from a sentiment
lexicon:

    "<subject> không/chẳng/chưa <positive word> <suffix>"  -> negative
    "<subject> không/chẳng/chưa <negative word> <suffix>"  -> positive (double neg)

These are appended to the TRAIN split only (never the test split) so evaluation
stays on 100% real data.
"""

from __future__ import annotations

import random

# Positive descriptors that become NEGATIVE under negation.
POS_WORDS = [
    "tốt", "đẹp", "khoẻ", "khỏe", "mượt", "bền", "nhanh", "ngon", "ổn", "ưng",
    "đáng tiền", "đáng", "nét", "sắc nét", "mạnh", "xịn", "rẻ", "hài lòng",
    "ngon lành", "trâu", "êm",
]
# Negative descriptors that become POSITIVE under negation (double negation).
NEG_WORDS = [
    "tệ", "kém", "dở", "chậm", "lag", "hỏng", "lỗi", "đắt", "nóng", "yếu", "xấu",
]
NEGATORS = ["không", "chẳng", "chưa", "ko", "đâu có"]
SUBJECTS = [
    "", "máy ", "sản phẩm ", "pin ", "màn hình ", "camera ", "điện thoại ",
    "máy này ", "con này ", "hàng ",
]
SUFFIXES = ["", " lắm", " chút nào", " gì cả", " tí nào", " như mong đợi", " như quảng cáo"]


def build_negation_examples(
    n_negative: int = 350,
    n_positive: int = 120,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """Return (raw_texts, labels) of synthetic negation sentences.

    labels: 0 = negative, 2 = positive. Texts are raw (not cleaned/segmented);
    the caller should run them through the same cleaning as the real data.
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    texts: list[str] = []
    labels: list[int] = []

    def emit(words: list[str], label: int, target: int) -> None:
        tries = 0
        added = 0
        while added < target and tries < target * 40:
            tries += 1
            subj = rng.choice(SUBJECTS)
            neg = rng.choice(NEGATORS)
            word = rng.choice(words)
            suf = rng.choice(SUFFIXES)
            sent = f"{subj}{neg} {word}{suf}".strip()
            if sent in seen:
                continue
            seen.add(sent)
            texts.append(sent)
            labels.append(label)
            added += 1

    emit(POS_WORDS, label=0, target=n_negative)  # "không tốt" -> negative
    emit(NEG_WORDS, label=2, target=n_positive)  # "không tệ" -> positive
    return texts, labels


if __name__ == "__main__":  # quick visual check
    t, y = build_negation_examples(8, 4)
    for s, lab in zip(t, y):
        print(lab, "|", s)
