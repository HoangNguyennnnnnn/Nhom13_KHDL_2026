"""Data augmentation cho sentiment (data-mining: targeted data synthesis).

EDA cho thấy 3 lỗ hổng dữ liệu khiến model đoán sai:
  1. Câu phủ định cực hiếm ("không tốt" 9 mẫu, "không khoẻ" 0 mẫu).
  2. Câu TRỘN (khen mặt này, chê mặt kia) gần như không có nhãn rõ → model đoán
     tích cực quá tự tin (vd "máy đẹp nhưng pin yếu" -> positive 0.79).
  3. Câu TRUNG TÍNH ("tạm được", "ổn trong tầm giá") thiếu -> đoán loạn.

Ta sinh câu nhân tạo từ lexicon để vá cả 3:
  - "<chủ ngữ> không/chẳng <từ tích cực>"         -> tiêu cực (0)
  - "<chủ ngữ> không <từ tiêu cực>"               -> tích cực (2, phủ định kép)
  - "<mệnh đề khen> nhưng <mệnh đề chê>"          -> trung tính (1)
  - "<cụm trung tính>"                            -> trung tính (1)

Chỉ thêm vào tập TRAIN (test giữ 100% thật).
"""

from __future__ import annotations

import random

# ── Phủ định ──────────────────────────────────────────────────────────────────
POS_WORDS = [
    "tốt", "đẹp", "khoẻ", "khỏe", "mượt", "bền", "nhanh", "ngon", "ổn", "ưng",
    "đáng tiền", "đáng", "nét", "sắc nét", "mạnh", "xịn", "rẻ", "hài lòng",
    "ngon lành", "trâu", "êm",
]
NEG_WORDS = [
    "tệ", "kém", "dở", "chậm", "lag", "hỏng", "lỗi", "đắt", "nóng", "yếu", "xấu",
]
NEGATORS = ["không", "chẳng", "chưa", "ko", "đâu có"]
SUBJECTS = [
    "", "máy ", "sản phẩm ", "pin ", "màn hình ", "camera ", "điện thoại ",
    "máy này ", "con này ", "hàng ",
]
SUFFIXES = ["", " lắm", " chút nào", " gì cả", " tí nào", " như mong đợi", " như quảng cáo"]

# ── Mệnh đề cho câu TRỘN (mixed -> trung tính) ────────────────────────────────
POS_CLAUSES = [
    "máy đẹp", "máy tốt", "cấu hình ổn", "camera đẹp", "màn hình đẹp", "pin trâu",
    "máy mượt", "giá rẻ", "thiết kế đẹp", "chụp ảnh nét", "hiệu năng tốt",
    "loa to", "sạc nhanh", "màn hình sắc nét", "máy chạy mượt", "pin khoẻ",
]
NEG_CLAUSES = [
    "pin yếu", "pin tụt nhanh", "hay nóng", "hay lag", "loa nhỏ", "chụp ảnh mờ",
    "mau hết pin", "hơi đắt", "máy nóng", "giật lag", "camera mờ", "sạc chậm",
    "pin không khoẻ", "không bền", "hay đơ", "màn hình hơi tối", "cấu hình yếu",
]
CONTRAST = ["nhưng", "mà", "tuy nhiên", "có điều", "cái tội"]

# ── Cụm TRUNG TÍNH ────────────────────────────────────────────────────────────
NEUTRAL_PHRASES = [
    "tạm được", "cũng được", "bình thường", "bình thường thôi", "tạm ổn",
    "ổn trong tầm giá", "không có gì đặc biệt", "dùng tạm ổn", "cũng bình thường",
    "không quá xuất sắc", "đủ dùng", "ở mức trung bình", "không tốt cũng không tệ",
    "cũng tạm", "không có gì nổi bật", "dùng ổn trong tầm giá", "tạm chấp nhận được",
    "không xuất sắc nhưng đủ dùng", "ổn không có gì để chê nhiều",
]
NEUTRAL_SUBJECTS = ["", "máy ", "sản phẩm ", "máy này ", "nhìn chung "]


def build_negation_examples(n_negative=300, n_positive=100, seed=42):
    rng = random.Random(seed)
    seen, texts, labels = set(), [], []

    def emit(words, label, target):
        added = tries = 0
        while added < target and tries < target * 40:
            tries += 1
            sent = f"{rng.choice(SUBJECTS)}{rng.choice(NEGATORS)} {rng.choice(words)}{rng.choice(SUFFIXES)}".strip()
            if sent in seen:
                continue
            seen.add(sent); texts.append(sent); labels.append(label); added += 1

    emit(POS_WORDS, 0, n_negative)   # "không tốt" -> tiêu cực
    emit(NEG_WORDS, 2, n_positive)   # "không tệ" -> tích cực
    return texts, labels


def build_mixed_examples(n=250, seed=43):
    """Câu khen + chê -> trung tính (1)."""
    rng = random.Random(seed)
    seen, texts, labels = set(), [], []
    tries = 0
    while len(texts) < n and tries < n * 40:
        tries += 1
        pos, neg, con = rng.choice(POS_CLAUSES), rng.choice(NEG_CLAUSES), rng.choice(CONTRAST)
        sent = (f"{pos} {con} {neg}" if rng.random() < 0.8 else f"{pos} {neg}").strip()
        if sent in seen:
            continue
        seen.add(sent); texts.append(sent); labels.append(1)
    return texts, labels


def build_neutral_examples(n=150, seed=44):
    """Cụm trung tính -> trung tính (1)."""
    rng = random.Random(seed)
    seen, texts, labels = set(), [], []
    tries = 0
    while len(texts) < n and tries < n * 40:
        tries += 1
        sent = f"{rng.choice(NEUTRAL_SUBJECTS)}{rng.choice(NEUTRAL_PHRASES)}".strip()
        if sent in seen:
            continue
        seen.add(sent); texts.append(sent); labels.append(1)
    return texts, labels


def build_augmentation():
    """Gộp toàn bộ augment: phủ định + trộn + trung tính."""
    texts, labels = [], []
    for fn in (build_negation_examples, build_mixed_examples, build_neutral_examples):
        t, y = fn()
        texts += t
        labels += y
    return texts, labels


if __name__ == "__main__":
    t, y = build_augmentation()
    from collections import Counter
    print("Tổng:", len(t), "| phân bố nhãn:", dict(Counter(y)))
    for lab in (0, 1, 2):
        ex = [s for s, l in zip(t, y) if l == lab][:4]
        print(f"  label {lab}:", ex)
