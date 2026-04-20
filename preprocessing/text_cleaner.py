"""
Vietnamese text preprocessing pipeline (Section 3.1).

Steps applied in order:
  1. clean()         — strip digits, special characters, sensitive PII patterns
  2. remove_stopwords() — filter Vietnamese stop words
  3. tokens_to_vectors() — look up FastText embeddings, return (N, embed_dim) array
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Vietnamese stop words
# Source: https://github.com/stopwords/vietnamese-stopwords
# Covers common function words, particles, conjunctions, and filler words.
# ---------------------------------------------------------------------------
VI_STOPWORDS: frozenset[str] = frozenset([
    "bị", "bởi", "cả", "các", "cái", "cần", "càng", "chỉ", "chiếc",
    "cho", "chứ", "chưa", "chuyện", "có", "có_thể", "cũng", "của",
    "cùng", "cụ_thể", "đã", "đang", "đây", "để", "đến", "đều", "đi",
    "được", "đó", "gì", "hay", "hơn", "hoặc", "hết", "này", "như",
    "nhưng", "những", "nào", "nếu", "khi", "không", "là", "lại", "lên",
    "lúc", "mà", "mình", "mọi", "một", "mỗi", "nên", "nó", "ra", "rất",
    "rồi", "sau", "sẽ", "thì", "theo", "thêm", "tôi", "tới", "từ", "và",
    "vào", "vì", "với", "vẫn", "vậy", "việc", "về", "xa", "xong",
    # speaker/role tags common in phishing transcripts
    "nạn_nhân", "kẻ_lừa_đảo", "người_gọi", "người_nghe",
])

# ---------------------------------------------------------------------------
# Patterns for sensitive PII commonly found in voice-phishing calls
# ---------------------------------------------------------------------------
_PHONE_RE = re.compile(r"\b0\d{9,10}\b|\+84\d{9,10}\b")
_ID_RE    = re.compile(r"\b\d{9,12}\b")           # national ID / passport numbers
_BANK_RE  = re.compile(r"\b\d{6,20}\b")           # account / card numbers
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_URL_RE   = re.compile(r"https?://\S+|www\.\S+")
_DIGIT_RE = re.compile(r"\d+")
_SPECIAL_RE = re.compile(r"[^\w\sàáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹỵÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỶỸỴ]")


def clean(text: str, mask_pii: bool = True) -> str:
    """
    Normalise a raw Vietnamese transcript line.

    Parameters
    ----------
    text     : raw string (single utterance or full transcript)
    mask_pii : replace phone / ID / account numbers with a <PII> token instead
               of dropping them, so the model sees a signal that sensitive info
               was present.
    """
    text = text.lower().strip()

    # Remove URLs unconditionally — they carry no phishing-relevant semantics
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" <pii> ", text)

    if mask_pii:
        text = _PHONE_RE.sub(" <pii> ", text)
        text = _ID_RE.sub(" <pii> ", text)
        text = _BANK_RE.sub(" <pii> ", text)
    else:
        text = _PHONE_RE.sub(" ", text)
        text = _ID_RE.sub(" ", text)
        text = _BANK_RE.sub(" ", text)

    # Remove all remaining digits
    text = _DIGIT_RE.sub(" ", text)

    # Remove non-Vietnamese characters (punctuation, Latin extras, etc.)
    text = _SPECIAL_RE.sub(" ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(tokens: List[str], stopwords: Optional[frozenset] = None) -> List[str]:
    """
    Filter Vietnamese stop words from an already-tokenised list.

    Parameters
    ----------
    tokens    : output of ViTokenizer.tokenize()
    stopwords : custom stop-word set; defaults to VI_STOPWORDS
    """
    sw = stopwords if stopwords is not None else VI_STOPWORDS
    return [t for t in tokens if t not in sw]


def tokens_to_vectors(
    tokens: List[str],
    embedding_matrix: np.ndarray,
    vocab: Dict[str, int],
    max_length: int,
    embed_dim: int = 300,
    strategy: str = "pad",
) -> np.ndarray:
    """
    Convert a token list to a dense float array using a pre-built FastText
    embedding matrix.

    Parameters
    ----------
    tokens           : tokenised + cleaned token list
    embedding_matrix : (vocab_size, embed_dim) array from load_fasttext()
    vocab            : word → index mapping (must contain "<PAD>" and "<UNK>")
    max_length       : sequence length; truncate or pad to this value
    embed_dim        : FastText vector dimension (default 300)
    strategy         : "pad"  → return (max_length, embed_dim) padded matrix
                       "mean" → return (embed_dim,) mean-pool vector

    Returns
    -------
    np.ndarray of shape (max_length, embed_dim) or (embed_dim,)
    """
    unk_id = vocab.get("<UNK>", 1)
    pad_id = vocab.get("<PAD>", 0)

    ids = [vocab.get(t, unk_id) for t in tokens[:max_length]]

    if strategy == "mean":
        if not ids:
            return np.zeros(embed_dim, dtype=np.float32)
        vecs = embedding_matrix[ids]          # (N, embed_dim)
        return vecs.mean(axis=0)              # (embed_dim,)

    # "pad" strategy: fixed-length matrix
    out = np.zeros((max_length, embed_dim), dtype=np.float32)
    for i, idx in enumerate(ids):
        out[i] = embedding_matrix[idx]
    return out


def preprocess_text(
    text: str,
    tokenizer,                  # ViTokenizer instance
    embedding_matrix: np.ndarray,
    vocab: Dict[str, int],
    max_length: int,
    embed_dim: int = 300,
    mask_pii: bool = True,
    stopwords: Optional[frozenset] = None,
    strategy: str = "pad",
) -> np.ndarray:
    """
    End-to-end pipeline: raw string → dense embedding array.

    1. clean()
    2. tokenize (Underthesea / PyVi)
    3. remove_stopwords()
    4. tokens_to_vectors()
    """
    cleaned   = clean(text, mask_pii=mask_pii)
    tokens    = tokenizer.tokenize(cleaned)
    tokens    = remove_stopwords(tokens, stopwords=stopwords)
    return tokens_to_vectors(tokens, embedding_matrix, vocab, max_length, embed_dim, strategy)
