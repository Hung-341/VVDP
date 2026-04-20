"""Builds embedding matrix from a FastText .vec file (cc.vi.300.vec)."""
from __future__ import annotations
from typing import Dict

import codecs
import numpy as np


def load_fasttext(
    vec_path: str,
    vocab: Dict[str, int],
    embed_dim: int = 300,
) -> np.ndarray:
    """
    Returns embedding_matrix of shape (vocab_size, embed_dim).
    OOV tokens keep zero-vector; PAD token stays zero.
    Download: https://fasttext.cc/docs/en/crawl-vectors.html  (cc.vi.300.vec)
    """
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)

    print(f"Loading FastText vectors from {vec_path} ...")
    loaded = 0
    with codecs.open(vec_path, encoding="utf-8") as f:
        next(f)  # skip header line
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab:
                embedding_matrix[vocab[word]] = np.array(parts[1:], dtype=np.float32)
                loaded += 1

    coverage = loaded / max(1, vocab_size - 2)   # exclude PAD, UNK
    print(f"Loaded {loaded}/{vocab_size-2} tokens ({coverage:.1%} coverage)")
    return embedding_matrix
