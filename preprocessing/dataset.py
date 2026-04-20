"""Vocabulary building, sequence encoding, and PyTorch Dataset."""
from __future__ import annotations
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocessing.text_cleaner import clean, remove_stopwords, tokens_to_vectors
from preprocessing.tokenizer import ViTokenizer


PAD, UNK = "<PAD>", "<UNK>"


def build_vocab(
    token_seqs: List[List[str]],
    max_vocab: int,
) -> Dict[str, int]:
    counter = Counter(tok for seq in token_seqs for tok in seq)
    vocab = {PAD: 0, UNK: 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def encode(
    token_seqs: List[List[str]],
    vocab: Dict[str, int],
    max_length: int,
) -> np.ndarray:
    """Encode token lists as integer index arrays (for embedding-lookup models)."""
    unk_id = vocab[UNK]
    pad_id = vocab[PAD]
    out = np.full((len(token_seqs), max_length), pad_id, dtype=np.int64)
    for i, seq in enumerate(token_seqs):
        ids = [vocab.get(t, unk_id) for t in seq[:max_length]]
        out[i, : len(ids)] = ids
    return out


def encode_fasttext(
    token_seqs: List[List[str]],
    embedding_matrix: np.ndarray,
    vocab: Dict[str, int],
    max_length: int,
    embed_dim: int = 300,
) -> np.ndarray:
    """
    Encode token lists directly as FastText vector sequences.

    Returns
    -------
    np.ndarray of shape (N, max_length, embed_dim)
    """
    N = len(token_seqs)
    out = np.zeros((N, max_length, embed_dim), dtype=np.float32)
    for i, seq in enumerate(token_seqs):
        out[i] = tokens_to_vectors(seq, embedding_matrix, vocab, max_length, embed_dim, strategy="pad")
    return out


class VishingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X may be int64 (index mode) or float32 (fasttext mode)
        dtype = torch.long if X.dtype == np.int64 else torch.float32
        self.X = torch.tensor(X, dtype=dtype)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _tokenize_corpus(
    texts: List[str],
    tokenizer: ViTokenizer,
    mask_pii: bool,
    stopwords: Optional[frozenset],
) -> List[List[str]]:
    """Clean → tokenize → remove stop words for each text."""
    result = []
    for text in texts:
        cleaned = clean(text, mask_pii=mask_pii)
        tokens  = tokenizer.tokenize(cleaned)
        tokens  = remove_stopwords(tokens, stopwords=stopwords)
        result.append(tokens)
    return result


def load_data(
    csv_path: str,
    text_col: str = "transcript",
    label_col: str = "label",
    tokenizer_backend: str = "underthesea",
    max_vocab: int = 20_000,
    max_length: int = 64,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
    # Cleaning options
    mask_pii: bool = True,
    stopwords: Optional[frozenset] = None,
    # FastText options (pass both to use pre-trained vectors)
    embedding_matrix: Optional[np.ndarray] = None,
    embed_dim: int = 300,
) -> Tuple[VishingDataset, VishingDataset, VishingDataset, Dict[str, int]]:
    """
    Full preprocessing pipeline.

    When `embedding_matrix` is provided the dataset tensors contain float32
    vectors of shape (max_length, embed_dim); otherwise integer indices.
    """
    df = pd.read_csv(csv_path)
    tokenizer = ViTokenizer(backend=tokenizer_backend)

    print("Cleaning and tokenizing...")
    token_seqs = _tokenize_corpus(
        df[text_col].astype(str).tolist(),
        tokenizer,
        mask_pii=mask_pii,
        stopwords=stopwords,
    )
    labels = df[label_col].values.astype(np.int64)

    vocab = build_vocab(token_seqs, max_vocab)

    if embedding_matrix is not None:
        X = encode_fasttext(token_seqs, embedding_matrix, vocab, max_length, embed_dim)
    else:
        X = encode(token_seqs, vocab, max_length)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_split)
    n_val  = int(len(X) * val_split)

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]

    train_ds = VishingDataset(X[train_idx], labels[train_idx])
    val_ds   = VishingDataset(X[val_idx],   labels[val_idx])
    test_ds  = VishingDataset(X[test_idx],  labels[test_idx])

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds, vocab
