"""
Technique 3 — Embedding-Space SMOTE for Vietnamese text.

Algorithm
---------
1. Encode each minority-class transcript as a mean FastText vector.
2. Run sklearn SMOTE in that (vocab_size,300)-dimensional embedding space
   to generate synthetic vectors.
3. Reconstruct synthetic text: for each synthetic vector, find its two
   nearest original minority samples (by cosine similarity) and produce
   a new transcript by randomly interleaving sentence fragments from both.
   This "fragment interpolation" is more linguistically coherent than
   interpolating raw token sequences.

Why this approach?
- Mirrors the SMOTE strategy used on the Korean dataset
  (Multilingual_BT_approach/SMOTE models for Korean Vishing.ipynb).
- Stays in embedding space for the SMOTE step (numerically stable).
- Reconstruction via fragment mixing preserves Vietnamese grammatical
  structure better than token-level interpolation.

Requires:
    pip install imbalanced-learn scikit-learn
"""
from __future__ import annotations
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from preprocessing.text_cleaner import clean, remove_stopwords
from preprocessing.tokenizer import ViTokenizer


def _mean_embed(
    text: str,
    tokenizer: ViTokenizer,
    embedding_matrix: np.ndarray,
    vocab: Dict[str, int],
    mask_pii: bool = True,
) -> np.ndarray:
    """Encode a single text as its mean FastText token embedding."""
    unk_id = vocab.get("<UNK>", 1)
    cleaned = clean(text, mask_pii=mask_pii)
    tokens  = tokenizer.tokenize(cleaned)
    tokens  = remove_stopwords(tokens)
    ids     = [vocab.get(t, unk_id) for t in tokens]
    if not ids:
        return np.zeros(embedding_matrix.shape[1], dtype=np.float32)
    return embedding_matrix[ids].mean(axis=0)


def _fragment_mix(
    text_a: str, text_b: str, rng: random.Random, min_ratio: float = 0.3
) -> str:
    """
    Construct a synthetic transcript by interleaving sentence fragments
    from two source texts.

    Split both texts at sentence boundaries (. ! ? …) and randomly
    draw fragments, weighted by a random α in [min_ratio, 1 - min_ratio].
    """
    import re
    def split_sentences(t: str) -> List[str]:
        parts = re.split(r"(?<=[.!?…])\s+", t.strip())
        return [p for p in parts if p.strip()]

    sents_a = split_sentences(text_a)
    sents_b = split_sentences(text_b)

    if not sents_a:
        return text_b
    if not sents_b:
        return text_a

    alpha = rng.uniform(min_ratio, 1.0 - min_ratio)
    n_a = max(1, round(alpha * len(sents_a)))
    n_b = max(1, round((1 - alpha) * len(sents_b)))

    chosen_a = rng.sample(sents_a, min(n_a, len(sents_a)))
    chosen_b = rng.sample(sents_b, min(n_b, len(sents_b)))
    combined = chosen_a + chosen_b
    rng.shuffle(combined)
    return " ".join(combined)


class EmbeddingSMOTE:
    """
    SMOTE-based oversampling for Vietnamese text via FastText embeddings.

    Parameters
    ----------
    embedding_matrix : (vocab_size, embed_dim) float32 array
    vocab            : word → index mapping
    tokenizer_backend: 'underthesea' or 'pyvi'
    k_neighbours     : SMOTE k parameter (default 5)
    mask_pii         : whether to mask PII during encoding
    seed             : RNG seed
    """

    def __init__(
        self,
        embedding_matrix: np.ndarray,
        vocab: Dict[str, int],
        tokenizer_backend: str = "underthesea",
        k_neighbours: int = 5,
        mask_pii: bool = True,
        seed: int = 42,
    ):
        self.embedding_matrix = embedding_matrix
        self.vocab = vocab
        self.tokenizer = ViTokenizer(backend=tokenizer_backend)
        self.k_neighbours = k_neighbours
        self.mask_pii = mask_pii
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def _encode_corpus(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts as mean FastText vectors → (N, embed_dim)."""
        return np.stack([
            _mean_embed(t, self.tokenizer, self.embedding_matrix, self.vocab, self.mask_pii)
            for t in texts
        ])

    def _nearest_originals(
        self, synthetic_vec: np.ndarray, original_vecs: np.ndarray, k: int = 2
    ) -> List[int]:
        """Return indices of the k nearest original samples to a synthetic vector."""
        norms_orig = np.linalg.norm(original_vecs, axis=1)
        norms_orig[norms_orig == 0] = 1e-9
        norm_syn = np.linalg.norm(synthetic_vec)
        if norm_syn == 0:
            return list(range(min(k, len(original_vecs))))
        sims = original_vecs.dot(synthetic_vec) / (norms_orig * norm_syn)
        return np.argsort(sims)[-k:][::-1].tolist()

    def fit_resample(
        self,
        texts: List[str],
        labels: List[int],
        minority_label: int = 1,
        strategy: str = "auto",
    ) -> Tuple[List[str], List[int]]:
        """
        Oversample the minority class to match majority class size.

        Parameters
        ----------
        texts          : all transcripts
        labels         : parallel label list (0 or 1)
        minority_label : class to oversample (default 1 = phishing)
        strategy       : 'auto' (balance to majority size) or float ratio

        Returns
        -------
        (augmented_texts, augmented_labels) — original + synthetic samples
        """
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError as e:
            raise ImportError(
                "EmbeddingSMOTE requires imbalanced-learn.\n"
                "Install with: pip install imbalanced-learn"
            ) from e

        labels_arr = np.array(labels)
        minority_mask = labels_arr == minority_label
        majority_mask = ~minority_mask

        minority_texts = [t for t, m in zip(texts, minority_mask) if m]
        majority_count = int(majority_mask.sum())
        minority_count = len(minority_texts)

        if minority_count == 0:
            raise ValueError(f"No samples with label={minority_label} found.")

        target_count = (
            majority_count if strategy == "auto"
            else int(minority_count / strategy)
        )
        n_synthetic = target_count - minority_count

        if n_synthetic <= 0:
            print("Dataset already balanced — no oversampling needed.")
            return texts, labels

        print(f"Minority ({minority_label}): {minority_count} | "
              f"Majority: {majority_count} | "
              f"Generating {n_synthetic} synthetic samples...")

        # ── Step 1: encode minority class ──────────────────────────────────
        minority_vecs = self._encode_corpus(minority_texts)

        # ── Step 2: SMOTE in embedding space ───────────────────────────────
        n_smote_input = len(minority_vecs)
        k = min(self.k_neighbours, n_smote_input - 1)
        if k < 1:
            # Too few minority samples — fall back to random oversampling
            print(f"  Too few minority samples for SMOTE (need ≥2). Using random oversampling.")
            synthetic_texts = [
                self._rng.choice(minority_texts) for _ in range(n_synthetic)
            ]
            return (
                texts + synthetic_texts,
                labels + [minority_label] * n_synthetic,
            )

        # Pad majority class with dummy zeros so SMOTE has a binary problem
        dummy_majority = np.zeros((majority_count, minority_vecs.shape[1]), dtype=np.float32)
        X_smote = np.vstack([minority_vecs, dummy_majority])
        y_smote = np.array(
            [1] * n_smote_input + [0] * majority_count, dtype=np.int64
        )

        smote = SMOTE(
            sampling_strategy={1: target_count},
            k_neighbors=k,
            random_state=42,
        )
        X_res, y_res = smote.fit_resample(X_smote, y_smote)

        # Extract only the newly generated minority vectors
        synthetic_vecs = X_res[y_res == 1][n_smote_input:]  # (n_synthetic, embed_dim)

        # ── Step 3: reconstruct text via fragment interpolation ────────────
        synthetic_texts = []
        for syn_vec in synthetic_vecs:
            top2 = self._nearest_originals(syn_vec, minority_vecs, k=2)
            if len(top2) == 1:
                synthetic_texts.append(minority_texts[top2[0]])
            else:
                mixed = _fragment_mix(
                    minority_texts[top2[0]],
                    minority_texts[top2[1]],
                    self._rng,
                )
                synthetic_texts.append(mixed)

        return (
            texts + synthetic_texts,
            labels + [minority_label] * len(synthetic_texts),
        )
