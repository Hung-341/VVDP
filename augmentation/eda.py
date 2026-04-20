"""
Technique 1 — Easy Data Augmentation (EDA) adapted for Vietnamese.

Operations (Wei & Zou, 2019):
  SR  Synonym Replacement   replace n non-stop words with nearest FastText neighbours
  RI  Random Insertion       insert a synonym of a random non-stop word at a random position
  RS  Random Swap            swap two random tokens n times
  RD  Random Deletion        delete each token with probability p

Vietnamese-specific notes:
  - Tokenisation via Underthesea preserves multi-syllable words (e.g. "ngân_hàng").
  - SR/RI require an embedding matrix + vocab; RS/RD work on raw token lists.
  - All operations respect the VI_STOPWORDS list so function words are not replaced.
"""
from __future__ import annotations
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from preprocessing.text_cleaner import VI_STOPWORDS
from preprocessing.tokenizer import ViTokenizer


def _cosine_neighbours(
    word: str,
    embedding_matrix: np.ndarray,
    vocab: Dict[str, int],
    id2word: List[str],
    top_k: int = 10,
    exclude: Optional[set] = None,
) -> List[str]:
    """Return the top-k vocabulary words most similar to `word` by cosine distance."""
    idx = vocab.get(word)
    if idx is None:
        return []
    vec = embedding_matrix[idx]
    norm = np.linalg.norm(vec)
    if norm == 0:
        return []
    # Vectorised cosine similarity against the whole matrix
    norms = np.linalg.norm(embedding_matrix, axis=1)
    norms[norms == 0] = 1e-9
    sims = embedding_matrix.dot(vec) / (norms * norm)
    sims[idx] = -1                               # exclude the word itself
    top_ids = np.argpartition(sims, -top_k)[-top_k:]
    top_ids = top_ids[np.argsort(sims[top_ids])[::-1]]
    neighbours = [id2word[i] for i in top_ids if id2word[i] not in (exclude or set())]
    return neighbours


class ViEDA:
    """
    Vietnamese Easy Data Augmentation.

    Parameters
    ----------
    backend          : tokeniser backend ('underthesea' or 'pyvi')
    embedding_matrix : FastText (vocab_size, embed_dim) array — required for SR/RI
    vocab            : word → index dict
    alpha            : fraction of tokens to modify (SR/RI/RS); default 0.1
    p_rd             : deletion probability for RD; default 0.1
    top_k            : synonym candidate pool size; default 10
    seed             : RNG seed
    """

    def __init__(
        self,
        backend: str = "underthesea",
        embedding_matrix: Optional[np.ndarray] = None,
        vocab: Optional[Dict[str, int]] = None,
        alpha: float = 0.1,
        p_rd: float = 0.1,
        top_k: int = 10,
        seed: int = 42,
    ):
        self.tokenizer = ViTokenizer(backend=backend)
        self.embedding_matrix = embedding_matrix
        self.vocab = vocab
        self.id2word: List[str] = (
            [w for w, _ in sorted(vocab.items(), key=lambda x: x[1])]
            if vocab else []
        )
        self.alpha = alpha
        self.p_rd = p_rd
        self.top_k = top_k
        self._rng = random.Random(seed)
        self._has_embeddings = embedding_matrix is not None and vocab is not None

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def synonym_replace(self, tokens: List[str], n: Optional[int] = None) -> List[str]:
        """SR: replace n non-stop content words with a random FastText neighbour."""
        if not self._has_embeddings:
            return tokens[:]
        content = [i for i, t in enumerate(tokens) if t not in VI_STOPWORDS]
        if not content:
            return tokens[:]
        n = n or max(1, int(len(tokens) * self.alpha))
        targets = self._rng.sample(content, min(n, len(content)))
        result = tokens[:]
        for i in targets:
            neighbours = _cosine_neighbours(
                tokens[i], self.embedding_matrix, self.vocab, self.id2word,
                top_k=self.top_k, exclude=VI_STOPWORDS,
            )
            if neighbours:
                result[i] = self._rng.choice(neighbours)
        return result

    def random_insert(self, tokens: List[str], n: Optional[int] = None) -> List[str]:
        """RI: insert a synonym of a random content word at a random position."""
        if not self._has_embeddings or not tokens:
            return tokens[:]
        content = [t for t in tokens if t not in VI_STOPWORDS]
        if not content:
            return tokens[:]
        n = n or max(1, int(len(tokens) * self.alpha))
        result = tokens[:]
        for _ in range(n):
            word = self._rng.choice(content)
            neighbours = _cosine_neighbours(
                word, self.embedding_matrix, self.vocab, self.id2word,
                top_k=self.top_k, exclude=VI_STOPWORDS,
            )
            if neighbours:
                synonym = self._rng.choice(neighbours)
                pos = self._rng.randint(0, len(result))
                result.insert(pos, synonym)
        return result

    def random_swap(self, tokens: List[str], n: Optional[int] = None) -> List[str]:
        """RS: randomly swap two tokens n times."""
        if len(tokens) < 2:
            return tokens[:]
        n = n or max(1, int(len(tokens) * self.alpha))
        result = tokens[:]
        for _ in range(n):
            i, j = self._rng.sample(range(len(result)), 2)
            result[i], result[j] = result[j], result[i]
        return result

    def random_delete(self, tokens: List[str]) -> List[str]:
        """RD: delete each token with probability p_rd (keep at least 1 token)."""
        if len(tokens) == 1:
            return tokens[:]
        result = [t for t in tokens if self._rng.random() > self.p_rd]
        return result if result else [self._rng.choice(tokens)]

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def augment(
        self,
        text: str,
        ops: Tuple[str, ...] = ("sr", "ri", "rs", "rd"),
        n_aug: int = 4,
    ) -> List[str]:
        """
        Generate `n_aug` augmented versions of `text`.

        Parameters
        ----------
        text   : raw Vietnamese text (will be tokenised internally)
        ops    : subset of {'sr','ri','rs','rd'} to apply
        n_aug  : number of augmented samples to return

        Returns
        -------
        List of augmented strings (joined tokens, same format as input)
        """
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return [text] * n_aug

        op_map = {
            "sr": self.synonym_replace,
            "ri": self.random_insert,
            "rs": self.random_swap,
            "rd": self.random_delete,
        }
        active = [op_map[o] for o in ops if o in op_map]
        if not active:
            return [text] * n_aug

        results = []
        for _ in range(n_aug):
            op = self._rng.choice(active)
            aug_tokens = op(tokens)
            results.append(" ".join(aug_tokens))
        return results

    def augment_batch(
        self,
        texts: List[str],
        ops: Tuple[str, ...] = ("sr", "ri", "rs", "rd"),
        n_aug: int = 1,
    ) -> List[str]:
        """Augment a list of texts; returns a flat list of len(texts) * n_aug."""
        out = []
        for t in texts:
            out.extend(self.augment(t, ops=ops, n_aug=n_aug))
        return out
