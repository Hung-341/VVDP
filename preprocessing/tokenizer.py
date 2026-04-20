"""Vietnamese word segmentation with pluggable backend."""
from __future__ import annotations
from typing import List


def _build_underthesea():
    from underthesea import word_tokenize

    def tokenize(text: str) -> List[str]:
        return word_tokenize(text, format="text").split()

    return tokenize


def _build_pyvi():
    from pyvi import ViTokenizer

    def tokenize(text: str) -> List[str]:
        return ViTokenizer.tokenize(text).split()

    return tokenize


class ViTokenizer:
    """Wraps Underthesea or PyVi; exposes a single `.tokenize()` method."""

    def __init__(self, backend: str = "underthesea"):
        if backend == "underthesea":
            self._fn = _build_underthesea()
        elif backend == "pyvi":
            self._fn = _build_pyvi()
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'underthesea' or 'pyvi'.")

    def tokenize(self, text: str) -> List[str]:
        text = text.strip().lower()
        return self._fn(text)

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        return [self.tokenize(t) for t in texts]
