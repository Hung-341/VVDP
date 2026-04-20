"""
Technique 2 — Back-Translation (VI → pivot → VI).

Pipeline: Helsinki-NLP MarianMT models (no API key needed, runs locally).
  VI → EN → VI  (primary pivot, highest translation quality)
  VI → ZH → VI  (optional pivot, captures different paraphrases)

The Korean project used EN/ZH/JA pivots (Multilingual_BT_approach/).
For Vietnamese we use EN as the default pivot because Helsinki-NLP provides
high-quality opus-mt-vi-en and opus-mt-en-vi checkpoints.

Requires:
    pip install transformers sentencepiece sacremoses
"""
from __future__ import annotations
from typing import List, Optional

PIVOT_CONFIGS = {
    "en": {
        "fwd": "Helsinki-NLP/opus-mt-vi-en",
        "bwd": "Helsinki-NLP/opus-mt-en-vi",
    },
    "zh": {
        "fwd": "Helsinki-NLP/opus-mt-vi-zh",
        "bwd": "Helsinki-NLP/opus-mt-zh-vi",
    },
}


class BackTranslator:
    """
    Back-translate Vietnamese text through a pivot language.

    Parameters
    ----------
    pivot      : 'en' (default) or 'zh'
    device     : 'cpu' or 'cuda'
    batch_size : number of sentences per translation call
    max_length : max token length passed to the translation model
    """

    def __init__(
        self,
        pivot: str = "en",
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 256,
    ):
        if pivot not in PIVOT_CONFIGS:
            raise ValueError(f"Pivot '{pivot}' not supported. Choose from {list(PIVOT_CONFIGS)}")

        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError as e:
            raise ImportError(
                "Back-translation requires the 'transformers' and 'sentencepiece' packages.\n"
                "Install with: pip install transformers sentencepiece sacremoses"
            ) from e

        cfg = PIVOT_CONFIGS[pivot]
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        print(f"Loading forward model ({cfg['fwd']})...")
        self._fwd_tok = MarianTokenizer.from_pretrained(cfg["fwd"])
        self._fwd_model = MarianMTModel.from_pretrained(cfg["fwd"]).to(device)

        print(f"Loading backward model ({cfg['bwd']})...")
        self._bwd_tok = MarianTokenizer.from_pretrained(cfg["bwd"])
        self._bwd_model = MarianMTModel.from_pretrained(cfg["bwd"]).to(device)

        self._fwd_model.eval()
        self._bwd_model.eval()

    def _translate(self, texts: List[str], tokenizer, model) -> List[str]:
        import torch
        outputs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            with torch.no_grad():
                translated = model.generate(**encoded, max_length=self.max_length)
            decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
            outputs.extend(decoded)
        return outputs

    def translate(self, texts: List[str]) -> List[str]:
        """
        Back-translate a list of Vietnamese strings.

        Steps: VI → pivot language → VI

        Returns
        -------
        List of back-translated Vietnamese strings (same length as input).
        """
        pivot_texts = self._translate(texts, self._fwd_tok, self._fwd_model)
        back_texts  = self._translate(pivot_texts, self._bwd_tok, self._bwd_model)
        return back_texts

    def augment_batch(self, texts: List[str]) -> List[str]:
        """Alias for translate(); matches the API used by balance_dataset.py."""
        return self.translate(texts)
