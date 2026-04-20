"""
Vishing inference wrapper for the mobile demo.

Priority order:
  1. Real trained model (if checkpoint + vocab exist in project root)
  2. Keyword-based heuristic scorer (always available, no dependencies)
"""
from __future__ import annotations
import re
import sys
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Keyword signals ──────────────────────────────────────────────────────────

_HIGH_RISK = [
    # Financial coercion
    r"chuyển tiền",    r"chuyển khoản",   r"nộp tiền",      r"đóng tiền",
    r"tài khoản.*bị", r"bị khóa",        r"bị đóng băng",  r"bị chặn",
    r"mã otp",        r"số tài khoản",   r"số thẻ",        r"mã pin",
    # Authority impersonation
    r"công an",       r"cảnh sát",       r"viện kiểm sát", r"tòa án",
    r"bắt giữ",       r"điều tra",       r"lệnh bắt",      r"rửa tiền",
    r"phong tỏa",
    # Bank impersonation
    r"ngân hàng.*gọi", r"bộ phận bảo mật", r"nhân viên ngân hàng",
    r"vietcombank",   r"vietinbank",     r"techcombank",   r"agribank",
    # Fake prize
    r"trúng thưởng",  r"giải thưởng",    r"phí xử lý",     r"nộp thuế",
    r"500 triệu",     r"1 tỷ",          r"khuyến mãi.*giải",
    # Urgency / secrecy
    r"ngay lập tức",  r"khẩn cấp",       r"không được kể", r"bí mật",
    r"30 phút",       r"24 giờ",         r"hết hạn",
]

_MEDIUM_RISK = [
    r"xác minh",      r"xác nhận thông tin", r"cung cấp",   r"họ tên",
    r"số điện thoại", r"địa chỉ",            r"căn cước",   r"chứng minh",
    r"quý khách",     r"anh.*chị.*vui lòng",
]

_HIGH_PATTERNS  = [re.compile(p, re.IGNORECASE) for p in _HIGH_RISK]
_MED_PATTERNS   = [re.compile(p, re.IGNORECASE) for p in _MEDIUM_RISK]


def _keyword_score(text: str) -> float:
    """Heuristic vishing probability from 0.0 to 1.0."""
    high_hits = sum(1 for p in _HIGH_PATTERNS if p.search(text))
    med_hits  = sum(1 for p in _MED_PATTERNS  if p.search(text))
    raw = high_hits * 0.18 + med_hits * 0.06
    return min(raw, 1.0)


# ── Model loading ────────────────────────────────────────────────────────────

def _try_load_model(project_root: Path):
    """
    Attempt to load the real CNN-BiLSTM-HAN model.
    Returns (model, tokenizer, vocab, embed_matrix, config) or None on failure.
    """
    try:
        import torch
        import pickle
        sys.path.insert(0, str(project_root))
        from config import Config
        from models.cnn_bilstm_han import CNNBiLSTMHAN
        from preprocessing.tokenizer import ViTokenizer

        cfg = Config()
        ckpt_path = project_root / cfg.model_save_path
        vocab_path = project_root / "checkpoints" / "vocab.pkl"
        embed_path = project_root / "checkpoints" / "embed_matrix.npy"

        if not ckpt_path.exists():
            logger.info("No checkpoint found at %s — using keyword mode", ckpt_path)
            return None

        import numpy as np
        vocab        = pickle.load(open(vocab_path, "rb"))
        embed_matrix = np.load(str(embed_path))

        model = CNNBiLSTMHAN(
            vocab_size=len(vocab),
            embed_dim=cfg.embed_dim,
            embedding_matrix=embed_matrix,
            freeze_embeddings=True,
            num_filters=cfg.num_filters,
            kernel_size=cfg.kernel_size,
            pool_size=cfg.pool_size,
            spatial_dropout=cfg.spatial_dropout,
            lstm_units_1=cfg.lstm_units_1,
            lstm_units_2=cfg.lstm_units_2,
            dense_units=cfg.dense_units,
            dropout=cfg.dropout,
            num_classes=cfg.num_classes,
        )
        state = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        tokenizer = ViTokenizer(backend=cfg.tokenizer_backend)
        logger.info("Loaded real model from %s", ckpt_path)
        return model, tokenizer, vocab, embed_matrix, cfg
    except Exception as e:
        logger.warning("Could not load real model (%s) — falling back to keyword mode", e)
        return None


# ── Public detector class ────────────────────────────────────────────────────

class VishingDetector:
    def __init__(self, project_root: Optional[str] = None):
        self._root    = Path(project_root) if project_root else Path(__file__).parent.parent
        self._model_bundle = _try_load_model(self._root)
        self.mode = "model" if self._model_bundle else "keyword"
        logger.info("VishingDetector ready (mode=%s)", self.mode)

    # ------------------------------------------------------------------
    def predict(self, text: str, cumulative_text: str = "") -> dict:
        """
        Predict vishing probability for the given text.

        Parameters
        ----------
        text            : latest transcript chunk
        cumulative_text : full conversation so far (for better context)

        Returns
        -------
        dict with keys:
          probability  : float 0-1
          label        : "Lừa đảo" | "Bình thường"
          confidence   : "Cao" | "Trung bình" | "Thấp"
          mode         : "model" | "keyword"
          signals      : list[str]  keywords found (keyword mode only)
        """
        full = (cumulative_text + " " + text).strip()

        if self.mode == "model":
            prob = self._model_predict(full)
            signals = []
        else:
            prob    = _keyword_score(full)
            signals = [p.pattern for p in _HIGH_PATTERNS if p.search(full)]
            signals = signals[:5]

        label = "Lừa đảo" if prob >= 0.45 else "Bình thường"

        if prob >= 0.70:
            confidence = "Cao"
        elif prob >= 0.40:
            confidence = "Trung bình"
        else:
            confidence = "Thấp"

        return {
            "probability": round(prob, 4),
            "label":       label,
            "confidence":  confidence,
            "mode":        self.mode,
            "signals":     signals,
        }

    # ------------------------------------------------------------------
    def _model_predict(self, text: str) -> float:
        import torch
        import torch.nn.functional as F
        model, tokenizer, vocab, embed_matrix, cfg = self._model_bundle
        from preprocessing.text_cleaner import clean, remove_stopwords

        cleaned = clean(text)
        tokens  = tokenizer.tokenize(cleaned)
        tokens  = remove_stopwords(tokens)

        unk_id = vocab.get("<UNK>", 1)
        ids = [vocab.get(t, unk_id) for t in tokens[:cfg.max_length]]
        if not ids:
            return 0.0

        # Pad
        padded = ids + [0] * (cfg.max_length - len(ids))
        x = torch.tensor([padded], dtype=torch.long)

        with torch.no_grad():
            logits = model(x)
            prob   = F.softmax(logits, dim=-1)[0, 1].item()
        return prob
