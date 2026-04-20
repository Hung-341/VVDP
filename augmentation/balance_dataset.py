"""
balance_dataset.py — Data-centric balancing for the Vietnamese VPD corpus.

Section 4.1 strategy: three complementary augmentation techniques are applied
exclusively on the minority (phishing) class to address class imbalance.

Techniques
----------
  eda    Easy Data Augmentation  (SR, RI, RS, RD on token sequences)
  bt     Back-Translation        (VI → EN → VI via MarianMT)
  smote  Embedding-space SMOTE   (FastText mean embeddings + fragment mixing)

Usage examples
--------------
# EDA only (no internet, no embeddings required):
python -m augmentation.balance_dataset \\
    --csv data/vishing_vi.csv \\
    --technique eda \\
    --output data/vishing_vi_balanced.csv

# Back-translation (downloads Helsinki-NLP models on first run):
python -m augmentation.balance_dataset \\
    --csv data/vishing_vi.csv \\
    --technique bt \\
    --bt_pivot en \\
    --output data/vishing_vi_bt.csv

# SMOTE with pre-trained FastText vectors:
python -m augmentation.balance_dataset \\
    --csv data/vishing_vi.csv \\
    --technique smote \\
    --fasttext_path embeddings/cc.vi.300.vec \\
    --output data/vishing_vi_smote.csv

# Combine all three:
python -m augmentation.balance_dataset \\
    --csv data/vishing_vi.csv \\
    --technique eda bt smote \\
    --fasttext_path embeddings/cc.vi.300.vec \\
    --output data/vishing_vi_augmented.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


# ── Helpers ──────────────────────────────────────────────────────────────────

def _print_distribution(labels: List[int], title: str = "") -> None:
    from collections import Counter
    c = Counter(labels)
    total = len(labels)
    print(f"\n{title}")
    for lbl in sorted(c):
        name = "phishing" if lbl == 1 else "non-phishing"
        print(f"  label={lbl} ({name}): {c[lbl]:>6}  ({c[lbl]/total:.1%})")
    print(f"  total: {total}")


def _load_csv(path: str, text_col: str, label_col: str):
    df = pd.read_csv(path)
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Columns not found in CSV: {missing}\nAvailable: {list(df.columns)}")
    df[label_col] = df[label_col].astype(int)
    return df


def _resolve_target_count(
    minority_count: int, majority_count: int, ratio: float
) -> int:
    """Return how many total minority samples we want after augmentation."""
    if ratio <= 0:
        return majority_count                            # full balance
    return min(majority_count, int(minority_count / ratio))


# ── Technique runners ─────────────────────────────────────────────────────────

def _run_eda(
    minority_texts: List[str],
    n_needed: int,
    backend: str,
    embedding_matrix,
    vocab,
    seed: int,
) -> List[str]:
    from augmentation.eda import ViEDA
    eda = ViEDA(
        backend=backend,
        embedding_matrix=embedding_matrix,
        vocab=vocab,
        seed=seed,
    )
    ops = ("sr", "ri", "rs", "rd") if embedding_matrix is not None else ("rs", "rd")
    if embedding_matrix is None:
        print("  [EDA] No embeddings provided — SR/RI disabled; using RS+RD only.")

    synthetic: List[str] = []
    i = 0
    while len(synthetic) < n_needed:
        text = minority_texts[i % len(minority_texts)]
        synthetic.extend(eda.augment(text, ops=ops, n_aug=1))
        i += 1
    return synthetic[:n_needed]


def _run_bt(
    minority_texts: List[str],
    n_needed: int,
    pivot: str,
    device: str,
    batch_size: int,
) -> List[str]:
    from augmentation.back_translation import BackTranslator
    bt = BackTranslator(pivot=pivot, device=device, batch_size=batch_size)

    # Cycle through minority texts until we have enough
    source_pool: List[str] = []
    while len(source_pool) < n_needed:
        source_pool.extend(minority_texts)
    source_pool = source_pool[:n_needed]

    print(f"  [BT]  Translating {len(source_pool)} samples (pivot={pivot})...")
    return bt.augment_batch(source_pool)


def _run_smote(
    all_texts: List[str],
    all_labels: List[int],
    embedding_matrix: np.ndarray,
    vocab: dict,
    backend: str,
    minority_label: int,
    seed: int,
) -> tuple:
    from augmentation.embedding_smote import EmbeddingSMOTE
    smote = EmbeddingSMOTE(
        embedding_matrix=embedding_matrix,
        vocab=vocab,
        tokenizer_backend=backend,
        seed=seed,
    )
    return smote.fit_resample(all_texts, all_labels, minority_label=minority_label)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Balance the Vietnamese VPD dataset using text augmentation (Section 4.1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv",         required=True,  help="Input CSV path")
    parser.add_argument("--output",      required=True,  help="Output CSV path")
    parser.add_argument("--text_col",    default="transcript", help="Text column name")
    parser.add_argument("--label_col",   default="label",      help="Label column name")
    parser.add_argument("--minority_label", type=int, default=1,
                        help="Label value of the minority class (default: 1 = phishing)")

    parser.add_argument(
        "--technique", nargs="+",
        choices=["eda", "bt", "smote"],
        default=["eda"],
        help="Augmentation technique(s) to apply. Multiple values are combined.",
    )
    parser.add_argument("--ratio", type=float, default=0.0,
                        help="Target minority/majority ratio after balancing (0 = full 1:1 balance).")

    # EDA / shared
    parser.add_argument("--tokenizer_backend", default="underthesea",
                        choices=["underthesea", "pyvi"])
    parser.add_argument("--fasttext_path", default=None,
                        help="Path to cc.vi.300.vec (enables SR/RI in EDA and SMOTE)")
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--max_vocab",  type=int, default=20_000)

    # Back-translation
    parser.add_argument("--bt_pivot",  default="en", choices=["en", "zh"])
    parser.add_argument("--bt_device", default="cpu")
    parser.add_argument("--bt_batch",  type=int, default=16)

    # SMOTE
    parser.add_argument("--smote_k", type=int, default=5)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)

    # ── Load ──────────────────────────────────────────────────────────────
    df = _load_csv(args.csv, args.text_col, args.label_col)
    texts  = df[args.text_col].astype(str).tolist()
    labels = df[args.label_col].tolist()
    _print_distribution(labels, "Original distribution:")

    minority_mask  = [l == args.minority_label for l in labels]
    minority_texts = [t for t, m in zip(texts, minority_mask) if m]
    majority_count = sum(1 for m in minority_mask if not m)
    minority_count = len(minority_texts)

    n_needed = _resolve_target_count(minority_count, majority_count, args.ratio) - minority_count
    if n_needed <= 0:
        print("\nDataset is already balanced. Nothing to do.")
        df.to_csv(args.output, index=False)
        return

    print(f"\nTarget: generate {n_needed} synthetic phishing samples.")

    # ── Load FastText embeddings if required ─────────────────────────────
    embedding_matrix = None
    vocab = None
    needs_embeddings = ("smote" in args.technique or
                        ("eda" in args.technique and args.fasttext_path))

    if needs_embeddings:
        if not args.fasttext_path:
            sys.exit(
                "[ERROR] --fasttext_path is required for 'smote' technique "
                "and for EDA synonym operations.\n"
                "Download cc.vi.300.vec from https://fasttext.cc/docs/en/crawl-vectors.html"
            )
        from preprocessing.dataset import build_vocab
        from preprocessing.text_cleaner import clean, remove_stopwords
        from preprocessing.tokenizer import ViTokenizer
        from embeddings.fasttext_loader import load_fasttext

        print("\nBuilding vocabulary from corpus...")
        tok = ViTokenizer(backend=args.tokenizer_backend)
        token_seqs = [
            remove_stopwords(tok.tokenize(clean(t)))
            for t in texts
        ]
        vocab = build_vocab(token_seqs, args.max_vocab)
        embedding_matrix = load_fasttext(args.fasttext_path, vocab, args.embed_dim)

    # ── Apply technique(s) ────────────────────────────────────────────────
    synthetic_texts: List[str] = []
    techniques = args.technique

    # When combining, split the budget evenly across techniques
    per_technique = n_needed // len(techniques)
    remainder     = n_needed - per_technique * len(techniques)

    for i, tech in enumerate(techniques):
        quota = per_technique + (1 if i < remainder else 0)
        print(f"\n[{tech.upper()}] Generating {quota} samples...")

        if tech == "eda":
            batch = _run_eda(
                minority_texts, quota,
                backend=args.tokenizer_backend,
                embedding_matrix=embedding_matrix,
                vocab=vocab,
                seed=args.seed,
            )
            synthetic_texts.extend(batch)

        elif tech == "bt":
            batch = _run_bt(
                minority_texts, quota,
                pivot=args.bt_pivot,
                device=args.bt_device,
                batch_size=args.bt_batch,
            )
            synthetic_texts.extend(batch)

        elif tech == "smote":
            if embedding_matrix is None:
                sys.exit("[ERROR] SMOTE requires --fasttext_path.")
            # SMOTE resamples the whole dataset at once — run it separately
            aug_texts, aug_labels = _run_smote(
                texts, labels,
                embedding_matrix=embedding_matrix,
                vocab=vocab,
                backend=args.tokenizer_backend,
                minority_label=args.minority_label,
                seed=args.seed,
            )
            # Extract only the new synthetic samples (appended at end)
            smote_synthetic = aug_texts[len(texts):]
            synthetic_texts.extend(smote_synthetic[:quota])

    # ── Assemble output ───────────────────────────────────────────────────
    aug_df = pd.DataFrame({
        args.text_col:  synthetic_texts,
        args.label_col: [args.minority_label] * len(synthetic_texts),
    })
    # Carry over any extra columns from original (set to NaN for synthetics)
    extra_cols = [c for c in df.columns if c not in {args.text_col, args.label_col}]
    for col in extra_cols:
        aug_df[col] = np.nan

    final_df = pd.concat([df, aug_df[df.columns]], ignore_index=True)

    _print_distribution(final_df[args.label_col].tolist(), "Balanced distribution:")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
