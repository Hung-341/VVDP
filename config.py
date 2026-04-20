from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Paths
    data_path: str = "data/vishing_vi.csv"
    fasttext_path: str = "embeddings/cc.vi.300.vec"   # https://fasttext.cc/docs/en/crawl-vectors.html
    model_save_path: str = "checkpoints/best_model.pt"
    log_dir: str = "logs"

    # Preprocessing
    max_vocab: int = 20_000
    max_length: int = 64      # Vietnamese sentences tend to be longer than Korean
    embed_dim: int = 300      # FastText dimension

    # Architecture  (1D CNN → BiLSTM → HAN)
    spatial_dropout: float = 0.2
    num_filters: int = 50
    kernel_size: int = 3
    pool_size: int = 2
    lstm_units_1: int = 64
    lstm_units_2: int = 32
    dense_units: int = 64
    dropout: float = 0.2
    num_classes: int = 2

    # Training
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-3
    lr_decay: float = 0.9
    lr_decay_steps: int = 10
    early_stop_patience: int = 5
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42
    freeze_embeddings: bool = True

    # Vietnamese NLP backend: "underthesea" | "pyvi"
    tokenizer_backend: str = "underthesea"
