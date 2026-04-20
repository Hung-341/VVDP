"""Entry point: load data → build model → train → evaluate."""
import torch
from torch.utils.data import DataLoader

from config import Config
from preprocessing.dataset import load_data
from embeddings.fasttext_loader import load_fasttext
from models.cnn_bilstm_han import CNNBiLSTMHAN
from training.trainer import Trainer


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds, vocab = load_data(
        csv_path=cfg.data_path,
        tokenizer_backend=cfg.tokenizer_backend,
        max_vocab=cfg.max_vocab,
        max_length=cfg.max_length,
        val_split=cfg.val_split,
        test_split=cfg.test_split,
        seed=cfg.seed,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_matrix = load_fasttext(cfg.fasttext_path, vocab, cfg.embed_dim)

    # ── Model ────────────────────────────────────────────────────────────────
    model = CNNBiLSTMHAN(
        vocab_size=len(vocab),
        embed_dim=cfg.embed_dim,
        embedding_matrix=embedding_matrix,
        freeze_embeddings=cfg.freeze_embeddings,
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
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Training ─────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg.lr,
        lr_decay=cfg.lr_decay,
        lr_decay_steps=cfg.lr_decay_steps,
        epochs=cfg.epochs,
        early_stop_patience=cfg.early_stop_patience,
        model_save_path=cfg.model_save_path,
        log_dir=cfg.log_dir,
    )
    trainer.train()
    trainer.evaluate(test_loader)


if __name__ == "__main__":
    main()
