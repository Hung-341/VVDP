"""
1D CNN → BiLSTM → HAN Attention → Classifier
Strict sequence as defined in the original Korean VPD architecture.
"""
import numpy as np
import torch
import torch.nn as nn

from models.attention import AttentionWithContext


class CNNBiLSTMHAN(nn.Module):
    """
    Flow (shapes assume batch=B, seq=T, embed=E):

    Embedding        (B, T)      → (B, T, E)
    SpatialDropout   (B, T, E)   → (B, T, E)   [drops entire feature maps]
    Conv1D + Pool    (B, T, E)   → (B, T', F)  [T'=(T-K+1)//P]
    Dropout
    BiLSTM-1         (B, T', F)  → (B, T', 2*L1)
    BiLSTM-2         (B, T', 2*L1) → (B, T', 2*L2)
    HAN Attention    (B, T', 2*L2) → (B, 2*L2)
    Dense + Dropout  (B, 2*L2)   → (B, D)
    Classifier       (B, D)      → (B, num_classes)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        embedding_matrix: np.ndarray | None = None,
        freeze_embeddings: bool = True,
        # CNN
        num_filters: int = 50,
        kernel_size: int = 3,
        pool_size: int = 2,
        spatial_dropout: float = 0.2,
        # BiLSTM
        lstm_units_1: int = 64,
        lstm_units_2: int = 32,
        # Classifier head
        dense_units: int = 64,
        dropout: float = 0.2,
        num_classes: int = 2,
    ):
        super().__init__()

        # ── Embedding ────────────────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(
                torch.tensor(embedding_matrix, dtype=torch.float32),
                requires_grad=not freeze_embeddings,
            )
        self.spatial_dropout = nn.Dropout2d(p=spatial_dropout)

        # ── 1D CNN ───────────────────────────────────────────────────────────
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=0,
        )
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.cnn_dropout = nn.Dropout(p=dropout)

        # ── BiLSTM × 2 ───────────────────────────────────────────────────────
        self.bilstm1 = nn.LSTM(
            input_size=num_filters,
            hidden_size=lstm_units_1,
            batch_first=True,
            bidirectional=True,
        )
        self.bilstm2 = nn.LSTM(
            input_size=lstm_units_1 * 2,
            hidden_size=lstm_units_2,
            batch_first=True,
            bidirectional=True,
        )

        # ── HAN Attention ────────────────────────────────────────────────────
        han_input_dim = lstm_units_2 * 2
        self.attention = AttentionWithContext(hidden_dim=han_input_dim)

        # ── Classifier head ──────────────────────────────────────────────────
        self.fc = nn.Linear(han_input_dim, dense_units)
        self.fc_dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(dense_units, num_classes)

        self._init_lstm_weights()

    def _init_lstm_weights(self):
        for name, p in self.named_parameters():
            if "bilstm" in name:
                if "weight" in name:
                    nn.init.xavier_uniform_(p)
                elif "bias" in name:
                    nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        pad_mask = (x == 0)                              # (B, T) True=padding

        # Embedding + spatial dropout
        emb = self.embedding(x)                          # (B, T, E)
        emb = emb.unsqueeze(1)                           # (B, 1, T, E)  ← 2D dropout needs 4D
        emb = self.spatial_dropout(emb).squeeze(1)       # (B, T, E)

        # Conv1D: needs (B, C_in, L) → permute
        out = emb.permute(0, 2, 1)                       # (B, E, T)
        out = torch.relu(self.conv(out))                 # (B, F, T-K+1)
        out = self.pool(out)                             # (B, F, T')
        out = self.cnn_dropout(out)
        out = out.permute(0, 2, 1)                       # (B, T', F)

        # Adjust pad_mask to match pooled sequence length T'
        # (approximate: trim to T' after conv+pool)
        T_prime = out.size(1)
        pad_mask_pooled = pad_mask[:, :T_prime]          # (B, T')

        # BiLSTM stack
        out, _ = self.bilstm1(out)                       # (B, T', 2*L1)
        out, _ = self.bilstm2(out)                       # (B, T', 2*L2)

        # HAN attention
        out = self.attention(out, mask=pad_mask_pooled)  # (B, 2*L2)

        # Classifier
        out = torch.relu(self.fc(out))                   # (B, D)
        out = self.fc_dropout(out)
        return self.classifier(out)                      # (B, num_classes)
