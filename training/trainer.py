"""Training loop with early stopping, LR scheduling, and TensorBoard."""
from __future__ import annotations
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        lr_decay: float = 0.9,
        lr_decay_steps: int = 10,
        epochs: int = 20,
        early_stop_patience: int = 5,
        model_save_path: str = "checkpoints/best_model.pt",
        log_dir: str = "logs",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.patience = early_stop_patience
        self.save_path = Path(model_save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=lr_decay_steps, gamma=lr_decay)
        self.writer = SummaryWriter(log_dir=log_dir)

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict:
        self.model.train(train)
        total_loss, all_preds, all_labels = 0.0, [], []

        with torch.set_grad_enabled(train):
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss = self.criterion(logits, y)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item() * len(y)
                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(y.cpu().tolist())

        n = len(all_labels)
        avg = "binary" if len(set(all_labels)) <= 2 else "macro"
        return {
            "loss": total_loss / n,
            "f1": f1_score(all_labels, all_preds, average=avg, zero_division=0),
            "precision": precision_score(all_labels, all_preds, average=avg, zero_division=0),
            "recall": recall_score(all_labels, all_preds, average=avg, zero_division=0),
        }

    def train(self):
        best_f1, no_improve = 0.0, 0
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_m = self._run_epoch(self.train_loader, train=True)
            val_m = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step()

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:02d}/{self.epochs} [{elapsed:.1f}s] "
                f"train_loss={train_m['loss']:.4f} train_f1={train_m['f1']:.4f} | "
                f"val_loss={val_m['loss']:.4f} val_f1={val_m['f1']:.4f}"
            )

            for tag, m in [("train", train_m), ("val", val_m)]:
                for k, v in m.items():
                    self.writer.add_scalar(f"{tag}/{k}", v, epoch)
            self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], epoch)

            if val_m["f1"] > best_f1:
                best_f1 = val_m["f1"]
                no_improve = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(f"  ✓ Saved best model (val_f1={best_f1:.4f})")
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        self.writer.close()
        print(f"\nTraining complete. Best val F1: {best_f1:.4f}")
        return best_f1

    def evaluate(self, loader: DataLoader) -> dict:
        self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))
        metrics = self._run_epoch(loader, train=False)
        print("Test metrics:", {k: f"{v:.4f}" for k, v in metrics.items()})
        return metrics
