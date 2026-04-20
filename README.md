# VVDP — Vietnamese Vishing Detection Project

A deep learning system for detecting phone fraud (vishing) in Vietnamese conversations. Uses a CNN-BiLSTM-HAN architecture trained on Vietnamese text with FastText embeddings.

---

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Configuration](#configuration)
4. [Data Preparation & Preprocessing](#data-preparation--preprocessing)
5. [Data Augmentation](#data-augmentation)
6. [Training](#training)
7. [Testing & Evaluation](#testing--evaluation)
8. [Adding New Scenarios](#adding-new-scenarios)
9. [Model Performance Check](#model-performance-check)
10. [Mobile Demo](#mobile-demo)

---

## Installation

**Requirements:** Python 3.8+, CUDA-capable GPU (optional but recommended)

```bash
# Clone the repository
git clone <repo-url>
cd VVDP

# Install dependencies
pip install -r requirements.txt

# Download FastText Vietnamese embeddings (cc.vi.300.vec, ~1.9 GB)
# Place the file at the path set in config.py (default: embeddings/cc.vi.300.vec)
```

For the mobile demo only:

```bash
pip install -r mobile_demo/requirements_demo.txt
```

---

## Project Structure

```
VVDP/
├── config.py                  # All hyperparameters and paths
├── train.py                   # Main training script
├── requirements.txt
├── preprocessing/
│   ├── dataset.py             # Vocabulary, encoding, PyTorch Dataset
│   ├── text_cleaner.py        # PII masking, stopword removal, normalization
│   └── tokenizer.py           # Vietnamese word segmentation (underthesea / pyvi)
├── embeddings/
│   └── fasttext_loader.py     # Load FastText cc.vi.300.vec
├── models/
│   ├── cnn_bilstm_han.py      # CNN → BiLSTM → HAN architecture
│   └── attention.py           # Hierarchical Attention Network module
├── training/
│   └── trainer.py             # Training loop, early stopping, TensorBoard
├── augmentation/
│   ├── eda.py                 # Easy Data Augmentation (SR / RI / RS / RD)
│   ├── back_translation.py    # Back-translation via MarianMT
│   ├── embedding_smote.py     # SMOTE in FastText embedding space
│   └── balance_dataset.py     # CLI tool to apply augmentation
└── mobile_demo/
    ├── app.py                 # Flask REST API
    ├── inference.py           # Model or keyword-based detector
    └── templates/index.html   # Demo frontend
```

---

## Configuration

All settings live in `config.py` as a Python dataclass. Edit this file before training.

```python
# config.py (key fields)

@dataclass
class Config:
    # --- Paths ---
    data_path: str = "data/vishing_vi.csv"          # Training CSV
    fasttext_path: str = "embeddings/cc.vi.300.vec"  # Pre-trained vectors
    model_path: str = "models/best_model.pt"         # Saved checkpoint
    log_dir: str = "logs/"                            # TensorBoard logs

    # --- Preprocessing ---
    max_vocab: int = 20000      # Maximum vocabulary size
    max_length: int = 64        # Token sequence length (pad/truncate)
    embed_dim: int = 300        # Embedding dimension (must match FastText)

    # --- Model Architecture ---
    cnn_filters: int = 50       # Number of CNN feature maps
    cnn_kernel: int = 3         # CNN kernel window size
    cnn_pool: int = 2           # Max-pool size after CNN
    lstm_units: int = 64        # BiLSTM-1 hidden units (BiLSTM-2 = lstm_units//2)
    dense_units: int = 64       # Dense layer size before classifier

    # --- Training ---
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    lr_decay: float = 0.9       # StepLR gamma
    lr_step: int = 10           # Decay every N epochs
    early_stop_patience: int = 5
    dropout: float = 0.3
    freeze_embeddings: bool = False  # Freeze FastText weights during training

    # --- Data Split ---
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # --- Vietnamese NLP ---
    tokenizer_backend: str = "underthesea"  # or "pyvi"
```

**Common changes before training:**

| Goal | Field to change |
|------|----------------|
| Use your own dataset | `data_path` |
| Longer/shorter conversations | `max_length` |
| Faster training (less accurate) | `lstm_units`, `cnn_filters` |
| Reduce overfitting | increase `dropout`, reduce `epochs` |
| Keep embeddings fixed | `freeze_embeddings = True` |
| Use pyvi instead of underthesea | `tokenizer_backend = "pyvi"` |

---

## Data Preparation & Preprocessing

### CSV Format

Your dataset must be a CSV file with at least two columns:

```csv
text,label
"Xin chào, tôi gọi từ ngân hàng Vietcombank...",1
"Bạn có muốn đặt hàng không?",0
```

| Column | Description |
|--------|-------------|
| `text` | Raw conversation text in Vietnamese |
| `label` | `1` = vishing (fraud), `0` = normal |

Set the path in `config.py`:

```python
data_path = "data/your_dataset.csv"
```

### Preprocessing Pipeline

The pipeline runs automatically when you call `train.py`. Steps applied to every text sample:

1. **Lowercase & strip whitespace**
2. **Remove URLs**
3. **Mask PII** — emails, phone numbers, ID numbers, bank accounts are replaced or removed
4. **Remove digits**
5. **Remove non-Vietnamese punctuation**
6. **Word segmentation** — via Underthesea or PyVi (preserves multi-syllable words like `ngân_hàng`)
7. **Stopword removal** — 149 Vietnamese function words and particles are filtered out
8. **Vocabulary encoding** — tokens mapped to integer indices (PAD=0, UNK=1)
9. **Sequence padding/truncation** to `max_length`

To test preprocessing on a single text:

```python
from config import Config
from preprocessing.text_cleaner import preprocess_text
from preprocessing.tokenizer import ViTokenizer

cfg = Config()
tokenizer = ViTokenizer(backend=cfg.tokenizer_backend)
tokens = preprocess_text("Tôi cần xác minh tài khoản ngân hàng của bạn ngay.", tokenizer)
print(tokens)
```

### Adding Your Own Text Cleaning Rules

Edit `preprocessing/text_cleaner.py`:

- Add regex patterns in the `clean_text()` function
- Add domain-specific stopwords to the `VI_STOPWORDS` set
- Add new PII patterns alongside the existing phone/ID/bank regexes

---

## Data Augmentation

Use augmentation to balance an imbalanced dataset (typically more normal samples than vishing). Augmentation is applied **only to minority-class (vishing) samples**.

### Run Augmentation

```bash
# Apply all three techniques
python -m augmentation.balance_dataset \
    --csv data/vishing_vi.csv \
    --output data/vishing_vi_balanced.csv \
    --technique eda bt smote \
    --target_ratio 0.4
```

| Argument | Description |
|----------|-------------|
| `--csv` | Input CSV file |
| `--output` | Output balanced CSV file |
| `--technique` | One or more of: `eda`, `bt`, `smote` |
| `--target_ratio` | Target minority class ratio (0.0–0.5) |

### Augmentation Techniques

**EDA (Easy Data Augmentation)** — fast, no GPU needed

Applies four token-level operations to existing vishing samples:
- **SR** — Synonym Replacement: swap a word with its FastText nearest neighbor
- **RI** — Random Insertion: insert a synonym at a random position
- **RS** — Random Swap: swap two tokens
- **RD** — Random Deletion: drop tokens with a fixed probability

**Back-Translation** — higher quality, GPU recommended

Translates VI → EN → VI (or VI → ZH → VI) using local MarianMT models. Produces natural paraphrases.

```python
from augmentation.back_translation import BackTranslator
bt = BackTranslator(pivot="en")  # or pivot="zh"
augmented = bt.augment(["Bạn cần chuyển tiền ngay bây giờ."])
```

**Embedding SMOTE** — interpolates in semantic space

1. Encodes texts as mean FastText vectors
2. Generates synthetic vectors with SMOTE
3. Reconstructs text by interpolating fragments from nearest originals

```python
from augmentation.embedding_smote import EmbeddingSmote
smote = EmbeddingSmote(fasttext_path=cfg.fasttext_path)
synthetic_texts = smote.augment(minority_texts, n_synthetic=100)
```

---

## Training

### Quick Start

```bash
python train.py
```

This will:
1. Load and preprocess the CSV at `config.data_path`
2. Build vocabulary and encode sequences
3. Load FastText embeddings
4. Train the CNN-BiLSTM-HAN model
5. Save the best checkpoint (by validation F1) to `config.model_path`
6. Print test-set metrics

### Monitor Training

```bash
tensorboard --logdir logs/
# Open http://localhost:6006
```

TensorBoard tracks: train/val loss, F1, precision, recall per epoch.

### Training Behavior

- **Early stopping**: Training halts if validation F1 does not improve for `early_stop_patience` epochs (default: 5).
- **Best checkpoint**: The model with the highest validation F1 is saved; the final epoch weights are not used.
- **Learning rate decay**: LR is multiplied by `lr_decay` every `lr_step` epochs.
- **Gradient clipping**: Norm clipped at 1.0 to prevent exploding gradients.

### Resuming / Fine-tuning

Load a checkpoint in `train.py` or your own script:

```python
import torch
from models.cnn_bilstm_han import CNNBiLSTMHAN
from config import Config

cfg = Config()
model = CNNBiLSTMHAN(vocab_size, cfg)
model.load_state_dict(torch.load(cfg.model_path))
```

---

## Testing & Evaluation

### Automated Test Split

The test split (default 10%) is evaluated automatically at the end of `train.py`. Metrics printed:

- **Loss**
- **F1 score** (binary)
- **Precision**
- **Recall**

### Manual Evaluation on a Custom CSV

```python
from config import Config
from preprocessing.dataset import load_data
from training.trainer import Trainer
from models.cnn_bilstm_han import CNNBiLSTMHAN
import torch

cfg = Config()
cfg.data_path = "data/test_set.csv"   # point to your test file

_, _, test_ds, vocab = load_data(cfg)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=cfg.batch_size)

model = CNNBiLSTMHAN(len(vocab), cfg)
model.load_state_dict(torch.load(cfg.model_path))

trainer = Trainer(model, cfg)
metrics = trainer.evaluate(test_loader)
print(metrics)
```

### Single-Text Prediction

```python
from mobile_demo.inference import VishingDetector
from config import Config

cfg = Config()
detector = VishingDetector(cfg)

result = detector.predict("Bạn cần chuyển 50 triệu ngay hôm nay để tránh bị phong tỏa tài khoản.")
print(result)
# {'label': 'Lừa đảo', 'probability': 0.87, 'confidence': 'Cao'}
```

The detector falls back to keyword-based scoring if no model checkpoint is found.

---

## Adding New Scenarios

Scenarios are demo conversation scripts shown in the mobile demo. They are loaded by `mobile_demo/demo_scripts.py`.

### Step 1 — Write the Scenario

A scenario is a dictionary with the following fields:

```python
{
    "id": "scenario_004",
    "title": "Giả mạo bảo hiểm xã hội",
    "category": "insurance_fraud",          # group tag (used for filtering)
    "label": 1,                             # 1 = vishing, 0 = normal
    "turns": [
        {"speaker": "caller",  "text": "Tôi gọi từ Bảo hiểm xã hội..."},
        {"speaker": "victim",  "text": "Vâng, tôi nghe."},
        {"speaker": "caller",  "text": "Bạn cần nộp tiền phạt 2 triệu..."},
    ]
}
```

### Step 2 — Add to demo_scripts.py

Open `mobile_demo/demo_scripts.py` and append your scenario dict to the `SCENARIOS` list.

### Step 3 — Test via the API

```bash
# Start the demo server
python mobile_demo/app.py

# List all scenarios
curl http://localhost:5000/api/scenarios

# Get your new scenario
curl http://localhost:5000/api/scenario/scenario_004

# Run inference on it
curl -X POST http://localhost:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Tôi gọi từ Bảo hiểm xã hội..."}'
```

### Step 4 — Add to Training Data (optional)

Extract the conversation text and add rows to your CSV, then re-run augmentation and training:

```csv
"Tôi gọi từ Bảo hiểm xã hội... Bạn cần nộp tiền phạt 2 triệu...",1
```

---

## Model Performance Check

### During Training

TensorBoard (`tensorboard --logdir logs/`) shows per-epoch curves for all metrics.

### After Training — Classification Report

```python
from sklearn.metrics import classification_report
import torch
from config import Config
from preprocessing.dataset import load_data
from models.cnn_bilstm_han import CNNBiLSTMHAN

cfg = Config()
_, _, test_ds, vocab = load_data(cfg)
loader = torch.utils.data.DataLoader(test_ds, batch_size=cfg.batch_size)

model = CNNBiLSTMHAN(len(vocab), cfg)
model.load_state_dict(torch.load(cfg.model_path))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for X, y in loader:
        logits = model(X)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())

print(classification_report(all_labels, all_preds, target_names=["Normal", "Vishing"]))
```

### Confusion Matrix

```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_predictions(all_labels, all_preds,
                                        display_labels=["Normal", "Vishing"])
plt.savefig("confusion_matrix.png")
```

### Key Metrics to Watch

| Metric | What it tells you | Target |
|--------|------------------|--------|
| **F1 (vishing class)** | Balance of precision & recall for fraud detection | > 0.85 |
| **Recall (vishing)** | How many actual frauds are caught | Maximize — missing fraud is costly |
| **Precision (vishing)** | How many flagged calls are truly fraud | Keep > 0.70 to avoid false alarms |
| **Val F1 plateau** | Overfitting — reduce epochs or increase dropout | — |

### Diagnosing Poor Performance

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| High train F1, low val F1 | Overfitting | Increase `dropout`, reduce `lstm_units`, add more data |
| Low recall on vishing | Class imbalance | Run `balance_dataset.py` with `--technique eda bt smote` |
| Model not improving | LR too high/low | Tune `learning_rate` in config |
| Early stopping triggers too soon | Patience too low | Increase `early_stop_patience` |

---

## Mobile Demo

```bash
cd mobile_demo
python app.py
# Open http://localhost:5000
```

The demo works without a trained model (keyword-based fallback). For full model inference, ensure `config.model_path` points to a trained checkpoint.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web frontend |
| GET | `/api/health` | Server status |
| GET | `/api/scenarios` | List all demo scenarios |
| GET | `/api/scenario/<id>` | Get scenario details |
| POST | `/api/predict` | Classify a text snippet |

**POST /api/predict** request body:

```json
{ "text": "Bạn cần chuyển tiền ngay để tránh bị bắt." }
```

Response:

```json
{
  "label": "Lừa đảo",
  "probability": 0.91,
  "confidence": "Cao"
}
```

| `label` | Meaning |
|---------|---------|
| `Lừa đảo` | Vishing detected |
| `Bình thường` | Normal call |

| `confidence` | Probability range |
|-------------|------------------|
| `Cao` | >= 0.70 |
| `Trung bình` | 0.40 – 0.70 |
| `Thấp` | < 0.40 |
