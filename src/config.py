
# =============================================================
# config.py — Central configuration for all hyperparameters
# =============================================================

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR    = "/content/facial-emotion-recognition"
DATA_DIR    = "/content/data"
TRAIN_DIR   = os.path.join(DATA_DIR, "train")
TEST_DIR    = os.path.join(DATA_DIR, "test")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")

# ── Classes ──────────────────────────────────────────────────
EMOTIONS    = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTIONS)

# ── Image settings ───────────────────────────────────────────
IMG_SIZE    = (96, 96)        # EfficientNetB0 works well at 96–224px
IMG_SHAPE   = (96, 96, 3)     # 3 channels (FER-2013 is grayscale → will be converted)

# ── Training hyperparameters ─────────────────────────────────
BATCH_SIZE       = 64
EPOCHS           = 50
LEARNING_RATE    = 1e-3       # initial LR for head training
FINE_TUNE_LR     = 1e-5       # LR for fine-tuning phase
VALIDATION_SPLIT = 0.15       # 15% of train set → validation

# ── Callbacks ────────────────────────────────────────────────
EARLY_STOPPING_PATIENCE  = 8
REDUCE_LR_PATIENCE       = 4
REDUCE_LR_FACTOR         = 0.3

# ── Model ────────────────────────────────────────────────────
MODEL_NAME       = "efficientnetb0_fer"
BEST_MODEL_PATH  = os.path.join(MODEL_DIR, f"{MODEL_NAME}_best.keras")

# ── Reproducibility ──────────────────────────────────────────
SEED = 42
