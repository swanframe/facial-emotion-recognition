
# =============================================================
# train.py — Build and train EfficientNetB0 for FER
# =============================================================

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

from config import (
    TRAIN_DIR, EMOTIONS, NUM_CLASSES, IMG_SHAPE,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, FINE_TUNE_LR,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR,
    BEST_MODEL_PATH, MODEL_DIR, OUTPUT_DIR, SEED
)
from prepare_data import get_datasets


# ── 1. Reproducibility ───────────────────────────────────────
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ── 2. Class weights (handle disgust imbalance) ──────────────
def compute_weights():
    """Compute class weights to handle imbalanced FER-2013 dataset."""
    labels = []
    for idx, emotion in enumerate(EMOTIONS):
        n = len(os.listdir(os.path.join(TRAIN_DIR, emotion)))
        labels.extend([idx] * n)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=np.array(labels)
    )
    class_weight_dict = dict(enumerate(weights))

    print("\n⚖️  Class weights:")
    for idx, emotion in enumerate(EMOTIONS):
        print(f"   {emotion:<12}: {class_weight_dict[idx]:.4f}")

    return class_weight_dict


# ── 3. Model architecture ─────────────────────────────────────
def build_model():
    """
    EfficientNetB0 backbone + custom classification head.
    Two-phase training: head only → fine-tune top layers.
    """
    # Base model — pretrained on ImageNet, top excluded
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False    # freeze for head training phase

    # Custom head
    inputs  = tf.keras.Input(shape=IMG_SHAPE)
    x       = tf.keras.layers.Rescaling(scale=255.0)(inputs)  # [0,1] → [0,255] for EfficientNet
    x       = base_model(x, training=False)
    x       = tf.keras.layers.GlobalAveragePooling2D()(x)
    x       = tf.keras.layers.BatchNormalization()(x)
    x       = tf.keras.layers.Dense(256, activation="relu")(x)
    x       = tf.keras.layers.Dropout(0.4)(x)
    x       = tf.keras.layers.Dense(128, activation="relu")(x)
    x       = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="EfficientNetB0_FER")

    print(f"\n🏗️  Model built:")
    print(f"   Total params    : {model.count_params():,}")
    print(f"   Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    return model, base_model


# ── 4. Callbacks ──────────────────────────────────────────────
def get_callbacks(phase="head"):
    """Return callbacks for training."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint_path = BEST_MODEL_PATH.replace(".keras", f"_{phase}.keras")

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        )
    ]


# ── 5. Plot training curves ───────────────────────────────────
def plot_history(history, phase="head"):
    """Plot and save accuracy & loss curves."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Curves — {phase.capitalize()} Phase", fontsize=13, fontweight="bold")

    metrics = [("accuracy", "Accuracy"), ("loss", "Loss")]
    for ax, (metric, label) in zip(axes, metrics):
        ax.plot(history.history[metric],        label=f"Train {label}", linewidth=2)
        ax.plot(history.history[f"val_{metric}"],label=f"Val {label}",   linewidth=2, linestyle="--")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"training_curves_{phase}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Training curves saved: {save_path}")


# ── 6. Main training pipeline ─────────────────────────────────
def main():
    print("=" * 60)
    print("  Facial Emotion Recognition — Training Pipeline")
    print("=" * 60)

    # Load data
    print("\n📦 Loading datasets...")
    train_ds, val_ds, test_ds = get_datasets()

    # Class weights
    class_weights = compute_weights()

    # Build model
    model, base_model = build_model()

    # ── Phase A: Train head only ──────────────────────────────
    print("\n🔵 Phase A: Training classification head (backbone frozen)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks(phase="head"),
        verbose=1
    )
    plot_history(history_head, phase="head")

    best_val_acc = max(history_head.history["val_accuracy"])
    print(f"\n✅ Phase A complete. Best val accuracy: {best_val_acc:.4f}")

    # ── Phase B: Fine-tune top layers ─────────────────────────
    print("\n🟠 Phase B: Fine-tuning top 30 layers of EfficientNetB0...")

    # Reload best Phase A weights before fine-tuning
    # This ensures Phase B starts from Phase A's best checkpoint,
    # not from a potentially degraded in-memory state
    phase_a_path = BEST_MODEL_PATH.replace(".keras", "_head.keras")
    model = tf.keras.models.load_model(phase_a_path)
    print(f"   ✅ Loaded best Phase A weights from: {phase_a_path}")

    # Re-access base model layer from loaded model
    base_model = model.get_layer("efficientnetb0")
    base_model.trainable = True

    # Freeze all except last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # CRITICAL: Always keep BatchNormalization frozen during fine-tuning.
    # Unfreezing BN layers resets running statistics and destroys
    # features learned in Phase A (causes accuracy collapse).
    frozen_bn = 0
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            frozen_bn += 1
    print(f"   BatchNormalization layers kept frozen: {frozen_bn}")

    trainable_after = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"   Trainable params after unfreeze: {trainable_after:,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks(phase="finetune"),
        verbose=1
    )
    plot_history(history_finetune, phase="finetune")

    best_val_acc_ft = max(history_finetune.history["val_accuracy"])
    print(f"\n✅ Phase B complete. Best val accuracy: {best_val_acc_ft:.4f}")

    # ── Save final model ──────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(BEST_MODEL_PATH)
    print(f"\n💾 Final model saved: {BEST_MODEL_PATH}")
    print("\n✅ train.py completed successfully.")


if __name__ == "__main__":
    main()
