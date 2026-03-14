
# =============================================================
# evaluate.py — Confusion matrix, metrics, and Grad-CAM
# =============================================================

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    TEST_DIR, EMOTIONS, NUM_CLASSES,
    BEST_MODEL_PATH, OUTPUT_DIR, SEED, IMG_SIZE, IMG_SHAPE
)
from prepare_data import get_datasets


# ── 1. Load model & test data ─────────────────────────────────
def load_model_and_data():
    """Load best model and test dataset."""
    print("📦 Loading model and test data...")
    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    print(f"   ✅ Model loaded: {BEST_MODEL_PATH}")

    _, _, test_ds = get_datasets()
    return model, test_ds


# ── 2. Generate predictions ───────────────────────────────────
def get_predictions(model, test_ds):
    """Run inference on full test set."""
    print("\n🔍 Running predictions on test set...")
    y_true, y_pred = [], []

    for images, labels in test_ds:
        preds   = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    return np.array(y_true), np.array(y_pred)


# ── 3. Confusion matrix ───────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred):
    """Plot and save normalized confusion matrix."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=EMOTIONS, yticklabels=EMOTIONS,
        linewidths=0.5, ax=ax
    )
    ax.set_title("Confusion Matrix (Normalized) — FER Test Set",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Confusion matrix saved: {save_path}")


# ── 4. Classification report ──────────────────────────────────
def print_classification_report(y_true, y_pred):
    """Print per-class precision, recall, F1."""
    print("\n📊 Classification Report:")
    print("=" * 60)
    report = classification_report(
        y_true, y_pred,
        target_names=EMOTIONS,
        digits=4
    )
    print(report)

    # Save to file
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Classification Report — FER-2013 Test Set\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    print(f"✅ Report saved: {report_path}")

    # Highlight weakest class
    from sklearn.metrics import f1_score
    f1_scores = f1_score(y_true, y_pred, average=None)
    weakest_idx = np.argmin(f1_scores)
    print(f"\n⚠️  Weakest class: {EMOTIONS[weakest_idx]} "
          f"(F1 = {f1_scores[weakest_idx]:.4f})")
    print(f"✅ Strongest class: {EMOTIONS[np.argmax(f1_scores)]} "
          f"(F1 = {f1_scores[np.argmax(f1_scores)]:.4f})")


# ── 5. Grad-CAM ───────────────────────────────────────────────
def build_grad_model(model):
    """
    Build a connected Grad-CAM model that outputs both
    top_conv feature maps and final predictions in ONE forward pass.
    Both outputs share the same graph so gradients can flow.
    """
    effnet          = model.get_layer("efficientnetb0")
    effnet_partial  = tf.keras.Model(
        inputs=effnet.input,
        outputs=[effnet.get_layer("top_conv").output, effnet.output]
    )

    model_input          = model.input
    rescaling_layer      = next(l for l in model.layers
                                if isinstance(l, tf.keras.layers.Rescaling))
    rescaled             = rescaling_layer(model_input)
    conv_out, effnet_out = effnet_partial(rescaled)

    x           = model.get_layer("global_average_pooling2d")(effnet_out)
    x           = model.get_layer("batch_normalization")(x)
    x           = model.get_layer("dense")(x)
    x           = model.get_layer("dropout")(x)
    x           = model.get_layer("dense_1")(x)
    x           = model.get_layer("dropout_1")(x)
    final_pred  = model.get_layer("dense_2")(x)

    return tf.keras.Model(inputs=model_input, outputs=[conv_out, final_pred])


def make_gradcam_heatmap(img_array, grad_model):
    """
    Compute Grad-CAM heatmap for a given image using a pre-built grad_model.
    Highlights regions the model focused on for its prediction.
    """
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        tape.watch(conv_outputs)
        pred_index    = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads        = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap      = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0)
    heatmap      = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index), predictions[0].numpy()


def plot_gradcam(model, last_conv_layer_name="top_conv"):
    """
    Plot Grad-CAM for sample images from each emotion class.
    Shows: original image | heatmap overlay | prediction bar.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import cv2, random
    random.seed(SEED)
    grad_model = build_grad_model(model)
    print(f"   Grad-CAM model built (connected graph)")

    fig, axes = plt.subplots(
        len(EMOTIONS), 3,
        figsize=(12, len(EMOTIONS) * 3)
    )
    fig.suptitle("Grad-CAM Visualization per Emotion Class",
                 fontsize=14, fontweight="bold", y=1.01)

    for row, emotion in enumerate(EMOTIONS):
        emotion_dir = os.path.join(TEST_DIR, emotion)
        sample_path = os.path.join(
            emotion_dir,
            random.choice(os.listdir(emotion_dir))
        )

        # Load & preprocess
        img_bgr  = cv2.imread(sample_path)
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rsz  = cv2.resize(img_rgb, IMG_SIZE)
        img_arr  = np.expand_dims(img_rsz.astype("float32") / 255.0, axis=0)

        # Grad-CAM
        heatmap, pred_idx, probs = make_gradcam_heatmap(img_arr, grad_model)

        # Overlay heatmap
        heatmap_rsz = cv2.resize(heatmap, IMG_SIZE)
        heatmap_col = np.uint8(255 * heatmap_rsz)
        heatmap_col = cv2.applyColorMap(heatmap_col, cv2.COLORMAP_JET)
        heatmap_col = cv2.cvtColor(heatmap_col, cv2.COLOR_BGR2RGB)
        overlay     = cv2.addWeighted(img_rsz, 0.6, heatmap_col, 0.4, 0)

        # Col 0: original
        axes[row][0].imshow(img_rsz)
        axes[row][0].set_title(f"True: {emotion}", fontsize=9, fontweight="bold")
        axes[row][0].axis("off")

        # Col 1: Grad-CAM overlay
        axes[row][1].imshow(overlay)
        correct = "[OK]" if pred_idx == EMOTIONS.index(emotion) else "[X]"
        axes[row][1].set_title(
            f"{correct} Pred: {EMOTIONS[pred_idx]} ({probs[pred_idx]:.2f})",
            fontsize=9
        )
        axes[row][1].axis("off")

        # Col 2: confidence bar chart
        colors = ["green" if i == pred_idx else
                  ("gold" if i == EMOTIONS.index(emotion) else "steelblue")
                  for i in range(NUM_CLASSES)]
        axes[row][2].barh(EMOTIONS, probs, color=colors)
        axes[row][2].set_xlim(0, 1)
        axes[row][2].tick_params(labelsize=7)
        axes[row][2].set_title("Confidence", fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "gradcam_visualization.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Grad-CAM visualization saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Facial Emotion Recognition — Evaluation Pipeline")
    print("=" * 60)

    model, test_ds = load_model_and_data()

    y_true, y_pred = get_predictions(model, test_ds)

    plot_confusion_matrix(y_true, y_pred)
    print_classification_report(y_true, y_pred)
    plot_gradcam(model)

    print("\n✅ evaluate.py completed successfully.")


if __name__ == "__main__":
    main()
