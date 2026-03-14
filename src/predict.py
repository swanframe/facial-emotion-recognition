
# =============================================================
# predict.py — Inference on a single image with face detection
# =============================================================
# Usage:
#   python src/predict.py --image path/to/photo.jpg
#   python src/predict.py --image path/to/photo.jpg --save
# =============================================================

import os
import sys
import argparse
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import (
    BEST_MODEL_PATH, EMOTIONS, NUM_CLASSES,
    IMG_SIZE, OUTPUT_DIR
)


# ── 1. Argument parser ────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Facial Emotion Recognition — Single Image Inference"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input image file"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save annotated output image to outputs/"
    )
    return parser.parse_args()


# ── 2. Face detection ─────────────────────────────────────────
def detect_faces(image_bgr):
    """
    Detect faces using OpenCV Haar Cascade.
    Returns list of (x, y, w, h) bounding boxes.
    Falls back to full image if no faces detected.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48)
    )

    if len(faces) == 0:
        print("⚠️  No faces detected — using full image as input.")
        h, w = image_bgr.shape[:2]
        return [(0, 0, w, h)]

    print(f"   Detected {len(faces)} face(s)")
    return faces


# ── 3. Preprocess face crop ───────────────────────────────────
def preprocess_face(image_bgr, bbox):
    """
    Crop face from image, convert to RGB, resize, normalize.
    Returns preprocessed array ready for model input.
    """
    x, y, w, h = bbox
    # Add small padding around face
    pad = int(min(w, h) * 0.1)
    x1  = max(0, x - pad)
    y1  = max(0, y - pad)
    x2  = min(image_bgr.shape[1], x + w + pad)
    y2  = min(image_bgr.shape[0], y + h + pad)

    face_bgr = image_bgr[y1:y2, x1:x2]
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rsz = cv2.resize(face_rgb, IMG_SIZE)
    face_arr = face_rsz.astype("float32") / 255.0
    return np.expand_dims(face_arr, axis=0), face_rgb


# ── 4. Predict emotion ────────────────────────────────────────
def predict_emotion(model, face_array):
    """
    Run model inference on preprocessed face.
    Returns predicted emotion label and confidence scores.
    """
    probs      = model.predict(face_array, verbose=0)[0]
    pred_idx   = np.argmax(probs)
    pred_label = EMOTIONS[pred_idx]
    confidence = probs[pred_idx]
    return pred_label, confidence, probs


# ── 5. Visualize result ───────────────────────────────────────
def visualize_result(image_bgr, faces, predictions, save=False, image_path=""):
    """
    Draw bounding boxes + emotion labels on image.
    Optionally save annotated output.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Facial Emotion Recognition — Prediction",
                 fontsize=13, fontweight="bold")

    # Left: annotated image
    axes[0].imshow(image_rgb)
    axes[0].set_title("Detected Face(s)", fontsize=11)
    axes[0].axis("off")

    for (x, y, w, h), (label, conf, _) in zip(faces, predictions):
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor="lime", facecolor="none"
        )
        axes[0].add_patch(rect)
        axes[0].text(
            x, y - 8,
            f"{label} ({conf:.2f})",
            color="lime", fontsize=10, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.5, pad=2)
        )

    # Right: confidence bar for first face
    if predictions:
        label, conf, probs = predictions[0]
        colors = ["green" if i == np.argmax(probs) else "steelblue"
                  for i in range(NUM_CLASSES)]
        axes[1].barh(EMOTIONS, probs, color=colors)
        axes[1].set_xlim(0, 1)
        axes[1].set_title(f"Confidence — Predicted: {label} ({conf:.2f})",
                          fontsize=11)
        axes[1].set_xlabel("Probability")
        for i, v in enumerate(probs):
            if v > 0.02:
                axes[1].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)

    plt.tight_layout()

    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base     = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"predict_{base}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\n💾 Annotated output saved: {out_path}")

    plt.show()


# ── Main ──────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Validate input
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)

    print("=" * 55)
    print("  Facial Emotion Recognition — Inference")
    print("=" * 55)
    print(f"\n📷 Input image : {args.image}")

    # Load model
    print("📦 Loading model...")
    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    print(f"   Model loaded : {BEST_MODEL_PATH}")

    # Load image
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print(f"❌ Cannot read image: {args.image}")
        sys.exit(1)
    print(f"   Image size   : {image_bgr.shape[1]}x{image_bgr.shape[0]}")

    # Detect faces
    print("\n🔍 Detecting faces...")
    faces = detect_faces(image_bgr)

    # Predict per face
    print("\n🧠 Running emotion recognition...")
    predictions = []
    for i, bbox in enumerate(faces):
        face_arr, _ = preprocess_face(image_bgr, bbox)
        label, conf, probs = predict_emotion(model, face_arr)
        predictions.append((label, conf, probs))
        print(f"   Face {i+1}: {label:<12} (confidence: {conf:.4f})")

    # Visualize
    visualize_result(image_bgr, faces, predictions,
                     save=args.save, image_path=args.image)

    print("\n✅ predict.py completed successfully.")


if __name__ == "__main__":
    main()
