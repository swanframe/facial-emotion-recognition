
# =============================================================
# prepare_data.py — Download dataset & build tf.data pipeline
# =============================================================

import os
import json
import random
import numpy as np
import tensorflow as tf
from getpass import getpass
from config import (
    DATA_DIR, TRAIN_DIR, TEST_DIR,
    IMG_SIZE, BATCH_SIZE, VALIDATION_SPLIT,
    EMOTIONS, SEED
)


# ── 1. Kaggle download ────────────────────────────────────────
def setup_kaggle():
    """Configure Kaggle API credentials interactively."""
    kaggle_username = input("Masukkan Kaggle username: ")
    kaggle_key      = getpass("Masukkan Kaggle API key: ")

    os.makedirs("/root/.kaggle", exist_ok=True)
    with open("/root/.kaggle/kaggle.json", "w") as f:
        json.dump({"username": kaggle_username, "key": kaggle_key}, f)
    os.chmod("/root/.kaggle/kaggle.json", 0o600)
    print("✅ Kaggle credentials configured.")


def download_dataset():
    """Download and extract FER-2013 from Kaggle."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(os.path.join(DATA_DIR, "train")):
        print("⚡ Dataset already exists, skipping download.")
        return
    os.system(f"kaggle datasets download -d msambare/fer2013 -p {DATA_DIR} --unzip")
    print("✅ Dataset downloaded and extracted.")


# ── 2. tf.data pipeline ───────────────────────────────────────
def parse_image(file_path, label, augment=False):
    """
    Load image from path, convert grayscale→RGB,
    resize, normalize, and optionally augment.
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.grayscale_to_rgb(img)          # FER-2013 is grayscale
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0        # normalize to [0, 1]

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.15)
        img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
        img = tf.image.random_saturation(img, lower=0.85, upper=1.15)
        img = tf.clip_by_value(img, 0.0, 1.0)   # ensure values stay in [0, 1]

    return img, label


def build_dataset(directory, augment=False):
    """
    Build a tf.data.Dataset from a directory with
    structure: directory/class_name/image.jpg
    """
    file_paths, labels = [], []

    for idx, emotion in enumerate(EMOTIONS):
        emotion_dir = os.path.join(directory, emotion)
        if not os.path.exists(emotion_dir):
            continue
        for fname in os.listdir(emotion_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                file_paths.append(os.path.join(emotion_dir, fname))
                labels.append(idx)

    # Shuffle consistently
    combined = list(zip(file_paths, labels))
    random.seed(SEED)
    random.shuffle(combined)
    file_paths, labels = zip(*combined)

    labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=len(EMOTIONS))

    dataset = tf.data.Dataset.from_tensor_slices(
        (list(file_paths), labels_one_hot)
    )
    dataset = dataset.map(
        lambda x, y: parse_image(x, y, augment=augment),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset, len(file_paths)


def get_datasets():
    """
    Return train, val, and test tf.data.Dataset objects.
    Splits train directory into train/val using VALIDATION_SPLIT.
    """
    # ── Build full train dataset (with augmentation) ──
    full_dataset, total_train = build_dataset(TRAIN_DIR, augment=True)

    val_size   = int(total_train * VALIDATION_SPLIT)
    train_size = total_train - val_size

    train_ds = full_dataset.take(train_size)
    val_ds   = full_dataset.skip(train_size)

    # val/test: no augmentation — rebuild without augment for val
    full_no_aug, _ = build_dataset(TRAIN_DIR, augment=False)
    val_ds = full_no_aug.skip(train_size)

    # ── Test dataset (no augmentation) ──
    test_ds, total_test = build_dataset(TEST_DIR, augment=False)

    # ── Batch & prefetch ──
    train_ds = (train_ds
                .shuffle(1024, seed=SEED)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))

    val_ds   = (val_ds
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))

    test_ds  = (test_ds
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))

    print(f"\n📊 Dataset split:")
    print(f"   Train : {train_size} images")
    print(f"   Val   : {val_size} images")
    print(f"   Test  : {total_test} images")
    print(f"   Batch size   : {BATCH_SIZE}")
    print(f"   Steps/epoch  : {train_size // BATCH_SIZE}")

    return train_ds, val_ds, test_ds


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    setup_kaggle()
    download_dataset()
    train_ds, val_ds, test_ds = get_datasets()

    # Sanity check — inspect one batch
    for images, labels in train_ds.take(1):
        print(f"\n✅ Batch shape  : {images.shape}")
        print(f"   Label shape  : {labels.shape}")
        print(f"   Pixel range  : [{images.numpy().min():.2f}, {images.numpy().max():.2f}]")
    print("\n✅ prepare_data.py completed successfully.")
