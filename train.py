"""
train.py
--------
Trains a CNN (Convolutional Neural Network) model for pest detection
in crop images.

The script:
  1. Loads images from  dataset/<class_name>/  folders.
  2. Splits data into training and validation sets.
  3. Builds and trains a CNN using TensorFlow / Keras.
  4. Saves the trained model as  pest_model.h5 .
  5. Saves a training-history plot as  static/training_history.png .

Usage:
    python train.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # non-interactive backend
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_DIR  = os.path.join(os.path.dirname(__file__), "dataset")
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "pest_model.h5")
HISTORY_IMG  = os.path.join(os.path.dirname(__file__), "static", "training_history.png")

IMG_SIZE     = (128, 128)
BATCH_SIZE   = 32
EPOCHS       = 10
VAL_SPLIT    = 0.2
SEED         = 42

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
print("=" * 60)
print("  PEST DETECTION — CNN MODEL TRAINING")
print("=" * 60)

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=VAL_SPLIT,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.15,
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    seed=SEED,
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    seed=SEED,
)

class_names = list(train_gen.class_indices.keys())
num_classes = len(class_names)

print(f"\nClasses detected : {class_names}")
print(f"Number of classes: {num_classes}")
print(f"Training samples : {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation="relu", input_shape=(*IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Block 2
    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Block 3
    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Classifier head
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                    save_best_only=True, verbose=1),
]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
print("\n>>> Starting training …\n")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ---------------------------------------------------------------------------
# Save final model (in case checkpoint didn't trigger)
# ---------------------------------------------------------------------------
model.save(MODEL_PATH)
print(f"\n[OK] Model saved to {MODEL_PATH}")

# ---------------------------------------------------------------------------
# Plot training history
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(HISTORY_IMG), exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history["accuracy"],     label="Train Accuracy")
ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
ax1.set_title("Model Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history["loss"],     label="Train Loss")
ax2.plot(history.history["val_loss"], label="Val Loss")
ax2.set_title("Model Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(HISTORY_IMG, dpi=120)
print(f"[OK] Training history plot saved to {HISTORY_IMG}")

# ---------------------------------------------------------------------------
# Save class names for the app to read
# ---------------------------------------------------------------------------
labels_path = os.path.join(os.path.dirname(__file__), "models", "class_names.txt")
os.makedirs(os.path.dirname(labels_path), exist_ok=True)
with open(labels_path, "w") as f:
    for name in class_names:
        f.write(name + "\n")
print(f"[OK] Class names saved to {labels_path}")

print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)
