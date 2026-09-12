"""
train_blink_classifier.py

Week 3 — Trains a small Keras classifier on the labeled EAR-window dataset
built by prepare_training_data.py. Classifies a 10-frame EAR sequence into
one of: open, short_blink, long_blink.

Uses class weighting to compensate for the long_blink class having fewer
samples than open/short_blink (see dataset notes for details).

Run from the project root:
    python train_blink_classifier.py

Output: models/blink_cnn_model.h5
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_CSV = os.path.join("data", "training_data.csv")
MODEL_OUT = os.path.join("models", "blink_cnn_model.h5")
WINDOW_SIZE = 10

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_dataset():
    df = pd.read_csv(DATA_CSV)
    ear_cols = [f"ear_{i}" for i in range(WINDOW_SIZE)]
    X_ear = df[ear_cols].values.astype("float32")
    X_duration = df["duration"].values.astype("float32").reshape(-1, 1)
    # normalize duration to a similar scale as EAR values
    X_duration = X_duration / 200.0
    X = np.hstack([X_ear, X_duration])  # now 11 features per sample
    y_raw = df["label"].values
    return X, y_raw, df


def build_model(input_length, num_classes):
    """Small 1D-CNN — appropriate for a short numeric sequence like this,
    not a large image-based network. Kept intentionally small so it runs
    fast even on low-spec hardware, consistent with the project's
    low-resource-deployment goal."""
    model = keras.Sequential([
        layers.Input(shape=(input_length, 1)),
        layers.Conv1D(16, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("Loading dataset...")
    X, y_raw, df = load_dataset()
    print(f"Total samples: {len(X)}")
    print(f"Class distribution:\n{df['label'].value_counts()}\n")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    class_names = encoder.classes_
    print(f"Classes (encoded order): {list(class_names)}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train samples: {len(X_train)}  |  Test samples: {len(X_test)}\n")

    # reshape for Conv1D: (samples, timesteps, features)
    X_train = X_train.reshape(-1, 11, 1)
    X_test = X_test.reshape(-1, 11, 1)

    # class weights to compensate for long_blink having fewer samples
    weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(weights)}
    print(f"Class weights: {dict(zip(class_names, weights))}\n")

    model = build_model(11, num_classes=len(class_names))
    model.summary()

    print("\nTraining...\n")
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=60,
        batch_size=16,
        class_weight=class_weight_dict,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )
        ],
        verbose=1,
    )

    print("\nEvaluating on held-out test set...\n")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"Order: {list(class_names)}")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_OUT)
    print(f"\nModel saved to: {MODEL_OUT}")

    # save the label encoder classes too — needed later to decode predictions
    with open(os.path.join("models", "label_classes.txt"), "w") as f:
        for c in class_names:
            f.write(c + "\n")
    print("Label classes saved to: models/label_classes.txt")


if __name__ == "__main__":
    main()