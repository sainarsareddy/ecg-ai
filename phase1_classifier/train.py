import os

import numpy as np
import tensorflow as tf
import wfdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (BatchNormalization, Conv1D, Dense,
                                     Dropout, Flatten, MaxPooling1D)
from tensorflow.keras.models import Sequential

# ── CONFIG ────────────────────────────────────────────────────────────────────
SEGMENT_LENGTH = 187
DATA_DIR = "data"
MODEL_PATH = "models/ecg_model.h5"

# MIT-BIH records
RECORDS = ['100','101','103','105','106','107','108','109','111','112',
           '113','114','115','116','117','118','119','121','122','123',
           '124','200','201','202','203','205','207','208','209','210',
           '212','213','214','215','217','219','220','221','222','223',
           '228','230','231','232','233','234']

# Label map — grouping into 4 classes
LABEL_MAP = {
    'N': 'Normal',
    'L': 'Normal',
    'R': 'Normal',
    'A': 'AFib',
    'a': 'AFib',
    'J': 'AFib',
    'S': 'AFib',
    'V': 'PVC',
    'E': 'PVC',
    'F': 'PVC',
    '/': 'Tachycardia',
    'f': 'Tachycardia',
    'Q': 'Tachycardia',
}

def load_data():
    signals, labels = [], []
    print("Loading MIT-BIH dataset...")

    for rec in RECORDS:
        try:
            record = wfdb.rdrecord(rec, sampto=10000, pn_dir='mitdb')
            annotation = wfdb.rdann(rec, 'atr', sampto=10000, pn_dir='mitdb')
            signal = record.p_signal[:, 0]

            for i, (sample, symbol) in enumerate(zip(annotation.sample, annotation.symbol)):
                if symbol not in LABEL_MAP:
                    continue
                start = sample - SEGMENT_LENGTH // 2
                end = sample + SEGMENT_LENGTH // 2
                if start < 0 or end > len(signal):
                    continue
                segment = signal[start:end]
                segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
                signals.append(segment)
                labels.append(LABEL_MAP[symbol])

        except Exception as e:
            print(f"Skipping record {rec}: {e}")
            continue

    print(f"Total samples loaded: {len(signals)}")
    return np.array(signals), np.array(labels)


def build_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(64, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train():
    os.makedirs("models", exist_ok=True)

    X, y_raw = load_data()

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"Classes: {le.classes_}")

    # Save label encoder classes
    np.save("models/classes.npy", le.classes_)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Reshape for Conv1D
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Fix class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")

    # Build and train
    model = build_model((SEGMENT_LENGTH, 1), len(le.classes_))
    model.summary()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True)
    ]

    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=64,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    loss, acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()