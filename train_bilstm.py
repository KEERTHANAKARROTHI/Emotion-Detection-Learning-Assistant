"""
train_bilstm.py
----------------
Trains a BiLSTM emotion classifier on dataset/emotions.csv and saves:
    models/bilstm_model.h5
    models/tokenizer.pkl
    models/label_encoder.pkl

Run:
    python train_bilstm.py
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from utils.preprocess import clean_text

DATA_PATH = os.path.join("dataset", "emotions.csv")
MODEL_DIR = "models"
MAX_WORDS = 5000
MAX_LEN = 40
EMBED_DIM = 64

os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python dataset/generate_dataset.py` "
            "first, or place your own labeled CSV there."
        )

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "emotion"])
    df["clean_text"] = df["text"].apply(clean_text)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["emotion"])
    y_cat = to_categorical(y)

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["clean_text"])
    sequences = tokenizer.texts_to_sequences(df["clean_text"])
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )

    num_classes = y_cat.shape[1]
    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBED_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=25,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")

    model.save(os.path.join(MODEL_DIR, "bilstm_model.h5"))

    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    print("Saved: models/bilstm_model.h5, tokenizer.pkl, label_encoder.pkl")


if __name__ == "__main__":
    main()
