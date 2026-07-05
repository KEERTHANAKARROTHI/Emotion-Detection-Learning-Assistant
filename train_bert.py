"""
train_bert.py
-------------
Fine-tunes a DistilBERT classifier on dataset/emotions.csv and saves the
model + tokenizer to models/bert_model/ (a folder, as required by
`from_pretrained` / `save_pretrained`).

Note: this downloads 'distilbert-base-uncased' from Hugging Face the
first time you run it, so you need an internet connection.

Run:
    python train_bert.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from utils.preprocess import clean_text

DATA_PATH = os.path.join("dataset", "emotions.csv")
MODEL_DIR = os.path.join("models", "bert_model")
BASE_MODEL = "distilbert-base-uncased"
MAX_LEN = 48

os.makedirs(MODEL_DIR, exist_ok=True)


class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


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
    num_labels = len(label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"].tolist(), y.tolist(),
        test_size=0.2, random_state=42, stratify=y
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    train_enc = tokenizer(
        X_train, truncation=True, padding=True, max_length=MAX_LEN
    )
    test_enc = tokenizer(
        X_test, truncation=True, padding=True, max_length=MAX_LEN
    )

    train_dataset = EmotionDataset(train_enc, y_train)
    test_dataset = EmotionDataset(test_enc, y_test)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=num_labels
    )

    # `eval_strategy` was renamed from `evaluation_strategy` in newer
    # transformers versions -- try the new name first, fall back if needed.
    common_args = dict(
        output_dir=os.path.join(MODEL_DIR, "checkpoints"),
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_strategy="no",
        logging_steps=10,
        learning_rate=2e-5,
        report_to=[],
    )
    try:
        training_args = TrainingArguments(eval_strategy="epoch", **common_args)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy="epoch", **common_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    # Save final model + tokenizer for later use with from_pretrained()
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    with open(os.path.join("models", "bert_label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"Saved fine-tuned BERT model -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
