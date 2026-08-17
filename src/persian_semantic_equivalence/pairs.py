import pandas as pd
from sklearn.model_selection import train_test_split


def build_supervised_pairs(pairs: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    positive = pairs.assign(label=1); negative = pairs.assign(informal=pairs.informal.sample(frac=1, random_state=seed).reset_index(drop=True), label=0); negative = negative[negative.formal != negative.informal]; return pd.concat([positive, negative], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def split_pairs(frame: pd.DataFrame, seed: int = 42):
    train, temporary = train_test_split(frame, test_size=.2, stratify=frame.label, random_state=seed); validation, test = train_test_split(temporary, test_size=.5, stratify=temporary.label, random_state=seed); return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)
