import numpy as np
import pandas as pd

from .normalization import tokenize_words


def word_statistics(pairs: pd.DataFrame) -> dict[str, float]:
    formal = pairs.formal.map(lambda value: len(tokenize_words(value))); informal = pairs.informal.map(lambda value: len(tokenize_words(value))); inflation = informal / formal.replace(0, np.nan)
    return {"pairs": len(pairs), "formal_mean_words": float(formal.mean()), "informal_mean_words": float(informal.mean()), "word_inflation_mean": float(inflation.mean())}


def transformer_statistics(pairs: pd.DataFrame, model_name: str, sample_size: int | None = None) -> tuple[dict, pd.DataFrame]:
    from transformers import AutoTokenizer
    sample = pairs.sample(min(sample_size, len(pairs)), random_state=42) if sample_size else pairs; tokenizer = AutoTokenizer.from_pretrained(model_name)
    rows = []
    for row in sample.itertuples():
        formal_tokens = tokenizer.tokenize(row.formal); informal_tokens = tokenizer.tokenize(row.informal); rows.append({"formal": row.formal, "informal": row.informal, "formal_tokens": len(formal_tokens), "informal_tokens": len(informal_tokens), "token_inflation": len(informal_tokens) / max(len(formal_tokens), 1)})
    detail = pd.DataFrame(rows); return {"model": model_name, "pairs": len(detail), "formal_mean_tokens": float(detail.formal_tokens.mean()), "informal_mean_tokens": float(detail.informal_tokens.mean()), "mean_token_inflation": float(detail.token_inflation.mean())}, detail
