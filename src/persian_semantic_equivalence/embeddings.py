import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def encode_sentences(sentences: list[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name); model = AutoModel.from_pretrained(model_name); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device).eval(); vectors = []
    for start in range(0, len(sentences), batch_size):
        encoded = tokenizer(sentences[start:start + batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device); output = model(**encoded).last_hidden_state; mask = encoded.attention_mask.unsqueeze(-1); pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1); vectors.extend(pooled.cpu().numpy())
    return np.asarray(vectors)


def contextual_similarities(pairs: pd.DataFrame, model_name: str, batch_size: int = 32) -> np.ndarray:
    left = encode_sentences(pairs.formal.tolist(), model_name, batch_size); right = encode_sentences(pairs.informal.tolist(), model_name, batch_size); return np.sum(left * right, axis=1) / (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)).clip(min=1e-12)
