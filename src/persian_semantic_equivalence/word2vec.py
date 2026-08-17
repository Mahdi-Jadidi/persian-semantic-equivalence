import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from .normalization import tokenize_words


def train_word2vec(pairs: pd.DataFrame, vector_size: int = 100, epochs: int = 10, seed: int = 42) -> Word2Vec:
    corpus = [tokenize_words(text) for text in pd.concat([pairs.formal, pairs.informal])]; return Word2Vec(corpus, vector_size=vector_size, window=5, min_count=2, workers=1, sg=1, epochs=epochs, seed=seed)


def sentence_vector(model: Word2Vec, text: str) -> np.ndarray:
    vectors = [model.wv[token] for token in tokenize_words(text) if token in model.wv]; return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


def pair_similarities(model: Word2Vec, pairs: pd.DataFrame) -> np.ndarray:
    scores = []
    for row in pairs.itertuples():
        left, right = sentence_vector(model, row.formal), sentence_vector(model, row.informal); denominator = np.linalg.norm(left) * np.linalg.norm(right); scores.append(float(left @ right / denominator) if denominator else 0)
    return np.asarray(scores)
