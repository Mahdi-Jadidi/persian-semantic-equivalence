import json
from pathlib import Path

import pandas as pd

from .io import load_pairs
from .tokenization import transformer_statistics, word_statistics
from .word2vec import pair_similarities, train_word2vec


def run_analysis(data_path: Path, output_dir: Path, formal_column: str, informal_column: str, model_name: str, sample_size: int = 2000) -> dict:
    pairs = load_pairs(data_path, formal_column, informal_column); output_dir.mkdir(parents=True, exist_ok=True); summary = {"word_statistics": word_statistics(pairs)}; model = train_word2vec(pairs); pairs.assign(word2vec_similarity=pair_similarities(model, pairs)).to_csv(output_dir / "word2vec_similarities.csv", index=False); model.save(str(output_dir / "word2vec.model")); transformer_summary, token_details = transformer_statistics(pairs, model_name, sample_size); summary["transformer_statistics"] = transformer_summary; token_details.to_csv(output_dir / "tokenization_details.csv", index=False); (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"); return summary
