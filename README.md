# Semantic Equivalence in Formal and Informal Persian

A complete Persian semantic-equivalence pipeline covering text normalization, corpus statistics, subword fragmentation and token inflation, Word2Vec sentence similarity, ParsBERT contextual embeddings, supervised pair construction, transformer fine-tuning, evaluation, and reusable inference.

## Commands

```bash
pip install -e .
persian-equivalence analyze --data-path trainset_40k_onlySent.xlsx --output-dir outputs/analysis
persian-equivalence train --data-path trainset_40k_onlySent.xlsx --output-dir outputs/model
persian-equivalence predict --model-dir outputs/model --formal "..." --informal "..."
```

The implementation is divided into `normalization.py`, `tokenization.py`, `word2vec.py`, `embeddings.py`, `pairs.py`, `dataset.py`, `training.py`, `evaluation.py`, and `inference.py`. Expensive model downloads happen only when transformer stages are requested.

## Data contract

The default columns are `formalForm` and `informalForm`; both can be overridden from the CLI. The train command constructs balanced positive and shuffled-negative pairs with deterministic train/validation/test splits.

## Topics

`persian-nlp` `semantic-similarity` `parsbert` `transformers` `word2vec` `text-classification`
