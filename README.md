<div align="center">

# Formal-Informal Persian Semantic Equivalence

**From Persian normalization and tokenization analysis to ParsBERT fine-tuning and inference**

[![CI](https://github.com/Mahdi-Jadidi/persian-semantic-equivalence/actions/workflows/ci.yml/badge.svg)](https://github.com/Mahdi-Jadidi/persian-semantic-equivalence/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-ParsBERT-FFD21E?logo=huggingface&logoColor=black)

</div>

## Overview

Informal Persian changes spelling, morphology, spacing, and word forms while often preserving meaning. This project measures that representation gap and trains a classifier to decide whether a formal and colloquial sentence are semantically equivalent.

## Results

| Metric | Held-out score |
|---|---:|
| Accuracy | **0.8934** |
| Macro precision | **0.9112** |
| Macro recall | **0.8934** |
| Macro F1 | **0.8923** |

Macro metrics are reported because negative pairs and linguistic variation can hide class-specific errors behind accuracy alone.

## Research workflow

```mermaid
flowchart LR
    A[Formal / informal sentence pairs] --> B[Persian normalization]
    B --> C[Corpus and tokenization analysis]
    C --> D1[Word2Vec baseline]
    C --> D2[ParsBERT embeddings]
    B --> E[Balanced positive / negative pairs]
    E --> F[Transformer fine-tuning]
    D1 --> G[Comparative evaluation]
    D2 --> G
    F --> G
    G --> H[Reusable inference]
```

## Highlights

- Character and spacing normalization designed for Persian text.
- Subword fragmentation and token-inflation analysis across registers.
- Static Word2Vec similarity and contextual ParsBERT baselines.
- Deterministic balanced-pair generation with shuffled hard negatives.
- Train/validation/test evaluation and a command-line prediction path.

## Data contract

| Field | Default column | Description |
|---|---|---|
| Formal sentence | `formalForm` | Standard written Persian |
| Informal sentence | `informalForm` | Colloquial or conversational equivalent |

Both column names are configurable from the CLI.

## Quick start

```bash
git clone https://github.com/Mahdi-Jadidi/persian-semantic-equivalence.git
cd persian-semantic-equivalence
pip install -e .
persian-equivalence analyze --data-path trainset_40k_onlySent.xlsx --output-dir outputs/analysis
persian-equivalence train --data-path trainset_40k_onlySent.xlsx --output-dir outputs/model
persian-equivalence predict --model-dir outputs/model --formal "..." --informal "..."
```

## Repository layout

The package separates normalization, tokenization, Word2Vec, transformer embeddings, pair construction, training, evaluation, and inference under `src/persian_semantic_equivalence`.

## Limitations

Randomly shuffled negatives may be easier than naturally occurring semantic confusions. Future evaluation should include human-curated hard negatives, dialect-specific slices, and robustness checks for code-switching and spelling noise.
