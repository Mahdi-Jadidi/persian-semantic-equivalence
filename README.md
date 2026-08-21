# Semantic Equivalence in Formal and Informal Persian

A Persian NLP project for measuring whether formal and colloquial sentences preserve the same meaning. It moves from language-aware normalization and corpus analysis to static embeddings, contextual ParsBERT embeddings, and a supervised semantic-equivalence classifier.

## Why this is difficult

Informal Persian changes spelling, token boundaries, morphology, and punctuation without necessarily changing meaning. A useful system must distinguish harmless register variation from a genuine semantic change, while avoiding conclusions that are merely artifacts of tokenization.

## What was built

- Persian-aware normalization and formal/informal corpus statistics.
- Subword-fragmentation and token-inflation analysis to quantify the representation cost of informal text.
- Word2Vec and ParsBERT similarity baselines.
- Deterministic construction of balanced positive and shuffled-negative sentence pairs.
- Transformer fine-tuning, held-out evaluation, and reusable formal/informal inference.

## Main takeaways

The project makes the representation gap visible before modelling it: informal writing can inflate tokenization complexity even when semantic content is preserved. Contextual transformer embeddings are therefore evaluated alongside simpler similarity baselines rather than assumed to be necessary by default.

## Reproduce

```bash
pip install -e .
persian-equivalence analyze --data-path trainset_40k_onlySent.xlsx --output-dir outputs/analysis
persian-equivalence train --data-path trainset_40k_onlySent.xlsx --output-dir outputs/model
```

The default data contract uses `formalForm` and `informalForm`, and both columns can be overridden from the CLI. GitHub Actions checks the package independently of model downloads.
