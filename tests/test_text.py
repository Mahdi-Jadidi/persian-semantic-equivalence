import pandas as pd

from persian_semantic_equivalence.normalization import normalize_persian
from persian_semantic_equivalence.pairs import build_supervised_pairs


def test_normalizes_arabic_codepoints() -> None:
    assert normalize_persian("كتاب يكي") == "کتاب یکی"


def test_supervised_pairs_are_balanced() -> None:
    pairs = pd.DataFrame({"formal": ["a", "b", "c"], "informal": ["x", "y", "z"]})
    output = build_supervised_pairs(pairs)
    assert set(output.label) == {0, 1}
