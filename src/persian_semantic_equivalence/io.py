from pathlib import Path

import pandas as pd

from .normalization import normalize_persian


def load_pairs(path: Path, formal_column: str, informal_column: str) -> pd.DataFrame:
    frame = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    missing = {formal_column, informal_column} - set(frame.columns)
    if missing: raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    return frame[[formal_column, informal_column]].dropna().rename(columns={formal_column: "formal", informal_column: "informal"}).assign(formal=lambda d: d.formal.map(normalize_persian), informal=lambda d: d.informal.map(normalize_persian)).query("formal != '' and informal != ''").reset_index(drop=True)
