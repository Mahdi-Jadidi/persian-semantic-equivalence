from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    data_path: Path
    output_dir: Path = Path("outputs/model")
    formal_column: str = "formalForm"
    informal_column: str = "informalForm"
    model_name: str = "HooshvareLab/bert-base-parsbert-uncased"
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    seed: int = 42
