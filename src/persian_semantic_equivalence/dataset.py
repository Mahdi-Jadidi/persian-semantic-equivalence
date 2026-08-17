import torch
from torch.utils.data import Dataset


class SemanticPairDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length: int) -> None:
        self.frame = frame.reset_index(drop=True); self.tokenizer = tokenizer; self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]; encoded = self.tokenizer(row.formal, row.informal, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"); return {key: value.squeeze(0) for key, value in encoded.items()} | {"labels": torch.tensor(int(row.label), dtype=torch.long)}
