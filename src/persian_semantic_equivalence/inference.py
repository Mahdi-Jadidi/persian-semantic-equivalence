from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class EquivalencePredictor:
    def __init__(self, model_dir: Path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir); self.model = AutoModelForSequenceClassification.from_pretrained(model_dir); self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, formal: str, informal: str) -> dict:
        encoded = self.tokenizer(formal, informal, return_tensors="pt", truncation=True, max_length=128).to(self.device); probabilities = self.model(**encoded).logits.softmax(1)[0].cpu(); prediction = int(probabilities.argmax()); return {"equivalent": bool(prediction), "confidence": float(probabilities[prediction]), "probabilities": {"not_equivalent": float(probabilities[0]), "equivalent": float(probabilities[1])}}
