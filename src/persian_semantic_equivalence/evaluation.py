import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval(); labels, predictions = [], []
    for batch in loader:
        targets = batch.pop("labels").to(device); output = model(**{key: value.to(device) for key, value in batch.items()}); labels.extend(targets.cpu().numpy()); predictions.extend(output.logits.argmax(1).cpu().numpy())
    labels, predictions = np.asarray(labels), np.asarray(predictions); metrics = {"accuracy": accuracy_score(labels, predictions), "precision": precision_score(labels, predictions, zero_division=0), "recall": recall_score(labels, predictions, zero_division=0), "f1": f1_score(labels, predictions, zero_division=0), "classification_report": classification_report(labels, predictions, output_dict=True, zero_division=0), "confusion_matrix": confusion_matrix(labels, predictions).tolist()}; return metrics, labels, predictions
