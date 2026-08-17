import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .config import TrainingConfig
from .dataset import SemanticPairDataset
from .evaluation import evaluate
from .io import load_pairs
from .pairs import build_supervised_pairs, split_pairs


def train_model(config: TrainingConfig) -> dict:
    torch.manual_seed(config.seed); pairs = load_pairs(config.data_path, config.formal_column, config.informal_column); train, validation, test = split_pairs(build_supervised_pairs(pairs, config.seed), config.seed); tokenizer = AutoTokenizer.from_pretrained(config.model_name); model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    train_loader = DataLoader(SemanticPairDataset(train, tokenizer, config.max_length), batch_size=config.batch_size, shuffle=True); validation_loader = DataLoader(SemanticPairDataset(validation, tokenizer, config.max_length), batch_size=config.batch_size); test_loader = DataLoader(SemanticPairDataset(test, tokenizer, config.max_length), batch_size=config.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate); scheduler = get_linear_schedule_with_warmup(optimizer, int(.1 * len(train_loader) * config.epochs), len(train_loader) * config.epochs); history, best_f1 = [], -1.0; config.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config.epochs + 1):
        model.train(); total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(); output = model(**{key: value.to(device) for key, value in batch.items()}); output.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); scheduler.step(); total_loss += output.loss.item()
        validation_metrics, _, _ = evaluate(model, validation_loader, device); record = {"epoch": epoch, "train_loss": total_loss / len(train_loader), **{f"validation_{key}": value for key, value in validation_metrics.items() if isinstance(value, (int, float))}}; history.append(record)
        if validation_metrics["f1"] > best_f1: best_f1 = validation_metrics["f1"]; model.save_pretrained(config.output_dir); tokenizer.save_pretrained(config.output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(config.output_dir).to(device); test_metrics, labels, predictions = evaluate(model, test_loader, device); (config.output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8"); (config.output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8"); test.assign(prediction=predictions, correct=labels == predictions).to_csv(config.output_dir / "test_predictions.csv", index=False); return test_metrics
