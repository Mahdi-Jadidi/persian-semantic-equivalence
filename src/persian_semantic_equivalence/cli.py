import argparse
import json
from pathlib import Path

from .config import TrainingConfig
from .inference import EquivalencePredictor
from .pipeline import run_analysis
from .training import train_model


def add_data_arguments(parser) -> None:
    parser.add_argument("--data-path", type=Path, required=True); parser.add_argument("--formal-column", default="formalForm"); parser.add_argument("--informal-column", default="informalForm")


def main() -> None:
    parser = argparse.ArgumentParser(prog="persian-equivalence"); commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze"); add_data_arguments(analyze); analyze.add_argument("--output-dir", type=Path, default=Path("outputs/analysis")); analyze.add_argument("--model-name", default="HooshvareLab/bert-fa-base-uncased"); analyze.add_argument("--sample-size", type=int, default=2000)
    train = commands.add_parser("train"); add_data_arguments(train); train.add_argument("--output-dir", type=Path, default=Path("outputs/model")); train.add_argument("--model-name", default="HooshvareLab/bert-base-parsbert-uncased"); train.add_argument("--epochs", type=int, default=3); train.add_argument("--batch-size", type=int, default=16)
    predict = commands.add_parser("predict"); predict.add_argument("--model-dir", type=Path, required=True); predict.add_argument("--formal", required=True); predict.add_argument("--informal", required=True); args = parser.parse_args()
    if args.command == "analyze": result = run_analysis(args.data_path, args.output_dir, args.formal_column, args.informal_column, args.model_name, args.sample_size)
    elif args.command == "train": result = train_model(TrainingConfig(args.data_path, args.output_dir, args.formal_column, args.informal_column, args.model_name, batch_size=args.batch_size, epochs=args.epochs))
    else: result = EquivalencePredictor(args.model_dir).predict(args.formal, args.informal)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
