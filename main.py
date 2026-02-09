
import argparse
import os
import sys

# 添加 src 到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train import train
from src.test import test
from src.data.generator import ProcessBasedGenerator

def main():
    parser = argparse.ArgumentParser(description="PINN for Rocket Fuel Yield Stress Prediction")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: train
    train_parser = subparsers.add_parser("train", help="Train the PINN model")

    # Command: test
    test_parser = subparsers.add_parser("test", help="Evaluate the model on test set")

    # Command: generate_data
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic dataset")
    gen_parser.add_argument("--samples", type=int, default=20, help="Samples per real condition")

    args = parser.parse_args()

    if args.command == "train":
        print("Starting training pipeline...")
        train()

    elif args.command == "test":
        print("Starting evaluation...")
        test()

    elif args.command == "generate":
        print("Generating synthetic data...")
        gen = ProcessBasedGenerator()
        # 默认生成15000个样本，过滤后约2000-3000个有效样本
        n_total = args.samples * 1000  # --samples参数现在代表千个样本
        gen.generate(n_samples=n_total, save_path="data/synthetic/dataset.csv")
        print("Done.")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
