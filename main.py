
import argparse
import os
import sys

# 添加 src 到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train import train
from src.data.loader import load_excel_data
from src.data.generator import SyntheticDataGenerator

def main():
    parser = argparse.ArgumentParser(description="PINN for Rocket Fuel Yield Stress Prediction")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: train
    train_parser = subparsers.add_parser("train", help="Train the PINN model")

    # Command: generate_data
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic dataset")
    gen_parser.add_argument("--samples", type=int, default=20, help="Samples per real condition")

    args = parser.parse_args()

    if args.command == "train":
        print("Starting training pipeline...")
        train()

    elif args.command == "generate":
        print("Generating synthetic data...")
        real_df = load_excel_data()
        if real_df.empty:
            print("No real data found in data/20251121处理后/")
            return

        gen = SyntheticDataGenerator(real_df)
        gen.generate(n_samples_per_real=args.samples, save_path="data/synthetic/dataset.csv")
        print("Done.")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
