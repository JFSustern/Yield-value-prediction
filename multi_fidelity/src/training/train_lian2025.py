"""Command-line entry point for the Lian 2025 experiments."""

from runpy import run_module


if __name__ == "__main__":
    run_module(
        "multi_fidelity.src.training.lian2025_experiments",
        run_name="__main__",
    )
