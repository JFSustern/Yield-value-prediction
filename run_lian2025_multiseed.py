"""
多种子鲁棒性验证：Lian 2025 HF-only vs PI-MFNN，5个随机种子
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from multi_fidelity.src.training.train_lian2025 import (
    project_root,
    split_hifi_data,
    train_hifi_only,
    train_low_fidelity,
    train_multifidelity,
)

SEEDS   = [0, 1, 2, 3, 42]
HF_PATH = 'data/lian2025/high_fidelity/all_400.csv'


def main():
    records = []

    # LF 预训练结果作为共同起点；每个 HF 分支仍使用各自 seed。
    print("=== 低保真预训练 ===")
    low_model, _ = train_low_fidelity()

    for seed in SEEDS:
        print(f"\n{'='*55}")
        print(f"  SEED = {seed}")
        print(f"{'='*55}")

        (X_tr, y_tr, _), (X_ev, y_ev, _), (X_te, y_te, _) = split_hifi_data(
            project_root / HF_PATH,
            n_train=30, n_eval=10, seed=seed,
        )

        _, res_c = train_hifi_only(
            X_tr, y_tr, X_ev, y_ev, X_te, y_te,
            lr=1e-4, epochs=1000, patience=150,
            exp_tag=f'ms_hifi_seed{seed}',
            seed=seed,
        )
        _, res_d = train_multifidelity(
            low_model, X_tr, y_tr, X_ev, y_ev, X_te, y_te,
            freeze_n=1, lr=1e-4, epochs=1000, patience=150,
            exp_tag=f'ms_mf_seed{seed}',
            seed=seed,
        )

        record = {
            "seed": seed,
            "C_r2": round(res_c["test"]["r2"], 4),
            "C_mape": round(res_c["test"]["mape"], 2),
            "C_mae": round(res_c["test"]["mae"], 4),
            "D_r2": round(res_d["test"]["r2"], 4),
            "D_mape": round(res_d["test"]["mape"], 2),
            "D_mae": round(res_d["test"]["mae"], 4),
        }
        records.append(record)
        print(json.dumps(record, indent=2))

    print("\n" + "="*60)
    print("多种子汇总 (5 seeds, 360 条测试)")
    print("="*60)
    for key in ["C_r2", "C_mape", "D_r2", "D_mape"]:
        vals = [record[key] for record in records]
        print(f"  {key:12s}  {np.mean(vals):.4f} ± {np.std(vals):.4f}  "
              f"[{min(vals):.4f}, {max(vals):.4f}]")

    gain_r2 = [record["D_r2"] - record["C_r2"] for record in records]
    gain_mape = [record["C_mape"] - record["D_mape"] for record in records]
    print(f"\n  MF增益ΔR²:   {np.mean(gain_r2):+.4f} ± {np.std(gain_r2):.4f}")
    print(f"  MF增益ΔMAPE: {np.mean(gain_mape):+.2f}pp ± {np.std(gain_mape):.2f}pp")

    out = project_root / "multi_fidelity/results/lian2025/multiseed_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as file:
        json.dump({"records": records, "seeds": SEEDS}, file, indent=2)
    print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
