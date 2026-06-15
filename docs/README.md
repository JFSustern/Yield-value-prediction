# 数据与实验产物说明

本文档用于项目交接，说明各数据目录的用途、样本数和已知限制。CSV 行数不含表头。

## Lian 2025 体系

| 路径 | 样本数 | 用途 |
|---|---:|---|
| `data/high_fidelity/hifi_table6.csv` | 16 | Table 6 整理数据 |
| `data/high_fidelity/hf_all400.csv` | 400 | 当前主实验使用的 HF 工作数据 |
| `data/synthetic_table6_v2/train_data.csv` | 1600 | LF 预训练 |
| `data/synthetic_table6_v2/test_data.csv` | 400 | LF 预训练评估 |
| `data/high_fidelity/v3_split_seed42/train.csv` | 30 | seed=42 HF 训练 |
| `data/high_fidelity/v3_split_seed42/eval.csv` | 10 | seed=42 早停评估 |
| `data/high_fidelity/v3_split_seed42/test.csv` | 360 | seed=42 最终测试 |

主要字段：

- `Phi`：固体体积分数。
- `SP_percent`：增塑剂掺量的百分比数值，例如 `0.50` 表示 0.50%。
- `Tau0_Pa`：屈服应力，单位 Pa。
- `phi_max`：仅出现在 LF 数据中，用于生成校验，不作为监督目标直接输入模型。

`hf_all400.csv` 当前缺少独立的数据生成/扩展脚本和来源记录。接手人应优先补齐该文件从 16 条 Table 6 数据扩展到 400 条工作数据的过程，否则论文中不宜直接称其为 400 条独立真实测量。

## Zhou 1999 体系

| 路径 | 样本数 | 用途 |
|---|---:|---|
| `data/zhou1999_lf/train.csv` | 1600 | YODEL LF 预训练 |
| `data/zhou1999_lf/test.csv` | 400 | LF 评估 |
| `data/zhou1999_hf/train_scarce.csv` | 30 | HF 微调 |
| `data/zhou1999_hf/eval.csv` | 10 | 早停评估 |
| `data/zhou1999_hf/test.csv` | 10 | 最终测试 |

主要字段：

- `phi`：固体体积分数。
- `d_s_um`：表面平均粒径，单位 um。
- `tau_Pa`：屈服应力，单位 Pa。
- `powder`：AKP 粉末型号。
- `m1_true` / `m1_lf`：生成和验证辅助字段，不作为模型输入。

重要限制：当前 HF 数据并非逐点精确数字化的原始论文数据。生成器使用 Zhou 1999 图形的视觉估算、YODEL 方程标定和模拟测量噪声构造工作数据。正式发表前应使用 WebPlotDigitizer 等工具重新数字化并保留原始点位、标定截图和校验记录。

## 模型权重

| 路径 | 含义 |
|---|---|
| `multi_fidelity/models/low_fidelity/lian_v3_low.pth` | Lian LF 预训练权重 |
| `multi_fidelity/models/high_fidelity/lian_v3_hifi_only.pth` | Lian HF-only 权重 |
| `multi_fidelity/models/high_fidelity/lian_v3_multifidelity.pth` | Lian PI-MFNN 权重 |

训练脚本会直接覆盖上述文件。权重通常包含 `model_state_dict`、训练历史和实验元数据；加载时应使用与训练一致的模型类和隐藏层宽度。

## 结果目录

- `multi_fidelity/results/zhou1999_exp/summary.json`：Zhou 四策略结果。
- `multi_fidelity/results/zhou1999_exp/baselines_summary.json`：Zhou 多方法基线。
- `multi_fidelity/results/zhou1999_exp/*.png`：对应结果图。
- `multi_fidelity/results/multiseed_results.json`：Lian 多种子汇总。
- `multi_fidelity/results/logs/`、`plots/`：运行时生成，默认被 Git 忽略。

结果 JSON 是实验快照，不等同于自动化测试。代码、数据或随机种子逻辑变化后，应重新运行对应实验再引用。

## 后续规划

`agent-requirements.md` 是屈服应力预测 Agent 的需求草案，涉及 FastAPI、LangGraph、SSE 和前端，但仓库当前没有这些实现或依赖。
