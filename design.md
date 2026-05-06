# 屈服应力预测 Agent 设计文档

## 概述

本系统基于 **LianPINN_v2（exp10_r2_earlystop）** 物理信息神经网络，构建一个可对话的屈服应力预测 Agent。整体架构参照 viscosity_agent 的技术体系，采用 **FastAPI + LangGraph + SSE 流式输出 + LLM 解释层** 的分层设计，以 Python 为主语言，支持多轮对话追问。

核心目标：用户输入浆体配合比参数，Agent 自动完成解析 → 验证 → 推理 → 不确定性估计 → 物理解释，以流式方式返回 τ₀ 预测值、φ_max 中间量、置信区间及自然语言解读。

---

## 架构

### 整体分层架构

```
┌─────────────────────────────────────────────────────┐
│                  前端展示层（Web UI）                  │
│         HTML/CSS/JS + SSE 流式接收 + 结果渲染          │
└────────────────────────┬────────────────────────────┘
                         │ HTTP / SSE
┌────────────────────────▼────────────────────────────┐
│               后端服务层（FastAPI）                    │
│      会话管理 / 请求校验 / SSE 流式响应 / 路由          │
└────────────────────────┬────────────────────────────┘
                         │ 函数调用
┌────────────────────────▼────────────────────────────┐
│            Agent 编排层（LangGraph）                  │
│  Parser → Validator → Predictor → Uncertainty →     │
│  Explainer → Reporter                               │
└──────┬──────────────────────────────────────────────┘
       │ 调用工具
┌──────▼──────────────────────────────────────────────┐
│                    工具层                             │
│  input_parser │ input_validator │ pinn_predictor    │
│  uncertainty_estimator │ result_explainer           │
│  knowledge_base（可选 RAG）                          │
└──────┬──────────────────────────────────────────────┘
       │ 调用
┌──────▼──────────────────────────────────────────────┐
│                  模型 & 服务层                         │
│  LianPINN_v2（PyTorch）│ LLMClient（豆包 / 兼容 API）  │
└─────────────────────────────────────────────────────┘
```

### LangGraph 工作流拓扑

```
START
  ↓
parser_node（输入解析）
  ↓
validator_node（范围验证）
  ↓
[条件路由：是否为无效输入？]
  ├─→ error_reporter_node（❌ 无效输入，直接返回错误）→ END
  └─→ predictor_node（LianPINN_v2 推理）
        ↓
      uncertainty_node（不确定性估计）
        ↓
      explainer_node（物理解释 + LLM 生成）
        ↓
      reporter_node（整合最终回答）
        ↓
      END
```

---

## 组件与接口

### 1. 目录结构

```
rheological_agent/
├── agent/
│   ├── __init__.py
│   ├── state.py            # AgentState TypedDict 定义
│   ├── graph.py            # LangGraph 图组装
│   ├── nodes.py            # 各节点实现
│   ├── chat_runtime.py     # 追问上下文构建
│   └── llm_client.py       # LLM 调用封装
├── tools/
│   ├── __init__.py
│   ├── input_parser.py     # 输入解析（含 LLM 自然语言解析）
│   ├── input_validator.py  # 三级范围验证
│   ├── pinn_predictor.py   # LianPINN_v2 推理封装
│   ├── uncertainty.py      # 不确定性估计
│   ├── result_explainer.py # 物理解释规则生成
│   └── knowledge_base.py   # PDF RAG 检索（可选）
├── prompts/
│   ├── __init__.py
│   └── system_prompt.py    # 各阶段系统提示词
├── data/
│   └── demo_cases.py       # 预置示例配合比
├── multi_fidelity/
│   └── models/
│       └── high_fidelity/
│           └── lian_v2_high_exp10_r2_earlystop.pth  # 模型权重
├── web/
│   └── static/
│       └── agent_console.html  # 前端页面
├── web_app.py              # FastAPI 入口
├── run_web.py              # 启动脚本
├── .env                    # 环境变量（API Key 等）
└── requirements.txt
```

### 2. AgentState 定义

```python
# agent/state.py
class AgentState(TypedDict, total=False):
    # ── 输入 ──
    raw_input: str                        # 用户原始输入文本
    parsed_params: Dict[str, float]       # 解析后的 {phi, sp_percent}
    parse_method: str                     # "direct" | "formula" | "llm"

    # ── 验证 ──
    validation_status: str                # "valid" | "extrapolation" | "invalid"
    validation_detail: Dict[str, Any]     # 超出范围的参数详情

    # ── 推理 ──
    prediction: Dict[str, Any]            # {tau0_pa, phi_max, inference_time_ms}

    # ── 不确定性 ──
    uncertainty: Dict[str, Any]           # {tau0_mean, ci_lower, ci_upper, margin, method, description}

    # ── 解释 ──
    physical_context: Dict[str, Any]      # 规则生成的物理上下文
    knowledge: Dict[str, Any]             # RAG 检索结果（可选）
    explanation: str                      # LLM 生成的解释文本

    # ── 输出 ──
    final_report: str                     # 最终整合回答
    error_message: str                    # 错误信息（无效输入时）

    # ── 调试 ──
    # thought_log 与 tool_log 合并为统一的 debug_log，避免职责重叠
    debug_log: List[Dict[str, Any]]       # 每条格式: {node, action, detail, ts}
    ui_flags: Dict[str, Any]              # {status_level, parse_method}
```

### 3. 各工具模块接口

#### tools/input_parser.py

```python
def parse_input(raw_input: str) -> Dict[str, Any]:
    """
    解析用户输入，返回：
    {
        "phi": float,
        "sp_percent": float,
        "method": "direct" | "formula" | "llm",
        "raw_params": Dict,   # 原始解析到的参数
        "success": bool,
        "error": str          # 解析失败时的说明
    }
    """
```

三种解析策略（按优先级尝试）：
1. **direct**：正则匹配 `Phi=X, SP%=Y` 格式，直接提取
2. **formula**：正则匹配 Vp/Vw/Cement/FA/SP，按公式换算
3. **llm**：调用 LLMClient，以结构化 JSON 格式提取参数

#### tools/input_validator.py

```python
TRAIN_RANGE = {
    "phi": (0.458, 0.504),
    "sp_percent": (0.40, 1.00),
}
PHYSICAL_RANGE = {
    "phi": (0.35, 0.60),
    "sp_percent": (0.10, 2.00),
}

def validate_input(phi: float, sp_percent: float) -> Dict[str, Any]:
    """
    返回：
    {
        "status": "valid" | "extrapolation" | "invalid",
        "label": "✅ 可信预测" | "⚠️ 外推预测" | "❌ 无效输入",
        "out_of_train": List[str],   # 超出训练分布的参数名
        "out_of_physical": List[str],# 超出物理范围的参数名
        "detail": Dict               # 各参数的边界对比
    }
    """
```

#### tools/pinn_predictor.py

**模型架构说明（LianPINN_v2）：**

```
网络结构: Linear(2→64) → Tanh → Linear(64→64) → Tanh → Linear(64→64) → Tanh → Linear(64→1)
输入:     x，shape [batch, 2]，dtype float32，列顺序 [Phi, SP_percent]
          - Phi: 固体体积分数，范围 [0.35, 0.60]
          - SP_percent: 减水剂掺量百分比数值，范围 [0.10, 2.00]
            （0.5% 掺量对应传入值 0.50，不是 0.005）
内部归一化: x_norm = x * [2.0, 1.5]（模型 forward 内部完成，调用方无需处理）
中间量:   raw_phi_max → sigmoid 解码为 phi_max_pred ∈ [phi+0.05, 0.95]
输出1:    tau0_pred，shape [batch]，单位 Pa，由论文公式计算：
          τ₀ = 0.72 × φ³ / [φ_max × (φ_max − φ)]
输出2:    phi_max_pred，shape [batch]，无量纲，表示虚拟最大堆积分数
固定参数: m1 = 0.72 Pa（论文标定值，不参与训练）
总参数量: 8,577
```

```python
class YieldStressPredictor:
    MODEL_PATH = "multi_fidelity/models/high_fidelity/lian_v2_high_exp10_r2_earlystop.pth"

    def __init__(self):
        # 单例加载，整个进程复用
        # 注意：LianPINN_v2 不含 Dropout 层，不支持 MC Dropout
        ...

    def predict(self, phi: float, sp_percent: float) -> Dict[str, Any]:
        """
        构造输入: torch.tensor([[phi, sp_percent]], dtype=torch.float32)
        调用:     tau0, phi_max = model(x)
        返回：
        {
            "tau0_pa": float,          # 屈服应力预测值，单位 Pa
            "phi_max": float,          # 虚拟最大堆积分数（中间量）
            "inference_time_ms": float
        }
        """
```

> ⚠️ **MC Dropout 不适用**：`LianPINN_v2` 为纯 MLP + Tanh 结构，不含任何 Dropout 层，多次前向传播结果完全相同。`predict_with_mc_dropout` 方法已从接口中移除，不确定性统一由 `uncertainty.py` 的固定误差法提供。

#### tools/uncertainty.py

```python
def estimate_uncertainty(
    tau0: float,
    validation_status: str,
) -> Dict[str, Any]:
    """
    采用固定误差法（LianPINN_v2 不含 Dropout，无法使用 MC Dropout）：
      - valid        → ±0.115 Pa（训练分布内实测 MAE）
      - extrapolation → ±0.20 Pa（外推区域扩大估计）

    返回：
    {
        "tau0_mean": float,   # 等于点预测值 tau0
        "ci_lower": float,    # tau0 - margin
        "ci_upper": float,    # tau0 + margin
        "margin": float,      # 误差边界
        "method": "fixed_mae",
        "description": str    # 如 "基于训练集 MAE 统计（±0.115 Pa）"
    }
    """
```

#### tools/result_explainer.py

```python
# 注：阈值基于论文 Table 6 数据分布（τ₀ 范围 0.19–1.95 Pa）的经验分级，非行业标准
FLOW_THRESHOLDS = [
    (0.5,  "极佳流动性，几乎无屈服应力"),
    (1.0,  "良好流动性，适合泵送"),
    (1.5,  "中等流动性，泵送临界值"),
    (3.0,  "流动性偏低，需较大驱动力"),
    (float("inf"), "流动性差，不适合泵送"),
]

SP_LEVELS = [
    (0.4,  "低掺量，分散效果弱"),
    (0.7,  "中等掺量，分散效果适中"),
    (float("inf"), "高掺量，分散效果强"),
]

def build_physical_context(
    phi: float,
    sp_percent: float,
    tau0: float,
    phi_max: float,
    validation_status: str,
) -> Dict[str, Any]:
    """
    返回规则生成的物理上下文，供 LLM 生成自然语言解释：
    {
        "packing_ratio": float,          # phi / phi_max，实际数据范围约 0.54–0.85
        "packing_description": str,      # 颗粒间距描述
        "sp_description": str,           # 减水剂效果描述
        "flow_level": str,               # 流动性等级描述
        "mape_reference": str,           # 模型误差参考
        "validation_label": str,         # 可信度标签
    }
    """
```

### 4. LangGraph 节点实现

```python
# agent/nodes.py

def parser_node(state: AgentState) -> AgentState:
    """调用 parse_input()，填充 parsed_params, parse_method"""

def validator_node(state: AgentState) -> AgentState:
    """调用 validate_input()，填充 validation_status, validation_detail"""

def predictor_node(state: AgentState) -> AgentState:
    """调用 YieldStressPredictor.predict()，填充 prediction"""

def uncertainty_node(state: AgentState) -> AgentState:
    """调用 estimate_uncertainty()，填充 uncertainty"""

def explainer_node(state: AgentState) -> AgentState:
    """
    1. 调用 build_physical_context() 生成规则上下文
    2. 可选：调用 retrieve_knowledge() 检索 RAG 证据
    3. 调用 LLMClient.generate() 生成自然语言解释
    填充 physical_context, knowledge, explanation
    """

def reporter_node(state: AgentState) -> AgentState:
    """调用 LLMClient.generate() 整合最终回答，填充 final_report"""

def error_reporter_node(state: AgentState) -> AgentState:
    """直接返回无效输入错误信息，填充 error_message"""

def _route_after_validation(state: AgentState) -> str:
    """路由：invalid → error_reporter，其他 → predictor"""
    return "error_reporter" if state.get("validation_status") == "invalid" else "predictor"
```

### 5. FastAPI 接口

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/` | 返回前端页面 |
| GET | `/api/demo-cases` | 返回预置示例配合比 |
| POST | `/api/sessions` | 创建会话，返回 session_id |
| POST | `/api/chat/stream` | SSE 流式对话（首轮分析 + 追问） |

**Pydantic 请求模型：**

```python
class SessionStartRequest(BaseModel):
    message: str                    # 用户初始输入

class ChatRequest(BaseModel):
    session_id: str
    message: str
```

**SSE 事件类型：**

| type | 说明 |
|------|------|
| `phase` | 阶段进度提示（target 见下表） |
| `answer` | 最终回答流式 delta |
| `done` | 完成，携带完整结构化结果 payload |
| `error` | 错误事件 |

**phase 事件 target 值与 LangGraph 节点对应关系：**

| SSE target | LangGraph 节点 | 含义 |
|-----------|---------------|------|
| `"parse"` | `parser_node` | 输入解析中 |
| `"validate"` | `validator_node` | 范围验证中 |
| `"predict"` | `predictor_node` | 模型推理中 |
| `"uncertainty"` | `uncertainty_node` | 不确定性估计中 |
| `"explain"` | `explainer_node` | 物理解释生成中 |
| `"answer"` | `reporter_node` | 整合最终回答中 |

**done payload 结构：**

```json
{
  "type": "done",
  "payload": {
    "answer": "...",
    "tau0_pa": 0.74,
    "phi_max": 0.69,
    "ci_lower": 0.625,
    "ci_upper": 0.855,
    "validation_status": "valid",
    "validation_label": "✅ 可信预测",
    "parse_method": "direct",
    "parsed_params": {"phi": 0.503, "sp_percent": 0.5}
  }
}
```

### 6. LLM 提示词设计

```python
# prompts/system_prompt.py

PROMPT_COMMON_RULES = """
你是水泥浆体流变学专家。规则：
1. 仅基于输入的预测值、物理上下文和知识库证据作答，禁止编造数据
2. 证据不足时明确说明不确定性
3. 输出中文，禁用 Markdown 表格/代码块/井号标题
4. 简洁专业，适合工程师阅读
5. 可引用"第X页"证据，但禁止大段照抄
"""

EXPLAINER_SYSTEM_PROMPT  # 生成物理解释段落
REPORTER_SYSTEM_PROMPT   # 整合最终回答（含预测值、置信区间、解释、警告）
CHAT_AGENT_SYSTEM_PROMPT # 回答追问（结合历史对话和上次预测结果）
PARSER_SYSTEM_PROMPT     # 从自然语言提取参数（返回 JSON）
```

---

## 数据模型

### 核心参数范围

```python
# 训练分布范围（高保真训练集）
TRAIN_RANGE = {
    "phi":        (0.458, 0.504),
    "sp_percent": (0.40,  1.00),
}

# 物理合理范围
PHYSICAL_RANGE = {
    "phi":        (0.35, 0.60),
    "sp_percent": (0.10, 2.00),
}

# 模型性能基准
MODEL_BENCHMARK = {
    "mae_pa":  0.115,    # 训练分布内 MAE
    "mape_pct": 18.8,    # 训练分布内 MAPE
    "extrapolation_mae_pa": 0.20,  # 外推区域估计误差
}
```

### 会话存储结构

```python
SESSIONS: Dict[str, Dict[str, Any]] = {
    "<session_id>": {
        "result": AgentState | None,   # 最近一次完整推理结果
        "history": [                   # 对话历史（最近 6 条）
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
        ],
        "last_params": {               # 最近一次解析成功的参数（用于追问差量合并）
            "phi": float,
            "sp_percent": float,
        },
        "last_active": float,          # Unix 时间戳，用于过期清理
    }
}
```

> ⚠️ **并发与过期限制**：`SESSIONS` 为进程内内存字典，仅支持**单进程部署**（`uvicorn --workers 1`）。多 worker 部署时会话无法跨进程共享，需迁移到 Redis。建议设置会话过期时间 30 分钟（无活动后清理），防止内存泄漏。

### Demo 案例

```python
DEMO_CASES = {
    "typical": {
        "label": "典型配比（训练分布内）",
        "phi": 0.490,
        "sp_percent": 0.60,
    },
    "high_phi": {
        "label": "高固含（轻度外推）",
        "phi": 0.510,
        "sp_percent": 0.55,
    },
    "low_sp": {
        "label": "低减水剂掺量（外推）",
        "phi": 0.475,
        "sp_percent": 0.30,
    },
    "raw_mix": {
        "label": "原始配合比输入示例",
        # 带单位格式，formula 正则须能匹配 "503L"、"1293kg" 等带单位字符串
        # 正则建议: r'Vp\s*=\s*([\d.]+)\s*L?' 忽略单位后缀
        "raw_text": "Vp=503L, Vw=490L, Cement=1293kg, FA=257kg, SP=8.52kg",
        # 等价的无单位格式（测试用）
        "raw_text_plain": "Vp=503, Vw=490, Cement=1293, FA=257, SP=8.52",
    },
}
```

---

## 错误处理

| 场景 | 处理策略 |
|------|----------|
| 输入解析失败（缺少必要参数） | 返回 `parse_error`，提示用户补充，不执行推理 |
| 输入超出物理范围（❌ 无效） | `error_reporter_node` 直接返回拒绝说明，不加载模型 |
| 模型文件不存在 | 启动时抛出 `FileNotFoundError`，日志记录，API 返回 503 |
| 模型推理异常（PyTorch 错误） | 捕获异常，返回 500 错误，记录到 tool_log |
| LLM API 不可用 | 降级到 Fallback 模式：直接拼接规则文本，不调用 LLM |
| LLM 返回空内容 | 使用结构化规则文本兜底，标注"LLM 未返回内容" |
| session_id 不存在 | HTTP 404，附带说明 |
| RAG 检索失败 | 静默降级，仅用规则文本，不影响主流程 |

### LLM 降级策略

```python
# agent/llm_client.py 中的 _fallback()
# 当 LLM 不可用时，直接返回由规则生成的物理上下文文本：
def _fallback(self, payload: Dict) -> str:
    ctx = payload.get("physical_context", {})
    return (
        f"预测结果：τ₀ = {payload.get('tau0_pa')} Pa，φ_max = {payload.get('phi_max')}。"
        f"{ctx.get('packing_description', '')} "
        f"{ctx.get('sp_description', '')} "
        f"{ctx.get('flow_level', '')} "
        f"可信度：{ctx.get('validation_label', '')}。"
        f"模型误差参考：训练分布内 MAPE ≈ 18.8%。"
        "[注：LLM 服务不可用，以上为规则生成文本]"
    )
```

---

## 测试策略

### 单元测试

| 测试文件 | 覆盖范围 |
|----------|----------|
| `tests/test_input_parser.py` | 三种解析方式（direct/formula/llm mock）、缺参错误 |
| `tests/test_input_validator.py` | 三级状态边界值、超出范围参数定位 |
| `tests/test_pinn_predictor.py` | 模型加载、推理输出格式、输入边界 |
| `tests/test_uncertainty.py` | 固定误差法计算（valid/extrapolation 两档）、ci_lower/ci_upper 边界值 |
| `tests/test_result_explainer.py` | 各阈值分段逻辑、物理上下文字段完整性 |
| `tests/test_graph_flow.py` | LangGraph 完整流程（mock 模型+LLM）、路由分支 |
| `tests/test_web_app.py` | FastAPI 端点响应格式、session 管理、SSE 事件序列 |
| `tests/test_llm_client.py` | LLM 调用成功/失败/降级、流式输出 |

### 集成测试关键场景

1. **典型输入端到端**：Phi=0.490, SP%=0.60 → 完整流程 → 验证输出字段完整
2. **外推输入警告**：Phi=0.510 → 验证 ⚠️ 标签出现在输出中
3. **无效输入拒绝**：Phi=0.70 → 验证返回 ❌ 且不调用模型
4. **原始配合比换算**：输入 Vp/Vw/Cement/FA/SP → 验证换算结果正确
5. **LLM 降级**：mock LLM 报错 → 验证规则文本兜底生效
6. **追问流程**：首轮预测后追问"改为 SP%=0.8 会怎样" → 验证上下文保持

### 测试工具

- `pytest` + `pytest-asyncio`（FastAPI 异步测试）
- `unittest.mock`（mock LLM、mock 模型推理）
- `httpx.AsyncClient`（FastAPI 测试客户端）

---

## 关键设计决策

### 1. 不确定性估计仅采用固定误差法，MC Dropout 已移除
**理由：** `LianPINN_v2` 网络结构为纯 MLP + Tanh，不含任何 Dropout 层。若在 `model.train()` 模式下多次前向传播，结果完全相同（标准差为 0），MC Dropout 无法产生有效的随机性，实现后会静默给出错误的零方差置信区间。固定误差法基于实测 MAE（0.115 Pa），物理含义明确，实现简单可靠，已作为唯一方案。

### 2. LangGraph 作为 Agent 编排框架
**理由：** 参照 viscosity_agent 的成熟实践，LangGraph 的有向图结构清晰分离各能力节点，便于独立测试和替换。条件路由（无效输入直接短路）避免无效的模型加载开销。

### 3. 模型单例加载
**理由：** PyTorch 模型加载（含权重反序列化）耗时较长，应在服务启动时完成一次加载，后续所有请求复用同一实例，避免每次推理重复 I/O。

### 4. SSE 流式输出分阶段
**理由：** 模型推理 + LLM 生成合计可能需要数秒，分阶段流式输出（解析 → 验证 → 推理 → 解释 → 回答）让用户感知到系统在持续工作，提升体验。

### 5. RAG 为可选模块
**理由：** 核心预测能力不依赖知识库。RAG 需要 PDF 文献资源和向量模型，增加部署复杂度。设计为可选模块，知识库未配置时静默降级，不影响主流程。

---

## 依赖清单

```
# requirements.txt
fastapi>=0.110.0
uvicorn>=0.29.0
pydantic>=2.0.0
torch>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
openai>=1.0.0          # LLM 调用（OpenAI 兼容接口）
langgraph>=0.1.0       # Agent 编排
# 可选 RAG 依赖
chromadb>=0.4.0
pymupdf>=1.23.0
rank-bm25>=0.2.2
```
