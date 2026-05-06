# 屈服应力预测 Agent 实施计划

## 任务列表

- [ ] 1. 搭建项目骨架与基础配置
  - 创建目录结构：`agent/`、`tools/`、`prompts/`、`data/`、`web/static/`、`tests/`
  - 创建 `requirements.txt`，写入所有核心依赖（fastapi、uvicorn、pydantic、torch、numpy、python-dotenv、openai、langgraph）
  - 创建 `.env.example`，列出 `ARK_API_KEY`、`ARK_MODEL`、`ARK_BASE_URL` 占位变量
  - 创建各包的 `__init__.py` 文件
  - 参考需求 7.1–7.6、设计文档目录结构章节

- [ ] 2. 实现 AgentState 数据模型
  - 在 `agent/state.py` 中定义 `AgentState(TypedDict, total=False)`，包含 `raw_input`、`parsed_params`、`parse_method`、`validation_status`、`validation_detail`、`prediction`、`uncertainty`、`physical_context`、`knowledge`、`explanation`、`final_report`、`error_message`、`debug_log`、`ui_flags` 字段
  - 编写 `tests/test_state.py`，验证 TypedDict 字段类型注解完整、`total=False` 使所有字段可选
  - 参考设计文档 AgentState 定义章节

- [ ] 3. 实现输入解析工具（input_parser）
  - 在 `tools/input_parser.py` 中实现 `parse_input(raw_input: str) -> Dict[str, Any]`
    - 策略一（direct）：正则匹配 `Phi=X`、`SP%=Y` / `SP_percent=Y` 格式，直接提取 float
    - 策略二（formula）：正则匹配 `Vp`、`Vw`、`Cement`、`FA`、`SP`（支持带单位后缀如 `503L`、`1293kg`），按公式 `Phi = Vp/(Vp+Vw)`、`SP_percent = SP/(Cement+FA)*100` 换算；确保 SP_percent 输出为百分比数值（0.50 而非 0.005）
    - 策略三（llm）：调用 `LLMClient` 的 `PARSER_SYSTEM_PROMPT`，要求 LLM 返回 `{"phi": float, "sp_percent": float}` JSON；追问场景需将 `last_params` 和对话历史传入 prompt 以支持差量合并
    - 返回统一结构：`{"phi", "sp_percent", "method", "raw_params", "success", "error"}`
    - 缺少任一参数时 `success=False`，填写 `error` 说明
  - 编写 `tests/test_input_parser.py`，覆盖：direct 格式、formula 带单位格式、formula 无单位格式、缺参返回 error、llm 路径（mock LLMClient）、SP_percent 单位换算正确性（Vp=503,Vw=490,Cement=1293,FA=257,SP=8.52 → sp_percent≈0.55）
  - 参考需求 1.1–1.5、设计文档 input_parser 章节

- [ ] 4. 实现输入验证工具（input_validator）
  - 在 `tools/input_validator.py` 中定义 `TRAIN_RANGE`、`PHYSICAL_RANGE` 常量
  - 实现 `validate_input(phi: float, sp_percent: float) -> Dict[str, Any]`，按三级逻辑：
    - 任一参数超出 `PHYSICAL_RANGE` → `status="invalid"`，`label="❌ 无效输入"`，填写 `out_of_physical`
    - 任一参数超出 `TRAIN_RANGE` 但在 `PHYSICAL_RANGE` 内 → `status="extrapolation"`，`label="⚠️ 外推预测"`，填写 `out_of_train` 及边界对比
    - 全部在 `TRAIN_RANGE` 内 → `status="valid"`，`label="✅ 可信预测"`
  - 编写 `tests/test_input_validator.py`，覆盖：训练分布内、phi 轻度外推、sp_percent 外推、phi 超物理范围、两参数同时超范围、边界值精确判断（如 phi=0.458 为 valid）
  - 参考需求 2.1–2.5、设计文档 input_validator 章节

- [ ] 5. 实现 LianPINN_v2 推理封装（pinn_predictor）
  - 在 `tools/pinn_predictor.py` 中实现 `YieldStressPredictor` 类
    - `__init__`：加载 `multi_fidelity/models/high_fidelity/lian_v2_high_exp10_r2_earlystop.pth`，调用 `torch.load(..., weights_only=False)`，`load_state_dict`，`model.eval()`；模型文件不存在时抛出 `FileNotFoundError` 并记录日志
    - `predict(phi, sp_percent)`：构造 `torch.tensor([[phi, sp_percent]], dtype=torch.float32)`，`with torch.no_grad()` 前向传播；模型输出 `tau0` 和 `phi_max` 均为 shape `[1]` 的一维张量，用 `.item()` 取标量；保留 4 位小数，记录推理耗时；返回 `{"tau0_pa", "phi_max", "inference_time_ms"}`
    - **注意**：checkpoint 中不含 `hidden_dim` 字段，实例化时固定使用 `LianPINN_v2(hidden_dim=64)`，不要尝试从 checkpoint 读取该参数
    - 提供模块级单例 `get_predictor()` 函数，首次调用时实例化并缓存
  - 编写 `tests/test_pinn_predictor.py`，覆盖：模型文件存在时正常加载、推理输出字段完整、tau0/phi_max 均为 float、输入典型值（Phi=0.490, SP%=0.60）输出合理范围（tau0 > 0）、模型文件不存在时抛出 FileNotFoundError
  - 参考需求 3.1–3.5、设计文档 pinn_predictor 章节（含模型架构说明）

- [ ] 6. 实现不确定性估计工具（uncertainty）
  - 在 `tools/uncertainty.py` 中定义 `MAE_VALID = 0.115`、`MAE_EXTRAPOLATION = 0.20`
  - 实现 `estimate_uncertainty(tau0: float, validation_status: str) -> Dict[str, Any]`：
    - `valid` → `margin = 0.115`，`description = "基于训练集 MAE 统计（±0.115 Pa）"`
    - `extrapolation` → `margin = 0.20`，`description = "外推区域扩大估计（±0.20 Pa）"`
    - 返回 `{"tau0_mean", "ci_lower", "ci_upper", "margin", "method": "fixed_mae", "description"}`，所有 float 值保留 4 位小数
  - 编写 `tests/test_uncertainty.py`，覆盖：valid 状态 margin=0.115、extrapolation 状态 margin=0.20、ci_lower = tau0 - margin、ci_upper = tau0 + margin、method 字段固定为 "fixed_mae"
  - 参考需求 4.1–4.3、设计文档 uncertainty 章节

- [ ] 7. 实现物理解释规则生成工具（result_explainer）
  - 在 `tools/result_explainer.py` 中定义 `FLOW_THRESHOLDS`（5 档）和 `SP_LEVELS`（3 档）常量
  - 实现 `build_physical_context(phi, sp_percent, tau0, phi_max, validation_status) -> Dict[str, Any]`：
    - 计算 `packing_ratio = phi / phi_max`，生成颗粒间距描述；**注意：实际评估集 packing_ratio 范围为 0.54–0.85，原阈值 >0.85 在真实数据中几乎不会触发**，建议调整分档为：packing_ratio > 0.80 → 颗粒间距小，流动阻力大；0.65–0.80 → 间距适中；< 0.65 → 间距较大，流动自由
    - 按 `SP_LEVELS` 分档生成 `sp_description`
    - 按 `FLOW_THRESHOLDS` 分档生成 `flow_level`
    - 填写 `mape_reference = "训练分布内 MAPE ≈ 18.8%"`，`validation_label`
  - 编写 `tests/test_result_explainer.py`，覆盖：各 packing_ratio 分档、各 SP_LEVELS 分档、各 FLOW_THRESHOLDS 分档、所有必要字段存在
  - 参考需求 5.1–5.4、设计文档 result_explainer 章节

- [ ] 8. 实现 LLM 客户端（llm_client）
  - 在 `agent/llm_client.py` 中实现 `LLMClient` 类，从 `.env` 读取 `ARK_API_KEY`、`ARK_MODEL`、`ARK_BASE_URL`
    - `generate(system_prompt, payload) -> str`：同步调用 OpenAI 兼容 API，失败时调用 `_fallback(payload)`
    - `generate_stream(system_prompt, payload) -> Generator[str, None, None]`：流式调用，逐块 yield，失败时 yield `_fallback` 分块文本
    - `_fallback(payload)`：从 `payload` 的 `physical_context` 拼接规则文本兜底，标注"[注：LLM 服务不可用]"
    - `get_status() -> Dict`：返回 `{mode, error, message}`
  - 编写 `tests/test_llm_client.py`，覆盖：未配置 API Key 时走 fallback、配置后调用成功（mock openai）、流式输出 yield 多个 chunk、fallback 输出包含物理上下文字段
  - 参考设计文档 llm_client 章节、错误处理章节

- [ ] 9. 实现系统提示词（system_prompt）
  - 在 `prompts/system_prompt.py` 中编写以下提示词常量：
    - `PROMPT_COMMON_RULES`：通用规则（禁止编造、中文输出、禁用 Markdown 特殊格式）
    - `PARSER_SYSTEM_PROMPT`：要求 LLM 从自然语言提取 `{"phi": float, "sp_percent": float}` JSON；需包含上下文槽位（`{last_params}`、`{history}`）供追问差量合并使用（需求 6.3）
    - `EXPLAINER_SYSTEM_PROMPT`：基于 `physical_context` 生成物理解释段落
    - `REPORTER_SYSTEM_PROMPT`：整合预测值、置信区间、解释、验证标签，输出最终回答
    - `CHAT_AGENT_SYSTEM_PROMPT`：追问模式，结合 `last_params`、`prediction`、`history` 回答用户问题
  - 参考设计文档提示词章节、需求 5.5、6.2–6.3

- [ ] 10. 实现 LangGraph 节点与计算图
  - 在 `agent/nodes.py` 中实现所有节点函数，每个节点通过 `state` 读写，调用对应工具，更新 `debug_log`：
    - `parser_node`：调用 `parse_input()`；解析失败时填写 `error_message`，设置 `validation_status="invalid"`
    - `validator_node`：调用 `validate_input()`，填充 `validation_status`、`validation_detail`、`ui_flags`
    - `predictor_node`：调用 `get_predictor().predict()`，填充 `prediction`
    - `uncertainty_node`：调用 `estimate_uncertainty()`，填充 `uncertainty`
    - `explainer_node`：调用 `build_physical_context()` 后调用 `LLMClient.generate(EXPLAINER_SYSTEM_PROMPT, ...)`，填充 `physical_context`、`explanation`
    - `reporter_node`：调用 `LLMClient.generate(REPORTER_SYSTEM_PROMPT, ...)`，填充 `final_report`
    - `error_reporter_node`：直接从 `validation_detail` 或 `error_message` 生成拒绝说明，填充 `final_report`
    - `_route_after_validation`：`invalid` → `"error_reporter"`，其他 → `"predictor"`
  - 在 `agent/graph.py` 中用 `StateGraph(AgentState)` 组装节点，添加条件路由，编译为 `compiled_graph`；提供 `_FallbackCompiledGraph` 降级实现（langgraph 不可用时顺序执行）
  - 编写 `tests/test_graph_flow.py`，覆盖：valid 输入走完整 6 节点路径、invalid 输入短路到 error_reporter、extrapolation 输入正常执行并在 final_report 中含警告文字（均 mock 模型和 LLM）
  - 参考设计文档 LangGraph 工作流章节、节点实现章节

- [ ] 11. 实现追问上下文构建（chat_runtime）
  - 在 `agent/chat_runtime.py` 中实现：
    - `build_followup_payload(message, last_result, history) -> Dict`：组装供 `CHAT_AGENT_SYSTEM_PROMPT` 使用的 payload，包含 `question`、`last_params`、`prediction`、`uncertainty`、`validation_label`、`history`（最近 6 条）
    - `is_new_prediction_request(message, last_params, llm_client) -> bool`：判断追问是否需要触发新一轮预测（如"改为 Phi=0.49"），通过 LLM 判断或简单关键词检测
  - 参考需求 6.2–6.3、设计文档会话存储章节

- [ ] 12. 实现 FastAPI 后端（web_app）
  - 在 `web_app.py` 中实现完整后端：
    - `SESSIONS` 全局字典，结构含 `result`、`history`、`last_params`、`last_active`
    - `POST /api/sessions`：接受 `SessionStartRequest(message: str)`，创建 session，返回 `session_id`
    - `GET /api/demo-cases`：返回 `data/demo_cases.py` 中的 `DEMO_CASES`
    - `POST /api/chat/stream`：接受 `ChatRequest(session_id, message)`，返回 `StreamingResponse`（SSE）
      - 首轮（`result is None`）：依次 emit phase 事件（parse/validate/predict/uncertainty/explain/answer），流式 yield `answer` delta，最终 emit `done` payload
      - 追问轮：判断是否触发新预测；若是则重走首轮流程；若否则用 `CHAT_AGENT_SYSTEM_PROMPT` 直接流式回答
    - `GET /`：返回 `web/static/agent_console.html`
    - 所有 SSE 事件通过 `sse_message(payload) -> str` 格式化为 `data: {...}\n\n`
    - session_id 不存在时返回 HTTP 404
  - 在 `run_web.py` 中用 `uvicorn.run("web_app:app", host="0.0.0.0", port=8000, workers=1)` 启动
  - 编写 `tests/test_web_app.py`，覆盖：创建会话返回 session_id、demo-cases 返回正确结构、chat/stream 首轮 SSE 事件序列包含 phase+answer+done、404 场景、Pydantic 校验失败返回 422
  - 参考需求 7.1–7.6、设计文档 FastAPI 接口章节

- [ ] 13. 实现预置 Demo 案例（demo_cases）
  - 在 `data/demo_cases.py` 中实现 `DEMO_CASES` 字典，包含 4 个案例：`typical`（训练分布内）、`high_phi`（轻度外推）、`low_sp`（外推）、`raw_mix`（原始配合比格式，含带单位和无单位两种 raw_text）
  - 参考设计文档 Demo 案例章节、需求 8.4

- [ ] 14. 实现前端页面（agent_console.html）
  - 在 `web/static/agent_console.html` 中实现单文件 HTML/CSS/JS 页面：
    - 左侧参数输入面板：Tab 切换"直接输入（Phi/SP%）"和"原始配合比（Vp/Vw/Cement/FA/SP）"两种模式（需求 8.1）
    - Demo 案例按钮区：点击一键填充对应参数（需求 8.4）
    - 发送按钮触发 `POST /api/sessions` 创建会话，再 `POST /api/chat/stream` 开启 SSE
    - 中间对话窗口：流式渲染 phase 进度提示和 answer 文本（需求 8.3）
    - 右侧结果卡片：展示 τ₀、φ_max、置信区间、验证状态标签，三种状态用不同颜色/图标区分（需求 8.2、8.5）
    - 追问输入框：复用同一 session_id 发送 `POST /api/chat/stream`
  - 参考需求 8.1–8.5、设计文档 SSE 事件类型章节

- [ ] 15. 端到端集成测试
  - 在 `tests/test_integration.py` 中编写集成测试（使用 `httpx.AsyncClient` + mock 模型）：
    - 典型输入完整流程：Phi=0.490, SP%=0.60 → SSE done payload 含 tau0_pa、phi_max、ci_lower、ci_upper、validation_label="✅ 可信预测"
    - 外推输入流程：Phi=0.510 → done payload 含 validation_label="⚠️ 外推预测"，answer 文本含警告
    - 无效输入拒绝：Phi=0.70 → done payload 含 error 或 answer 含"❌"，不调用模型
    - 原始配合比换算：输入 `Vp=503, Vw=490, Cement=1293, FA=257, SP=8.52` → 验证 parsed_params 中 `phi≈0.5065`、`sp_percent≈0.5497`（约 0.55，百分比数值而非小数）
    - LLM 降级：mock LLM 抛出异常 → final_report 含"[注：LLM 服务不可用]"
    - 追问差量合并：首轮 Phi=0.490 SP%=0.60，追问"改为 SP%=0.8" → 新预测使用 phi=0.490, sp_percent=0.80
  - 参考设计文档集成测试关键场景章节、需求 1–7
