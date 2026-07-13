# SmartStress 论文—代码对齐审计与执行计划

更新：2026-07-13（Asia/Shanghai）  
状态：**等待工程与科研口径确认，尚未开始业务实现**

## 1. 目标与完成定义

目标不是把一个分类函数塞进现有节点，而是让本仓库能够真实、可复现地支持论文的完整叙述：

1. 700 Hz ECG 经 20 秒窗口、1 秒步长和 12 维特征处理后，由 Attention-DNN 输出压力概率。
2. PhysioSense 同时输出模型版本、数据质量、特征值和 SHAP 生理驱动因素。
3. MindCare 使用用户话语、压力概率和生理驱动因素识别压力源，并据此形成 RAG 查询与有证据约束的回复。
4. Meta-Reflective Orchestrator 根据生理紧迫度、语义明确度、检索质量和用户反馈决定继续支持、重新推理或提出 TaskRelief。
5. 任何环境改变都必须经过明确的人类确认；拒绝后进入 refine，而不是静默终止或误执行。
6. 论文中的每个结果都能追溯到版本化代码、配置、权重、数据划分和机器可读实验产物。
7. 论文 PDF 能从当前源码重新编译，标题、表格、图片和结果与源码一致。

只有以上链路和证据同时成立，才视为“代码仓库贴合论文逻辑”。

## 2. 本次审计使用的真源

### 2.1 论文

- 当前叙述真源：`D:\NUS\BMI5101\SmartStress\main.tex`
- 论文仓库提交：`931d600347505bac9fd38ecbaffb6429898a3702`
- `main.tex` SHA-256：`37108B76374B13C4EBCBE384288182AF83DBDA79DE9DD35B09E7036D4BC4C11E`
- 论文仓库目前另有两个未跟踪的 refined 图片；后续不得覆盖。
- `D:\NUS\BMI5101\SmartStress\main.pdf` 经逐页渲染确认是 IEEE 8 页模板，不是 SmartStress 论文。
- 本仓库 `overleaf_project/main.pdf` 同样是 IEEE 模板；`overleaf_project/main.tex` 比论文真源落后 47 行新增/68 行删除，并缺少最新版引用的两张 RGB 系统图。

### 2.2 分类模型

- 只读来源：`D:\NUS\BMI5101\smart-stress-model`
- 当前 HEAD：`53230f5`
- 来源仓库存在用户未提交改动，因此迁移时只读取，不在该仓库修改、清理或提交。
- 与论文 S17 数字完全一致的候选 Attention 权重：
  `Models_CrossVal_Full/fold_14_Attention/epoch_37.pth`
- 候选权重大小：267,297 bytes
- 候选权重 SHA-256：`198CC7CE41D113CCA1CAFB4D2D08F7693D7CF052145E1BD953887A60B3F75161`
- 已在 `torch 2.9.1+cu128` 环境复验 S17：Accuracy 0.9797、Precision 0.9626、Recall 0.9531、F1 0.9578。

### 2.3 当前 Agent 仓库

- 审计基线提交：`f7f5374`
- 当前 PhysioSense 是以 HR 为输入的线性 heuristic，占位实现不是论文模型。
- 当前无标准 `tests/` 单元/集成测试体系；主要依赖手工 smoke 脚本。

## 3. 论文当前叙述的可执行规格

### 3.1 PhysioSense

输入：

- 原始 ECG，论文口径为 700 Hz。
- 20 秒滑动窗口；论文口径为 1 秒 stride。
- 用户中性基线/用户画像。

特征顺序必须固定为：

1. Mean HR（来源代码命名为 Mean Freq）
2. Std HR（来源代码命名为 Std Freq）
3. TINN
4. HRV Index
5. NN50
6. pNN50
7. Mean HRV
8. Std HRV
9. RMSSD
10. FFT Mean
11. FFT Std
12. Sum PSD

模型：

- 每个标量特征经 `Linear(1, 32)` 嵌入。
- 4-head self-attention，`embed_dim=32`。
- 展平为 384 维。
- MLP：384→128→64→16→4→1。
- 各隐藏层包含 BatchNorm、Dropout 0.5、LeakyReLU 0.2。
- 最终 Sigmoid 输出 `P(stress)`。

输出：

- 压力概率和二分类决策。
- 12 维标准化特征及其顺序/单位。
- SHAP 局部贡献、Top-N 正/负向生理驱动因素。
- 模型版本、权重校验和、阈值和输入质量标记。

### 3.2 MindCare

执行顺序：

1. 基于用户话语识别压力源，但不做临床诊断。
2. 将压力源、`P(stress)`、SHAP 驱动因素和用户查询组成结构化检索查询。
3. Top-K 检索经过来源管理的心理支持材料。
4. 使用检索文本和生理上下文构建结构化提示词。
5. 生成支持性、非诊断、可追溯到检索证据的回复。
6. 高风险/危机话语走独立安全升级策略，禁止继续 TaskRelief。

### 3.3 Meta-Reflective Orchestrator 与 TaskRelief

编排状态至少包含：

- physiological urgency；
- physiological drivers / data quality；
- semantic stressor / specificity；
- retrieved evidence / retrieval quality；
- proposed intervention；
- user acceptance；
- refinement reason；
- execution result 和审计轨迹。

控制要求：

- 低于阈值：继续监测，不制造伪造的低压力读数。
- 高于阈值：进入压力源识别和有依据的支持。
- 压力源可行动：提出 TaskRelief，但不直接执行。
- 仅精确、结构化的 ACCEPTED 状态可以执行。
- REJECTED/CANCELLED：保留当前证据，记录原因，回到 refine。
- 数据无效/模型失败：降级为文本支持并显式标记，不把异常伪装成正常预测。

## 4. 已确认的不一致与风险

### 4.1 P0：必须在实现前决定

| 编号 | 证据 | 影响 |
| --- | --- | --- |
| P0-1 | 两份 `main.pdf` 都是 IEEE 模板 | 当前没有可交付、可视觉核验的论文 PDF |
| P0-2 | 95.78% F1 是 S17 单一留出折；15 折 Attention 平均 F1 是 80.82% | 论文把单折结果表述为总体 WESAD 性能，证据口径不成立 |
| P0-3 | LOSO 脚本用外层测试 subject 同时选最佳 epoch，再在同一数据上报告结果 | 存在测试集用于模型选择的数据泄漏，现有指标偏乐观 |
| P0-4 | StressID adaptation 脚本加载的是标准 DNN checkpoints，不是 Attention-DNN | 论文把 Attention 主模型与 DNN 外部验证结果串成同一检测器，模型口径不一致 |
| P0-5 | SHAP 脚本从 `Model_testing.py` 导入标准 DNN，并非 Attention 模型 | 论文的 Attention 可解释性主张没有对应实验 |
| P0-6 | 中性基线切片终点把已有的 700 Hz 标签索引再次乘以 700，实际扩展到记录末尾 | “subject-specific neutral baseline”与实现不符 |
| P0-7 | 长时基线特征直接与 20 秒窗特征相除；S17 中性特征中位数明显远离 1 | 模型学到窗口长度/统计量尺度差异，不能解释为相对中性偏移 |
| P0-8 | 所谓 RMSSD 实际为 `sqrt(mean(RR^2))`；NN50 为所有 RR 两两比较而非相邻差 | 特征名称、定义与生理学标准不一致；修正后旧权重失效 |
| P0-9 | 当前 Agent 只用 HR heuristic，且无输入时写入 0.1 | 会制造论文链路不存在的压力概率和历史记录 |

### 4.2 P1：核心闭环缺口

1. API 只有无约束 `values: Dict[str, Any]`，没有 ECG 采样率、窗口长度、基线和预计算特征的强类型契约。
2. State 没有特征、SHAP、质量、阈值、模型版本和结构化生理证据。
3. `> 0.9` 在 MindCare 三处硬编码；论文阈值 `tau` 未校准、未配置、未版本化。
4. MindCare 的实际 RAG 查询主要只用用户文本，没有把压力概率和 SHAP 驱动因素写进 query。
5. 检索结果只返回字符串，不保留 document id、similarity、source、section，无法做 groundedness 审计。
6. 回复提示词允许参考检索材料，但没有强制逐条证据绑定，也没有 unsupported-claim 检查。
7. 当前没有独立 meta-reflective node；编排主要是 MindCare 后的条件路由。
8. 拒绝分支没有真正重新形成支持/干预方案。
9. 确认解析使用子串命中，单字符 `y`/`n` 可能把普通话语误判为同意或拒绝。
10. TaskRelief 只记录 mock calendar 文本；缺少明确的 dry-run/real adapter 边界和幂等键。
11. 执行后仍保留高压力概率，现有图可能立刻再次触发压力源探索。
12. 图定义有重复 import，异常恢复和 checkpoint 继续语义没有标准集成测试。

### 4.3 P2：实验与工程质量缺口

1. 没有黄金样本保证“迁移前后同一 12 维输入得到同一概率”。
2. 没有 ECG 数值边界、采样率错误、R-peak 失败、NaN/Inf、过短信号等测试。
3. 没有模型缺失/哈希错误/版本不兼容/CPU fallback 测试。
4. 没有概率校准、Brier score、ECE、可靠性图和阈值敏感性分析。
5. 20 秒窗、1 秒 stride 造成强自相关；当前结果按 window 统计但未报告 episode/subject 级置信区间。
6. RAG ablation 只用 TF-IDF/BERTScore，且绝对值很低；没有检索 Recall@K/nDCG、回答忠实度、引用正确性或盲评。
7. 没有危机话语、安全拒绝、误触发 TaskRelief 的系统级测试。
8. 无 CI、统一配置 schema、实验 manifest、权重 provenance 和可重复环境锁定。

### 4.4 原始数据抽样与运行时量化

在不修改来源仓库的前提下，对 S17 原始 WESAD ECG 做了只读抽样：

- ECG 与标签均为 4,144,000 个 700 Hz 样本，说明标签索引已经与 ECG 样本对齐。
- neutral 标签索引为 64,564–891,263；旧代码再把终点乘 700，所请求终点是 ECG 长度的 150.55 倍，因此 NumPy 实际切到记录末尾。
- 这个所谓 baseline 中只有 20.27% 是 neutral；其余包含 stress、amusement、meditation、其他实验状态和 unlabeled 段。故它不能称为 neutral baseline。
- 各抽取 12 个 neutral/stress 的 20 秒窗口后，neutral 的旧“RMSSD”中位数为 0.9481 s，标准 RMSSD 为 0.1035 s，约相差 8.75 倍；stress 分别为 0.5334 s 和 0.0140 s，约相差 38.58 倍。
- 旧 NN50 统计所有 RR 的两两差异；20 秒 neutral 窗口的旧 NN50 中位数为 288，而标准相邻 RR 定义中位数为 11，二者不是同一指标。

运行时验证：

- Attention-DNN 共 63,361 个参数，checkpoint 为 267,297 bytes。
- CPU 单窗口前向平均约 0.40 ms；2,907 个窗口批量前向约 16.1 ms，因此模型推理不是主要性能瓶颈。
- 使用 32 个旧 S17 样本作为 `DeepExplainer` background 时，加和检查最大残差为 0.01257，超过 SHAP 默认 0.01 容差并失败。
- 使用代表修正后 neutral ratio 的单位向量作为 background 时，最大残差降至 0.001324；8 个样本解释约 56 ms。

结论：corrected 流程若使 neutral-normalized feature 真正以 1 为中心，可同时改善生理语义和快速 SHAP 的 background 定义；但该 background 必须作为版本化实验设计，而不能为了绕过检查临时替换。

## 5. 需要用户确认的设计决策

### D1. 科研兼容路线

可选路线：

- **A — 原样复现**：严格迁移旧特征算法和 S17 Attention 权重，优先复现 95.78%。优点是快；缺点是代码继续不符合论文方法，不能修复科研风险。
- **B — 直接纠正**：修正 neutral baseline、RMSSD、NN50 等定义，重训并只保留新模型。优点是方法正确；缺点是旧表格和 SHAP 全部失效，短期无法复现现论文数字。
- **C — 双轨迁移（建议）**：`legacy_v1` 明确标为论文旧结果复现路径；`corrected_v2` 作为默认研究/部署候选，修正后重训。论文最终按新证据更新，并清楚说明旧结果不可与新结果混用。

### D2. 传感输入契约

建议同时支持两种显式模式：

- `raw_ecg`：生产/论文主路径，包含 samples、sampling_rate_hz、window timestamps 和 neutral baseline reference。
- `normalized_features`：只用于黄金测试、离线回放和已经过同版本预处理的可信上游。

禁止继续以任意 HR/HRV 字典静默进入论文模型。

### D3. 模型发布策略

候选：

- 立即打包 S17 对应 267 KB 权重：只能标记为 legacy reproducibility artifact。
- 将 15 个 Attention 折做平均集成：可减少单折选择，但不解决旧预处理和测试泄漏。
- 修正流程后，用 nested subject validation 选择超参数，再训练明确的 production artifact：建议作为默认运行模型。

建议：C 路线下先打包 legacy 权重用于回归测试；production 默认保持不可用或显式 demo 模式，直到 corrected 权重通过实验门。

候选权重的体积和 CPU 延迟都足够小，选择单折、集成或重训模型应由科研有效性决定，而不是由部署性能决定。

### D4. 阈值和触发策略

建议拆分：

- `classification_threshold`：复现实验时为 0.5。
- `support_activation_policy`：部署时由校准阈值 + 连续窗口持续性 + 数据质量共同决定，不再用单窗口硬编码 0.9。
- 任何 TaskRelief 仍必须经 HITL；低置信/域外输入只提供非行动型支持。

### D5. TaskRelief 范围

建议当前阶段实现：

- 明确的 intervention adapter 接口；
- 默认 `dry_run`；
- 结构化 proposed action、风险级别、可逆性、幂等键；
- 精确同意后才调用 adapter。

真实日历/任务系统连接器应作为独立后续 step，除非本轮明确指定目标系统和权限。

### D6. 论文仓库写权限

本次读取不等于授权修改 `D:\NUS\BMI5101\SmartStress`。建议在实验产物稳定后同步修改 TeX、图表和真实 PDF，并在论文仓库中按小 step 单独提交；开始前需用户明确授权。

## 6. 分步实现与提交计划

以下每个编号默认对应至少一个独立 Git commit；若一个编号内出现两个可独立验证的行为，会继续拆小提交。每次变更同时追加 `.codex/WORKLOG.md`。

### Step 0 — 决策与基线锁定

交付：

- 本审计计划。
- 记录论文、代码、候选权重的 commit/hash。
- 用户确认 D1–D6。

验收：仓库业务代码不变，工作区干净。

复杂度：S。

### Step 1 — 测试骨架与黄金推理样本

交付：

- 建立 `tests/`、pytest 配置和最小 CI/本地测试命令。
- 从 S17 固定一个或多个 12 维输入，记录旧 Attention 权重的确定性概率。
- 测试权重 hash、模型结构和 feature order。

验收：CPU/GPU 容许极小数值误差；迁移模型输出与来源仓库一致。

复杂度：M。

### Step 2 — 版本化 PhysioSense 核心

交付：

- 独立 `physio` 包：feature schema、preprocessor protocol、model loader、inference result。
- 配置化 device、artifact path、hash、阈值和模型版本。
- 缺失/损坏 artifact 直接显式失败，不回退 heuristic。

验收：12 维批量/单窗推理、NaN/shape 错误、CPU fallback 测试通过。

复杂度：M。

### Step 3 — ECG 预处理

若选双轨：

- `legacy_v1` 仅用于复现，并对非标准实现发出 metadata warning。
- `corrected_v2` 使用严格 neutral-window aggregation、标准 RMSSD/NN50/pNN50、明确的频域实现和质量门。
- 采样率不一致时必须按确认策略拒绝或重采样，不能静默当作 700 Hz。

验收：合成 ECG、已知 RR 序列、边界条件和真实 WESAD 窗口黄金测试通过。

复杂度：L；这是最容易改变实验结论的工程点。

### Step 4 — SHAP 与生理证据输出

交付：

- 解释与实际运行的 Attention 权重严格绑定。
- background 策略版本化；优先使用同一预处理版本的 neutral reference/background asset。
- 输出正/负贡献、特征原值、标准化值、基线、plain-language label。
- 解释失败不能改变预测，但必须进入状态和审计。

验收：局部贡献维度为 12；解释模型 hash 与预测模型 hash 相同；Top drivers 可序列化。

复杂度：M–L。

### Step 5 — State/API 契约迁移

交付：

- 强类型 `raw_ecg` / `normalized_features` 请求。
- State/View 增加 inference status、quality、features、drivers、model metadata、decision policy。
- 无新传感输入时不追加虚假概率；模型失败时保留最后一次有效读数并标记 stale。
- 兼容期内对旧 HR payload 返回明确 validation error 或受控 legacy-demo 响应。

验收：Pydantic contract、序列化、checkpoint 恢复、API 错误码测试通过。

复杂度：M–L，可能影响现有前端。

### Step 6 — MindCare 生理—语义—RAG 对齐

交付：

- 独立 stressor extraction schema。
- RAG query builder 明确包含 stressor、概率区间和 Top physiological drivers。
- 检索结果保留 id/source/section/score。
- 回复输出引用关系和 unsupported evidence 标记。
- 危机话语策略在普通 stressor/TaskRelief 之前执行。

验收：mock LLM/RAG 下可确定地验证 query 和 prompt 内容；无证据时不能声称“有证据支持”。

复杂度：L。

### Step 7 — 显式 Meta-Reflective Orchestrator

交付：

- 独立 orchestration decision node 和枚举状态。
- 决策输入覆盖 urgency、quality、semantic specificity、retrieval quality、user feedback。
- ACCEPT/REJECT/CANCEL 使用结构化命令或严格解析。
- REJECT 进入 refine 并保留证据；执行后进入 cooldown/monitoring，而非立即重触发。

验收：状态转移表、循环上限、checkpoint resume 和每条 HITL 分支集成测试通过。

复杂度：L。

### Step 8 — TaskRelief adapter

交付：

- dry-run adapter、action schema、risk/reversibility/idempotency metadata。
- 未确认、重复确认、过期 proposal、危机状态都不能执行。
- 若授权真实连接器，再单独设计权限和回滚。

验收：零误执行测试；重复请求幂等。

复杂度：M（dry-run）/ XL（真实外部系统）。

### Step 9 — 系统测试与运行文档

交付：

- 单元、图路由、API、持久化和端到端测试。
- 模型下载/打包、CPU/GPU、环境变量和数据契约文档。
- README 叙述与实际实现一致，删除乱码和过时示例。

验收：干净环境可复现 install→test→demo；不要求真实 LLM/DB 的测试使用 fake adapter。

复杂度：L。

### Step 10 — 实验重做与证据固化

交付按第 7 节拆分提交。任何新数字先写机器可读 JSON/CSV 和 manifest，再进入论文。

复杂度：XL；工程实现、GPU 运行、统计复核和人工评估需分开估算。

### Step 11 — 论文同步与 PDF 核验

前提：获得论文仓库写权限且实验门通过。

交付：

- 修正 WESAD、StressID、Attention ablation 和 SHAP 叙述。
- 同步最新版系统图与实验图。
- 编译真实 SmartStress PDF，逐页检查标题、表格、算法、图片、引用、页眉页脚。
- 论文仓库逐 step 提交；本 Agent 仓库只同步经确认的副本/链接。

验收：PDF 第一页标题为 SmartStress，不再是模板；所有数字可回溯到实验 artifact。

复杂度：M–L。

## 7. 必补实验计划

### E0. Legacy reproduction gate

- 用迁移权重复现 S17 0.9578 F1。
- 对至少一个黄金窗口比较来源仓库和 Agent 的概率。
- 输出 model/feature/preprocess hash。

目的：证明迁移没有改变旧模型，而不是证明旧方法正确。

### E1. 预处理定义消融

比较：

- legacy baseline + legacy feature definitions；
- 修正 neutral-window baseline；
- 再修正 RMSSD/NN50/pNN50；
- 必要时比较频域与 TINN 定义。

报告特征分布、NaN/失败率、class separability 和最终分类性能。该实验决定论文的方法段落应如何重写。

### E2. 无测试泄漏的 WESAD subject-level evaluation

- 外层：15 个 subject LOSO。
- 内层：只从外层训练 subjects 中选择 validation subjects/epoch/阈值。
- 外层 test subject 只评估一次。
- DNN 与 Attention 使用完全相同的数据划分、训练预算和选择规则。
- 报告每 subject 指标、mean±std、bootstrap CI 和 pooled confusion；明确 macro 与 micro。
- 增加 subject/episode 级结果，避免把重叠窗口数量当作独立证据。

### E3. Attention ablation

- 同一 corrected feature set、同一 seeds、同一 nested splits。
- 至少 5 seeds 或给出可解释的方差估计。
- 报告 Attention 赢得多少 subject，而不只报告单一 S17。
- 若 Attention 平均提升很小，应调整论文贡献措辞。

### E4. 概率校准与触发策略

- Brier score、ECE、reliability diagram。
- 阈值仅在 inner validation 上选择。
- 比较单窗、连续 N 窗、hysteresis/cooldown 的 false alarm 与 missed episode。
- 输出最终 `support_activation_policy` 配置。

### E5. StressID 外部验证

- 使用与主模型一致的 Attention architecture 和 corrected preprocessing。
- subject-aware adaptation/eval 拆分保持完全不重叠。
- zero-shot、CORAL、final-layer adaptation 使用相同 seeds/budget。
- 报告 5 seeds 之外，增加每 subject error、域偏移特征和校准变化。
- 不能把标准 DNN 的 StressID 数字与 Attention 的 WESAD 数字当作同一模型链路。

### E6. Attention SHAP 重做

- 解释实际主模型，不再解释标准 DNN。
- background 数据与外层 test 隔离。
- 生成 global bar、beeswarm、stress/non-stress local examples。
- 定量检查贡献方向与标准 HRV 定义是否一致；避免只凭视觉主观解释。
- 记录 SHAP version、background ids、model hash 和样本 ids。

### E7. RAG 与回复质量

检索层：

- 构建有 relevance 标注的查询集。
- Recall@K、MRR/nDCG、无结果率、延迟。
- 比较 text-only query 与 text+physiology query。

生成层：

- groundedness/faithfulness、引用正确性、helpfulness、non-diagnostic compliance。
- 盲评 RAG on/off；报告评审一致性和置信区间。
- TF-IDF/BERTScore 只作为辅助指标，不能单独证明临床或安全质量。

### E8. 安全与闭环系统实验

- 危机、自伤、医疗诊断请求、普通压力、假阳性、低质量 ECG、模型不可用等场景集。
- 统计 inappropriate intervention、consent bypass、unsupported advice、refinement success。
- 测试模型/RAG/LLM/DB 失败时的降级行为。
- 测量预处理、推理、SHAP、RAG、LLM 各阶段延迟和总延迟。

### E9. Paper build verification

- 在固定 TeX 环境编译。
- 检查 citation keys、figure assets、表格数字和实验 artifact 的自动一致性。
- 渲染全部 PDF 页面并视觉核验。

## 8. 工程复杂度评估

| 范围 | 复杂度 | 估算 | 主要不确定性 |
| --- | --- | --- | --- |
| 只迁移旧 Attention 权重与 12 维特征输入 | M | 3–5 人日 | artifact 打包、环境、黄金测试 |
| 加入原始 ECG 和 legacy 预处理 | L | 5–8 人日 | HeartPy 稳定性、基线输入、质量门 |
| corrected preprocessing + 重训 | XL | 10–18 人日 + GPU | 特征定义改变、嵌套验证、结果可能下降 |
| SHAP 运行时集成 | M–L | 2–4 人日 | background、延迟、BatchNorm/Attention 兼容 |
| MindCare/RAG 生理上下文和证据追踪 | L | 4–7 人日 | LLM 输出 schema、评估集、TiDB 延迟 |
| 显式编排 + HITL + dry-run TaskRelief | L | 4–7 人日 | checkpoint/interrupt 语义、循环与幂等 |
| 真实日历/任务连接器 | XL | 另估 5–15+ 人日 | 目标平台、OAuth、权限、回滚 |
| 完整实验、统计、人工评估与论文更新 | XL | 10–25+ 人日 | GPU 排队、标注者、结果反复 |

推荐双轨路线总体为 XL。代码量本身不大，复杂度主要来自“修正方法后必须重建证据”，而不是 PyTorch 层数。

## 9. 最终验收清单

- [ ] 用户已确认 D1–D6。
- [ ] legacy 推理在黄金样本上与来源仓库一致。
- [ ] corrected feature definitions 有单元测试和方法说明。
- [ ] 运行模型与 SHAP 解释模型 hash 完全一致。
- [ ] PhysioSense 不再包含 HR heuristic 或伪造 0.1。
- [ ] State/API 输出 features、quality、drivers、model metadata。
- [ ] RAG query 明确包含生理和语义证据。
- [ ] 检索与回复保留可审计来源。
- [ ] HITL 无子串误判、拒绝可 refine、执行幂等。
- [ ] WESAD 无测试集选 epoch/阈值。
- [ ] StressID 使用与论文主模型一致的 architecture/preprocessing。
- [ ] Attention SHAP 已重做并记录 provenance。
- [ ] RAG 有检索层、生成层和安全层评价。
- [ ] README、API 示例、实验说明与当前行为一致。
- [ ] 真实 SmartStress PDF 已编译并逐页通过视觉检查。
- [ ] 每个小 step 都有独立 Git commit 和 `.codex/WORKLOG.md` 记录。

## 10. 当前停止点

在用户回答 D1–D6 前，不修改 PhysioSense、State/API、Graph、MindCare、TaskRelief、模型权重或论文仓库。下一步从 Step 1 开始，并严格按小 step 提交。
