# SmartStress Agent

SmartStress 是与当前论文叙述对齐的闭环压力支持原型：PhysioSense 从 ECG 或 12 维生理特征估计压力并生成 SHAP 解释；MindCare 将压力概率、主要生理驱动和用户表达写入 RAG 查询；Meta-Reflective Orchestrator 决定支持、澄清、提案、确认或结束；TaskRelief 只生成和模拟 dry-run 调整。

当前实现固定使用 `wesad_attention_v1`，即论文 Attention-DNN 的 S17 checkpoint。论文中的 95.78% F1 指 WESAD S17 单一留出结果，不是重新训练结果，也不是跨全部受试者的平均值。本仓库不会重训模型。

## 闭环

```text
12-D normalized features ─┐
                          ├─> PhysioSense ─> probability + SHAP drivers
raw ECG + neutral baseline┘                         │
                                                   v
user utterance ──────────────────────────────> MindCare + RAG
                                                   │
                                                   v
                                      Meta-Reflective Orchestrator
                                                   │
                                stressor ─> TaskRelief proposal
                                                   │
                                      explicit yes / no / cancel
                                                   │
                                    allowlisted dry-run simulation only
```

TaskRelief 没有真实日历、任务管理器或消息服务连接。即使用户回答 `yes`，结果仍为 `execution_mode="dry_run"` 和 `external_side_effects=false`。

## 安装与配置

需要 Python 3.10+。

```bash
pip install -r requirements.txt
```

项目根目录可创建 `.env`：

```env
GOOGLE_API_KEY=your_google_api_key
SMARTSTRESS_STRESS_THRESHOLD=0.5

# 仅在使用 TiDB RAG 时需要
DB_HOST=your_tidb_host
DB_PORT=4000
DB_USERNAME=your_tidb_user
DB_PASSWORD=your_tidb_password
DB_DATABASE=your_tidb_database
```

模型 checkpoint、SHA-256、特征顺序和 S17 指标记录在 `smartstress_langgraph/physio/artifacts/wesad_attention_v1.json`。加载时会验证 checkpoint 哈希。

## 输入契约

每个 `SensorData` 必须且只能选择一种输入方式。

### 方式 A：12 维归一化特征

这是模型的直接输入。顺序固定为：

1. `mean_hr`
2. `std_hr`
3. `tinn`
4. `hrv_index`
5. `nn50`
6. `pnn50`
7. `mean_hrv`
8. `std_hrv`
9. `rmssd`
10. `fft_mean`
11. `fft_std`
12. `sum_psd`

```python
from datetime import datetime, timezone

from smartstress_langgraph.io_models import SensorData

sensor = SensorData(
    timestamp=datetime.now(timezone.utc).isoformat(),
    normalized_features=[
        1.4521011, 0.1006440, 0.0561910, 0.0005347,
        0.0000027, 0.0986588, 0.6364315, 0.0987973,
        0.6163059, 1.4509790, 1.5290556, 0.0092296,
    ],
)
```

### 方式 B：原始 ECG

原始 ECG 至少为 20 秒。WESAD 原始采样率为 700 Hz；其他正采样率会先重采样到 700 Hz。归一化必须提供同一用户的中性基线，形式为中性 ECG 或已计算的 12 维未归一化基线特征。

服务端按“一个当前窗口/一次请求”处理数据。若按论文的一秒 stride 连续监测，采集端应每秒提交向前滑动后的当前 ECG 窗口；没有新窗口时 PhysioSense 不写入伪造概率。

```python
sensor = SensorData(
    timestamp=datetime.now(timezone.utc).isoformat(),
    raw_ecg=current_window_samples,
    sample_rate_hz=700,
    baseline_ecg=neutral_baseline_samples,
    baseline_sample_rate_hz=700,
)
```

由于不重训模型，ECG 路径严格复现原 checkpoint 的特征公式。不能在保留该权重时把 NN50、RMSSD 等公式替换为另一套实现；那会改变模型输入分布。

## SDK 示例

```python
from smartstress_langgraph.api import start_monitoring_session
from smartstress_langgraph.examples.sample_data import DEMO_STRESS_FEATURES
from smartstress_langgraph.io_models import SensorData, StartSessionRequest, UserInfo

handle, view = start_monitoring_session(
    StartSessionRequest(
        user=UserInfo(user_id="user-1", session_id="session-1"),
        initial_sensor_data=SensorData(
            timestamp="2026-07-13T18:00:00Z",
            normalized_features=DEMO_STRESS_FEATURES,
        ),
    )
)

print(view.current_stress_prob)
print(view.physio_top_drivers)
print(view.orchestration_decision)
```

完整示例：

```bash
python -m smartstress_langgraph.examples.demo_session
python run_api_key_test.py
python server.py
```

FastAPI 入口：

- `GET /health`
- `POST /api/start_session`
- `POST /api/continue_session`
- `GET /docs`

## 输出状态

前端/API 可读取：

- `current_stress_prob`、`stress_detected`、`stress_threshold`
- `physio_model_id`、`physio_input_source`、`physio_feature_map`
- `physio_attributions`、`physio_top_drivers`
- `rag_context`、`current_stressor`
- `orchestration_decision`、`orchestration_reason`、`orchestration_signals`
- `safety_escalation`（危机语言优先阻断 TaskRelief）
- `suggested_action`、`tool_execution_mode`、`external_side_effects`
- `audit_trail`、`error_log`

## 测试

全部测试使用标准库 `unittest`：

```bash
python -m unittest discover -s tests -v
```

关键覆盖包括：

- S17 金标准概率和 checkpoint 哈希
- 源预处理算法金标准特征
- 12 维/原始 ECG 输入校验与中性基线归一化
- SHAP 归因和失败隔离
- MindCare 生理增强 RAG 查询
- `yes` / `no` / `cancel`、refinement 和 Meta-Reflective 路由
- TaskRelief dry-run allowlist
- 不访问 Gemini、TiDB 或外部任务服务的闭环集成流程

## 边界

该项目是非临床研究原型，不提供诊断、药物或危机干预建议。当前检测结果只对应已记录的 WESAD S17 单一留出评估；模型外推、跨数据集适配和真实环境部署不在本次代码修改范围内。
