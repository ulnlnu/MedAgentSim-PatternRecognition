# 面向临床场景的多轮交互式诊断模拟系统

> 基于开源项目 [MedAgentSim](https://github.com/MAXNORM8650/MedAgentSim) 的课程实践项目

## 项目简介

当前大语言模型在医疗诊断场景中，通常依赖"病情信息一次性给出"的静态问答设定，要求模型直接输出答案。这种设定虽然便于评估，却难以反映真实临床诊断过程的复杂性。

本项目构建了一个开放式的临床多智能体模拟环境，包含**医生智能体**、**患者智能体**和**检查智能体**，通过多轮问诊和主动申请检查来推进诊断过程，模拟真实临床中的动态诊断流程。

### 核心关注点

1. **多轮问诊**：医生智能体通过逐步追问而非"一步到位"地输出答案
2. **检查调用机制**：模型只有在提出合理的检查请求时才能获得对应结果
3. **认知偏差注入**：支持 13 种认知偏差类型（近因偏差、频率偏差、确认偏差、性别/种族/文化偏差等）
4. **多医生辩论**：引入多智能体协作与投票机制，提高诊断质量与稳定性
5. **经验增强**：通过历史病例检索与自我反思优化诊断策略

## 系统架构

```
┌─────────────┐     提问      ┌─────────────┐
│  医生智能体  │ ──────────→ │  患者智能体  │
│  DoctorAgent│ ←────────── │ PatientAgent│
└──────┬──────┘    回答/症状  └─────────────┘
       │
       │ 请求检查 (REQUEST TEST)
       ▼
┌─────────────┐
│  检查智能体  │
│Measurement  │  ← 返回检查结果
│   Agent     │
└─────────────┘
       │
       ▼
  诊断结果 (DIAGNOSIS READY)
```

**交互流程**：
1. 医生智能体根据当前对话历史向患者提问
2. 患者智能体根据场景设定回答症状与病史
3. 医生可请求医学检查（血液检查、影像检查等），由检查智能体返回结果
4. 多轮交互后，医生给出最终诊断

## 项目结构

```
MedAgentSim/
├── medsim/                    # 核心源码
│   ├── main.py                # 命令行模式入口（argparse）
│   ├── run.py                 # YAML 配置模式入口
│   ├── agents.py              # 多医生辩论 Agent
│   ├── query_model.py         # LLM 后端适配层
│   ├── utils.py               # 工具函数
│   ├── core/
│   │   ├── agent.py           # 三大智能体实现（Doctor/Patient/Measurement）
│   │   └── scenario.py        # 数据集加载器（MedQA/NEJM/MIMIC-IV）
│   ├── configs/               # YAML 运行配置文件
│   ├── server/                # Django 可视化服务端
│   └── simulate/              # Phaser 可视化前端启动器
├── datasets/                  # 医疗数据集
│   └── _medqa.jsonl           # MedQA 场景数据（107 例）
├── requirements.txt           # Python 依赖
├── environment.yml            # Conda 环境配置
├── setup.py                   # 包安装配置
└── test_run.py                # 快速测试脚本
```

## 环境搭建

### 1. 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate mgent
pip install -r requirements.txt
```

### 2. 配置 LLM 后端

本项目支持多种 LLM 后端，按优先级自动选择：

| 后端 | 说明 | 配置方式 |
|------|------|----------|
| **Ollama** (推荐) | 本地部署，无需 API Key | 安装 Ollama，拉取模型后直接使用 |
| **vLLM** | GPU 加速推理 | 启动 vLLM 服务端 |
| **OpenAI API** | 云端调用 | 设置 `--openai_api_key` |
| **Anthropic** | Claude 模型 | 设置 `--anthropic_api_key` |

#### Ollama 本地部署（推荐）

```bash
# 安装 Ollama 后拉取模型
ollama pull qwen2.5:7b

# 运行模拟
python medsim/main.py \
    --doctor_llm qwen2.5:7b \
    --patient_llm qwen2.5:7b \
    --measurement_llm qwen2.5:7b \
    --moderator_llm qwen2.5:7b \
    --agent_dataset MedQA \
    --num_scenarios 5 \
    --total_inferences 20
```

#### OpenAI API

```bash
python medsim/main.py \
    --openai_api_key sk-xxx \
    --doctor_llm gpt-4o \
    --patient_llm gpt-4o \
    --measurement_llm gpt-4o \
    --moderator_llm gpt-4o \
    --agent_dataset MedQA \
    --num_scenarios 10 \
    --total_inferences 20
```

### 3. 命令行参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--inf_type` | 交互模式：`llm` / `human_doctor` / `human_patient` | `llm` |
| `--doctor_bias` | 医生认知偏差类型 | `None` |
| `--patient_bias` | 患者认知偏差类型 | `None` |
| `--doctor_llm` | 医生使用的模型 | `llama3b` |
| `--patient_llm` | 患者使用的模型 | `llama3b` |
| `--measurement_llm` | 检查智能体使用的模型 | `llama3b` |
| `--moderator_llm` | 评判诊断正确性的模型 | `llama3b` |
| `--agent_dataset` | 数据集：`MedQA` / `NEJM` / `MIMICIV` | `MedQA` |
| `--num_scenarios` | 模拟场景数 | 全部 |
| `--total_inferences` | 每场景最大交互轮数 | `20` |

### 支持的认知偏差类型

**医生偏差**：`recency`、`frequency`、`false_consensus`、`confirmation`、`status_quo`、`gender`、`race`、`sexual_orientation`、`cultural`、`education`、`religion`、`socioeconomic`

**患者偏差**：`recency`、`frequency`、`false_consensus`、`self_diagnosis`、`gender`、`race`、`sexual_orientation`、`cultural`、`education`、`religion`、`socioeconomic`

## 运行示例

### 基础诊断模拟

```bash
# 无偏差基线实验
python medsim/main.py \
    --inf_type llm \
    --doctor_bias None \
    --patient_bias None \
    --doctor_llm qwen2.5:7b \
    --patient_llm qwen2.5:7b \
    --measurement_llm qwen2.5:7b \
    --moderator_llm qwen2.5:7b \
    --agent_dataset MedQA \
    --num_scenarios 3 \
    --total_inferences 20
```

### 带认知偏差的实验

```bash
# 注入医生确认偏差
python medsim/main.py \
    --doctor_bias confirmation \
    --patient_bias None \
    --doctor_llm qwen2.5:7b \
    --patient_llm qwen2.5:7b \
    --measurement_llm qwen2.5:7b \
    --moderator_llm qwen2.5:7b \
    --agent_dataset MedQA
```

### 人类参与模式

```bash
# 人类扮演医生
python medsim/main.py --inf_type human_doctor --patient_llm qwen2.5:7b ...

# 人类扮演患者
python medsim/main.py --inf_type human_patient --doctor_llm qwen2.5:7b ...
```

### 使用 YAML 配置运行

```bash
python medsim/run.py --config medsim/configs/config.yaml
```

## 输出说明

模拟结果保存在 `output/` 目录下：

```
output/
└── scenario_0/
    └── dialogue_history.json   # 完整对话记录
```

对话记录格式：

```json
[
  {"speaker": "Doctor", "text": "你好，请问你哪里不舒服？"},
  {"speaker": "Patient", "text": "我最近感觉四肢无力..."},
  {"speaker": "Doctor", "text": "REQUEST TEST: 血液检查"},
  {"speaker": "Measurement", "text": "乙酰胆碱受体抗体阳性..."},
  {"DIAGNOSIS_READY_Answer": "重症肌无力 (Myasthenia gravis)",
   "DIAGNOSIS_READY_Simulation": "Scene 0, The diagnosis was CORRECT, 100%"}
]
```

## 支持的数据集

| 数据集 | 说明 | 样本数 |
|--------|------|--------|
| MedQA | 模拟诊断场景 | 107 |
| MedQA Extended | 扩展诊断场景 | 214 |
| NEJM | 真实复杂病例 | 15 |
| NEJM Extended | 扩展真实病例 | 120 |
| MIMIC-IV | 真实临床病例 | 288 |

## 致谢

- 原始项目：[MedAgentSim](https://github.com/MAXNORM8650/MedAgentSim) by Mohammad Almansoori, Komal Kumar, Hisham Cholakkal
- 论文：*Self-Evolving Multi-Agent Simulations for Realistic Clinical Interactions* (MICCAI 2025)
- 参考项目：[AgentClinic](https://github.com/samuelschmidgall/agentclinic)、[Microsoft PromptBase](https://github.com/microsoft/promptbase)

## 许可证

本项目基于 CC BY-NC-SA 4.0 许可证发布。
