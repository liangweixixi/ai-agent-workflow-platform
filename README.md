# AI Agent Workflow Platform

多模型协同 AI Agent 工作流平台，支持 Claude、MiMo、DeepSeek、GPT 等多种大模型的统一调度与协同工作。

## 核心特性

- **多 Agent 协同**：Planner（任务规划）、Executor（任务执行）、Reviewer（质量审查）三角色协作
- **多模型支持**：统一接口适配 Claude、Xiaomi MiMo、DeepSeek、GPT 等主流模型
- **工作流编排**：可视化定义任务流水线，支持串行、并行、条件分支
- **实时监控**：任务状态追踪、Token 消耗统计、执行日志流
- **插件化架构**：工具和模型均可扩展，开箱即用

## 架构

```
┌─────────────────────────────────────────────────┐
│                  API Layer (FastAPI)              │
├─────────────────────────────────────────────────┤
│              Workflow Engine                      │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐      │
│  │  Planner  │ │  Executor │ │  Reviewer │      │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘      │
│        └─────────────┼─────────────┘             │
├──────────────────────┼──────────────────────────┤
│              Model Adapter Layer                  │
│  ┌────────┐ ┌────────┐ ┌─────────┐ ┌──────┐    │
│  │ Claude │ │  MiMo  │ │DeepSeek │ │ GPT  │    │
│  └────────┘ └────────┘ └─────────┘ └──────┘    │
└─────────────────────────────────────────────────┘
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Key

# 启动服务
python -m api.main
```

访问 http://localhost:8000/docs 查看 API 文档。

## 使用示例

```python
from core.workflow import WorkflowEngine
from models.manager import ModelManager

engine = WorkflowEngine()
result = await engine.execute(
    task="分析 GitHub 仓库 liangweixixi/ai-agent-workflow-platform 的代码质量",
    model="mimo-v2.5-pro"
)
print(result)
```

## 技术栈

- Python 3.11+
- FastAPI + Uvicorn
- Pydantic 数据校验
- AsyncIO 异步并发
- 支持 OpenAI 兼容 API 协议

## License

MIT
