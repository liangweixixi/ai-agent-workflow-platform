"""多 Agent 协同引擎 - Planner / Executor / Reviewer"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from models.manager import ModelManager


class AgentRole(str, Enum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"


AGENT_PROMPTS = {
    AgentRole.PLANNER: """你是一个任务规划专家 (Planner)。你的职责是：
1. 分析用户的复杂需求
2. 将其分解为可执行的子任务
3. 确定子任务之间的依赖关系
4. 输出结构化的任务计划

输出格式必须是 JSON：
{
    "goal": "总体目标描述",
    "tasks": [
        {
            "id": "task_1",
            "description": "任务描述",
            "depends_on": [],
            "priority": "high|medium|low"
        }
    ],
    "estimated_complexity": "low|medium|high"
}""",

    AgentRole.EXECUTOR: """你是一个任务执行专家 (Executor)。你的职责是：
1. 接收具体的任务描述
2. 分析任务需求并制定执行方案
3. 输出详细的执行结果
4. 如果任务无法完成，明确说明原因

输出格式：
{
    "task_id": "任务ID",
    "status": "success|failed|partial",
    "result": "执行结果的详细描述",
    "artifacts": ["生成的文件或代码路径"],
    "token_used": 0
}""",

    AgentRole.REVIEWER: """你是一个质量审查专家 (Reviewer)。你的职责是：
1. 审查 Executor 的执行结果
2. 评估结果质量、完整性和正确性
3. 提出改进建议
4. 决定是否通过或需要返工

输出格式：
{
    "task_id": "任务ID",
    "verdict": "approve|reject|revise",
    "score": 0-100,
    "feedback": "详细的审查意见",
    "suggestions": ["改进建议1", "改进建议2"]
}"""
}


@dataclass
class TaskResult:
    task_id: str
    status: str
    result: str
    artifacts: list[str] = field(default_factory=list)
    token_used: int = 0
    review: Optional[dict] = None


@dataclass
class WorkflowResult:
    goal: str
    status: str
    tasks: list[TaskResult] = field(default_factory=list)
    total_tokens: int = 0
    summary: str = ""


class Agent:
    def __init__(self, role: AgentRole, model_name: str = "mimo-v2.5-pro"):
        self.role = role
        self.model_name = model_name
        self.model_manager = ModelManager()
        self.system_prompt = AGENT_PROMPTS[role]

    async def execute(self, user_message: str, context: str = "") -> dict:
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        if context:
            messages.append({"role": "system", "content": f"上下文信息：\n{context}"})
        messages.append({"role": "user", "content": user_message})

        response = await self.model_manager.chat(
            model=self.model_name,
            messages=messages,
            temperature=0.3 if self.role == AgentRole.EXECUTOR else 0.1,
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"raw_response": response, "parsed": False}


class WorkflowEngine:
    def __init__(self, model_name: str = "mimo-v2.5-pro"):
        self.model_name = model_name
        self.planner = Agent(AgentRole.PLANNER, model_name)
        self.executor = Agent(AgentRole.EXECUTOR, model_name)
        self.reviewer = Agent(AgentRole.REVIEWER, model_name)

    async def execute(self, task: str, max_retries: int = 2) -> WorkflowResult:
        result = WorkflowResult(goal=task, status="running")

        # Step 1: Planner 分解任务
        plan = await self.planner.execute(task)
        tasks = plan.get("tasks", [])
        if not tasks:
            result.status = "failed"
            result.summary = "任务分解失败"
            return result

        # Step 2: Executor 逐个执行
        context = f"总体目标：{plan.get('goal', task)}"
        for task_def in tasks:
            task_result = await self._execute_task(task_def, context, max_retries)
            result.tasks.append(task_result)
            result.total_tokens += task_result.token_used
            context += f"\n\n任务 {task_def['id']} 结果：{task_result.result}"

        # Step 3: 汇总
        completed = sum(1 for t in result.tasks if t.status == "success")
        result.status = "completed" if completed == len(result.tasks) else "partial"
        result.summary = f"共 {len(result.tasks)} 个任务，{completed} 个成功完成"

        return result

    async def _execute_task(self, task_def: dict, context: str, max_retries: int) -> TaskResult:
        task_id = task_def["id"]
        task_desc = task_def["description"]

        for attempt in range(max_retries + 1):
            # Executor 执行
            exec_result = await self.executor.execute(
                f"请执行以下任务：\n{task_desc}",
                context=context
            )

            # Reviewer 审查
            review = await self.reviewer.execute(
                f"请审查任务 {task_id} 的执行结果：\n{json.dumps(exec_result, ensure_ascii=False)}",
                context=context
            )

            if review.get("verdict") == "approve" or attempt == max_retries:
                return TaskResult(
                    task_id=task_id,
                    status=exec_result.get("status", "success"),
                    result=exec_result.get("result", str(exec_result)),
                    artifacts=exec_result.get("artifacts", []),
                    token_used=exec_result.get("token_used", 0),
                    review=review,
                )

        return TaskResult(task_id=task_id, status="failed", result="超过最大重试次数")
