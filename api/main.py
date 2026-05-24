"""FastAPI 后端服务 - AI Agent Workflow Platform"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.agents import WorkflowEngine
from models.manager import ModelManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_manager = ModelManager()
    yield
    await app.state.model_manager.close()


app = FastAPI(
    title="AI Agent Workflow Platform",
    description="多模型多 Agent 协同工作流平台",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Request / Response Models ---

class WorkflowRequest(BaseModel):
    task: str
    model: str = "mimo-v2.5-pro"
    max_retries: int = 2


class ChatRequest(BaseModel):
    model: str = "mimo-v2.5-pro"
    message: str
    temperature: float = 0.3


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path("static/index.html")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/models")
async def list_models():
    manager: ModelManager = app.state.model_manager
    return {"models": manager.list_models()}


@app.post("/api/workflow")
async def run_workflow(req: WorkflowRequest):
    engine = WorkflowEngine(model_name=req.model)
    result = await engine.execute(task=req.task, max_retries=req.max_retries)
    return {
        "goal": result.goal,
        "status": result.status,
        "summary": result.summary,
        "total_tokens": result.total_tokens,
        "tasks": [
            {
                "task_id": t.task_id,
                "status": t.status,
                "result": t.result,
                "artifacts": t.artifacts,
                "review": t.review,
            }
            for t in result.tasks
        ],
    }


@app.post("/api/chat")
async def direct_chat(req: ChatRequest):
    manager: ModelManager = app.state.model_manager
    try:
        response = await manager.chat(
            model=req.model,
            messages=[{"role": "user", "content": req.message}],
            temperature=req.temperature,
        )
        return {"model": req.model, "response": response}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
