"""
app.py — FastAPI server for MLOps Incident Response Environment.

OpenEnv HTTP protocol:
  POST /reset   → initial observation
  POST /step    → action → observation + reward + done
  GET  /state   → episode metadata (root cause hidden)
  GET  /health  → liveness check
  WS   /ws      → WebSocket (required by openenv Python client)
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from environment import MLOpsEnvironment
# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MLOps Incident Response Environment",
    description=(
        "OpenEnv-compatible RL environment where AI agents act as on-call ML engineers. "
        "Diagnose and resolve production ML incidents across 3 difficulty levels."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singleton env (one per worker) ────────────────────────────────────────────
env = MLOpsEnvironment()


# ── Request schemas ───────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_id: str = Field(
        default="easy",
        description="Task difficulty: 'easy' | 'medium' | 'hard'",
    )


class StepRequest(BaseModel):
    action_type: str = Field(
        default="inspect",
        description=(
            "inspect | query_logs | check_metrics | compare_configs | "
            "check_feature_drift | submit_diagnosis | request_rollback"
        ),
    )
    target: str = Field(default="", description="Component name or root cause string")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra params e.g. {'root_cause': '...', 'fix': '...'}",
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
def health() -> Dict[str, str]:
    return {"status": "ok", "env": "mlops-incident-env", "version": "1.0.0"}


@app.post("/reset", tags=["openenv"])
def reset(request: Optional[ResetRequest] = None) -> Dict[str, Any]:
    """Reset environment. validate-submission.sh pings this — must return 200."""
    task_id = request.task_id if request else "easy"
    if task_id not in ("easy", "medium", "hard"):
        task_id = "easy"
    obs = env.reset(task_id=task_id)
    return obs.to_dict()


@app.post("/step", tags=["openenv"])
def step(request: StepRequest) -> Dict[str, Any]:
    """Execute one action, return observation + reward + done."""
    obs = env.step(
        action_type=request.action_type,
        target=request.target,
        parameters=request.parameters,
    )
    return obs.to_dict()


@app.get("/state", tags=["openenv"])
def state() -> Dict[str, Any]:
    """Episode metadata. true_root_cause excluded intentionally."""
    return env.state.to_dict()


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Required by openenv Python client (HTTPEnvClient).

    Client sends JSON:
      {"command": "reset", "task_id": "easy"}
      {"command": "step", "action_type": "inspect", "target": "api_gateway", "parameters": {}}
      {"command": "state"}
    """
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            command = msg.get("command", "")

            if command == "reset":
                obs = env.reset(task_id=msg.get("task_id", "easy"))
                await websocket.send_json(obs.to_dict())

            elif command == "step":
                obs = env.step(
                    action_type=msg.get("action_type", "inspect"),
                    target=msg.get("target", ""),
                    parameters=msg.get("parameters", {}),
                )
                await websocket.send_json(obs.to_dict())

            elif command == "state":
                await websocket.send_json(env.state.to_dict())

            else:
                await websocket.send_json(
                    {"error": f"Unknown command '{command}'. Use: reset, step, state"}
                )

    except WebSocketDisconnect:
        pass

# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    import os, uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=1, log_level="info")

if __name__ == "__main__":
    main()
