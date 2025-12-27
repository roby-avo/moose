from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Literal

import httpx
import ollama
from fastapi import Depends, FastAPI, Header, HTTPException, Security
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from moose.config import Settings, get_settings
from moose.llm import create_client
from moose_api.queue import (
    JobRecord,
    WorkerPool,
    build_backends,
    utc_now,
)


class LLMOverrides(BaseModel):
    provider: Literal["ollama", "openrouter"] | None = None
    model: str | None = None
    ollama_token: str | None = None
    openrouter_api_key: str | None = None


class NERTaskIn(BaseModel):
    task_id: str
    text: str


class NERRequest(BaseModel):
    schema: Literal["coarse", "fine"]
    tasks: list[NERTaskIn] = Field(min_length=1)
    include_scores: bool = False
    llm: LLMOverrides | None = None


class TabularTaskIn(BaseModel):
    task_id: str
    table_id: str
    sampled_rows: list[dict[str, Any]] = Field(min_length=1)


class TabularRequest(BaseModel):
    schema: Literal["coarse", "fine"]
    tasks: list[TabularTaskIn] = Field(min_length=1)
    include_scores: bool = False
    llm: LLMOverrides | None = None


STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Moose API", version="0.1.0", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


async def require_api_key(
    api_key: str | None = Security(api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> None:
    expected = app.state.settings.MOOSE_API_KEY
    if not expected:
        raise HTTPException(status_code=500, detail="API key not configured")
    token = api_key or (bearer.credentials if bearer else None)
    if token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.on_event("startup")
async def startup() -> None:
    settings = get_settings()
    if not settings.MOOSE_API_KEY:
        raise RuntimeError("MOOSE_API_KEY is required for the API service")
    job_store, queue_backend, info = await build_backends(settings)
    llm_client = None
    if settings.MOOSE_LLM_PROVIDER.lower() != "openrouter" or settings.MOOSE_OPENROUTER_API_KEY:
        llm_client = create_client(settings)

    app.state.settings = settings
    app.state.job_store = job_store
    app.state.queue_backend = queue_backend
    app.state.queue_info = info
    app.state.llm_client = llm_client

    worker_pool = WorkerPool(queue_backend, job_store, llm_client, settings)
    await worker_pool.start()
    app.state.worker_pool = worker_pool


@app.on_event("shutdown")
async def shutdown() -> None:
    worker_pool: WorkerPool = app.state.worker_pool
    await worker_pool.stop()
    llm_client = app.state.llm_client
    await llm_client.close()
    queue_info = app.state.queue_info
    mongo_client = queue_info.get("mongo_client")
    if mongo_client is not None:
        mongo_client.close()


async def _enqueue_job(endpoint_type: str, payload: dict):
    settings: Settings = app.state.settings
    queue_backend = app.state.queue_backend
    job_store = app.state.job_store

    queue_size = await queue_backend.size()
    if queue_size >= settings.MOOSE_QUEUE_MAXSIZE:
        raise HTTPException(status_code=429, detail="Queue is full, try again later")

    job_id = str(uuid.uuid4())
    now = utc_now()
    job = JobRecord(
        job_id=job_id,
        endpoint_type=endpoint_type,
        payload=payload,
        status="queued",
        created_at=now,
        updated_at=now,
        retries=0,
    )
    await job_store.put_job(job)
    try:
        await queue_backend.enqueue(job_id)
    except Exception as exc:  # noqa: BLE001
        await job_store.update_job(
            job_id,
            status="failed",
            updated_at=utc_now(),
            error=f"Failed to enqueue job: {exc}",
        )
        raise HTTPException(status_code=429, detail="Queue is full, try again later")
    return job_id


def _build_llm_payload(
    request_llm: LLMOverrides | None,
    openrouter_api_key: str | None,
    ollama_token: str | None,
) -> dict[str, Any] | None:
    llm_payload = request_llm.model_dump(exclude_none=True) if request_llm else {}
    if openrouter_api_key:
        llm_payload["openrouter_api_key"] = openrouter_api_key
    if ollama_token:
        llm_payload["ollama_token"] = ollama_token
    return llm_payload or None


@app.post("/ner", dependencies=[Depends(require_api_key)])
async def submit_ner(
    request: NERRequest,
    openrouter_api_key: str | None = Header(default=None, alias="X-OpenRouter-API-Key"),
    ollama_token: str | None = Header(default=None, alias="X-Ollama-Token"),
):
    payload = {
        "schema": request.schema,
        "tasks": [task.model_dump() for task in request.tasks],
        "include_scores": request.include_scores,
        "llm": _build_llm_payload(request.llm, openrouter_api_key, ollama_token),
    }
    job_id = await _enqueue_job("ner", payload)
    return {"job_id": job_id, "status": "queued"}


@app.post("/tabular/annotate", dependencies=[Depends(require_api_key)])
async def submit_tabular(
    request: TabularRequest,
    openrouter_api_key: str | None = Header(default=None, alias="X-OpenRouter-API-Key"),
    ollama_token: str | None = Header(default=None, alias="X-Ollama-Token"),
):
    payload = {
        "schema": request.schema,
        "tasks": [task.model_dump() for task in request.tasks],
        "include_scores": request.include_scores,
        "llm": _build_llm_payload(request.llm, openrouter_api_key, ollama_token),
    }
    job_id = await _enqueue_job("tabular", payload)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
async def get_job(job_id: str):
    job_store = app.state.job_store
    job = await job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    response = {
        "job_id": job.job_id,
        "status": job.status,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }
    if job.status == "completed":
        response["result"] = job.result
    if job.status == "failed":
        response["error"] = job.error
    return response


@app.get("/health", dependencies=[Depends(require_api_key)])
async def health():
    settings: Settings = app.state.settings
    queue_info = app.state.queue_info
    return {
        "status": "ok",
        "provider": settings.MOOSE_LLM_PROVIDER,
        "model": settings.MOOSE_MODEL,
        "worker_count": settings.MOOSE_WORKER_COUNT,
        "queue_backend": queue_info.get("queue_backend"),
    }


def _parse_price(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@app.get("/models", dependencies=[Depends(require_api_key)])
async def list_models(
    provider: Literal["ollama", "openrouter", "all"] = "all",
    ollama_token: str | None = Header(default=None, alias="X-Ollama-Token"),
    openrouter_api_key: str | None = Header(default=None, alias="X-OpenRouter-API-Key"),
):
    settings: Settings = app.state.settings
    results: dict[str, Any] = {}

    if provider in {"ollama", "all"}:
        headers = {}
        token = ollama_token or settings.MOOSE_OLLAMA_TOKEN
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            async with httpx.AsyncClient(
                base_url=settings.MOOSE_OLLAMA_HOST,
                timeout=settings.MOOSE_TIMEOUT_SECS,
            ) as client:
                resp = await client.get("/api/tags", headers=headers)
                resp.raise_for_status()
                data = resp.json()
            models = [item.get("name") for item in data.get("models", []) if item.get("name")]
            results["ollama"] = {"models": models}
        except Exception as exc:  # noqa: BLE001
            results["ollama"] = {"error": str(exc)}

    if provider in {"openrouter", "all"}:
        api_key = openrouter_api_key or settings.MOOSE_OPENROUTER_API_KEY
        if not api_key:
            results["openrouter"] = {"error": "OpenRouter API key not configured"}
        else:
            try:
                headers = {"Authorization": f"Bearer {api_key}"}
                async with httpx.AsyncClient(
                    base_url=settings.MOOSE_OPENROUTER_BASE_URL,
                    timeout=settings.MOOSE_TIMEOUT_SECS,
                ) as client:
                    resp = await client.get("/models", headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                free_models = []
                for item in data.get("data", []):
                    pricing = item.get("pricing", {})
                    prompt_price = _parse_price(pricing.get("prompt"))
                    completion_price = _parse_price(pricing.get("completion"))
                    if prompt_price == 0 and completion_price == 0:
                        free_models.append(
                            {
                                "id": item.get("id"),
                                "name": item.get("name"),
                                "context_length": item.get("context_length"),
                            }
                        )
                results["openrouter"] = {"models": free_models}
            except Exception as exc:  # noqa: BLE001
                results["openrouter"] = {"error": str(exc)}

    return results


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="Moose API Docs",
        swagger_favicon_url="/static/moose-logo.png",
        swagger_css_url="/static/docs.css",
    )
