from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Security
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from moose.config import Settings, get_settings
from moose.llm import create_client
from moose.schema import get_schema_config
from moose_api.queue import (
    JobRecord,
    WorkerPool,
    build_backends,
    utc_now,
)


class LLMOverrides(BaseModel):
    provider: Literal["openrouter", "ollama"]
    model: str


class NERTaskIn(BaseModel):
    task_id: str
    text: str


class BaseNERRequest(BaseModel):
    tasks: list[NERTaskIn] = Field(min_length=1)
    include_scores: bool = False
    llm: LLMOverrides


class NERRequest(BaseNERRequest):
    schema: str = Field(
        description="Schema/vocabulary name to annotate against.",
        examples=["coarse", "fine", "dpv"],
    )

    @field_validator("schema")
    @classmethod
    def validate_schema(cls, value: str) -> str:
        try:
            get_schema_config(value)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return value


class TabularTaskIn(BaseModel):
    task_id: str
    table_id: str
    sampled_rows: list[dict[str, Any]] = Field(min_length=1)


class BaseTabularRequest(BaseModel):
    tasks: list[TabularTaskIn] = Field(min_length=1)
    include_scores: bool = False
    llm: LLMOverrides


class TabularRequest(BaseTabularRequest):
    schema: str = Field(
        description="Schema/vocabulary name to annotate against.",
        examples=["coarse", "fine", "dpv"],
    )

    @field_validator("schema")
    @classmethod
    def validate_schema(cls, value: str) -> str:
        try:
            get_schema_config(value)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return value


class SchemaNERRequest(BaseNERRequest):
    pass


class SchemaTabularRequest(BaseTabularRequest):
    pass


class DpvNERRequest(BaseNERRequest):
    pass


class DpvTabularRequest(BaseTabularRequest):
    pass


class JobQueuedResponse(BaseModel):
    job_id: str
    status: Literal["queued"]


STATIC_DIR = Path(__file__).resolve().parent / "static"

TAG_METADATA = [
    {"name": "NER", "description": "Named entity recognition endpoints."},
    {"name": "Tabular", "description": "Tabular semantic typing endpoints."},
    {"name": "Schemas", "description": "Schema-specific annotation endpoints."},
    {"name": "DPV", "description": "DPV classification endpoints."},
]

app = FastAPI(
    title="Moose API",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_tags=TAG_METADATA,
)
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


def _require_llm_overrides(
    request_llm: LLMOverrides,
    llm_api_key: str | None,
) -> None:
    provider = request_llm.provider.lower()
    if provider == "openrouter" and not llm_api_key:
        raise HTTPException(
            status_code=400,
            detail="LLM API key is required via X-LLM-API-Key for OpenRouter.",
        )


def _ensure_schema_supported(
    schema: str,
    require_text: bool = False,
    require_table: bool = False,
) -> None:
    try:
        config = get_schema_config(schema)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if require_text and not config.supports_text:
        raise HTTPException(
            status_code=400,
            detail=f"Schema '{schema}' does not support text annotation.",
        )
    if require_table and not config.supports_table:
        raise HTTPException(
            status_code=400,
            detail=f"Schema '{schema}' does not support tabular annotation.",
        )


@app.on_event("startup")
async def startup() -> None:
    settings = get_settings()
    if not settings.MOOSE_API_KEY:
        raise RuntimeError("MOOSE_API_KEY is required for the API service")
    job_store, queue_backend, info = await build_backends(settings)
    llm_client = None
    if settings.MOOSE_LLM_PROVIDER.lower() != "openrouter":
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
    request_llm: LLMOverrides,
    llm_api_key: str | None,
    llm_endpoint: str | None,
) -> dict[str, Any]:
    llm_payload = request_llm.model_dump()
    if llm_api_key:
        if request_llm.provider.lower() == "openrouter":
            llm_payload["openrouter_api_key"] = llm_api_key
        else:
            llm_payload["ollama_token"] = llm_api_key
    if llm_endpoint:
        llm_payload["endpoint"] = llm_endpoint
    return llm_payload


@app.post(
    "/ner",
    dependencies=[Depends(require_api_key)],
    tags=["NER"],
    response_model=JobQueuedResponse,
)
async def submit_ner(
    request: NERRequest,
    llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
    llm_endpoint: str | None = Header(default=None, alias="X-LLM-Endpoint"),
):
    _require_llm_overrides(request.llm, llm_api_key)
    _ensure_schema_supported(request.schema, require_text=True)
    payload = {
        "schema": request.schema,
        "tasks": [task.model_dump() for task in request.tasks],
        "include_scores": request.include_scores,
        "llm": _build_llm_payload(request.llm, llm_api_key, llm_endpoint),
    }
    job_id = await _enqueue_job("ner", payload)
    return JobQueuedResponse(job_id=job_id, status="queued")


@app.post(
    "/schemas/{schema}/ner",
    dependencies=[Depends(require_api_key)],
    tags=["Schemas"],
    response_model=JobQueuedResponse,
)
async def submit_schema_ner(
    schema: str,
    request: SchemaNERRequest,
    llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
    llm_endpoint: str | None = Header(default=None, alias="X-LLM-Endpoint"),
):
    _require_llm_overrides(request.llm, llm_api_key)
    _ensure_schema_supported(schema, require_text=True)
    payload = {
        "schema": schema,
        "tasks": [task.model_dump() for task in request.tasks],
        "include_scores": request.include_scores,
        "llm": _build_llm_payload(request.llm, llm_api_key, llm_endpoint),
    }
    job_id = await _enqueue_job("ner", payload)
    return JobQueuedResponse(job_id=job_id, status="queued")


@app.post(
    "/dpv/ner",
    dependencies=[Depends(require_api_key)],
    tags=["DPV"],
    response_model=JobQueuedResponse,
    deprecated=True,
)
async def submit_dpv_ner(
    request: DpvNERRequest,
    llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
    llm_endpoint: str | None = Header(default=None, alias="X-LLM-Endpoint"),
):
    _require_llm_overrides(request.llm, llm_api_key)
    _ensure_schema_supported("dpv", require_text=True)
    payload = {
        "schema": "dpv",
        "tasks": [task.model_dump() for task in request.tasks],
        "include_scores": request.include_scores,
        "llm": _build_llm_payload(request.llm, llm_api_key, llm_endpoint),
    }
    job_id = await _enqueue_job("ner", payload)
    return JobQueuedResponse(job_id=job_id, status="queued")


@app.post(
    "/tabular/annotate",
    dependencies=[Depends(require_api_key)],
    tags=["Tabular"],
    response_model=JobQueuedResponse,
)
async def submit_tabular(
    request: TabularRequest,
    llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
    llm_endpoint: str | None = Header(default=None, alias="X-LLM-Endpoint"),
):
    _require_llm_overrides(request.llm, llm_api_key)
    _ensure_schema_supported(request.schema, require_table=True)
    payload = {
        "schema": request.schema,
        "tasks": [task.model_dump() for task in request.tasks],
        "include_scores": request.include_scores,
        "llm": _build_llm_payload(request.llm, llm_api_key, llm_endpoint),
    }
    job_id = await _enqueue_job("tabular", payload)
    return JobQueuedResponse(job_id=job_id, status="queued")


@app.post(
    "/schemas/{schema}/tabular/annotate",
    dependencies=[Depends(require_api_key)],
    tags=["Schemas"],
    response_model=JobQueuedResponse,
)
async def submit_schema_tabular(
    schema: str,
    request: SchemaTabularRequest,
    llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
    llm_endpoint: str | None = Header(default=None, alias="X-LLM-Endpoint"),
):
    _require_llm_overrides(request.llm, llm_api_key)
    _ensure_schema_supported(schema, require_table=True)
    payload = {
        "schema": schema,
        "tasks": [task.model_dump() for task in request.tasks],
        "include_scores": request.include_scores,
        "llm": _build_llm_payload(request.llm, llm_api_key, llm_endpoint),
    }
    job_id = await _enqueue_job("tabular", payload)
    return JobQueuedResponse(job_id=job_id, status="queued")


@app.post(
    "/dpv/tabular/annotate",
    dependencies=[Depends(require_api_key)],
    tags=["DPV"],
    response_model=JobQueuedResponse,
    deprecated=True,
)
async def submit_dpv_tabular(
    request: DpvTabularRequest,
    llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
    llm_endpoint: str | None = Header(default=None, alias="X-LLM-Endpoint"),
):
    _require_llm_overrides(request.llm, llm_api_key)
    _ensure_schema_supported("dpv", require_table=True)
    payload = {
        "schema": "dpv",
        "tasks": [task.model_dump() for task in request.tasks],
        "include_scores": request.include_scores,
        "llm": _build_llm_payload(request.llm, llm_api_key, llm_endpoint),
    }
    job_id = await _enqueue_job("tabular", payload)
    return JobQueuedResponse(job_id=job_id, status="queued")


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
    provider: Literal["ollama", "openrouter", "all"] = "openrouter",
    llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
    llm_endpoint: str | None = Header(default=None, alias="X-LLM-Endpoint"),
):
    settings: Settings = app.state.settings
    results: dict[str, Any] = {}
    if provider in {"ollama", "all"}:
        headers = {}
        if llm_api_key:
            headers["Authorization"] = f"Bearer {llm_api_key}"
        base_url = settings.MOOSE_OLLAMA_HOST
        if provider == "ollama" and llm_endpoint:
            base_url = llm_endpoint
        try:
            async with httpx.AsyncClient(
                base_url=base_url,
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
        if not llm_api_key:
            results["openrouter"] = {"error": "OpenRouter API key is required"}
        else:
            try:
                base_url = settings.MOOSE_OPENROUTER_BASE_URL
                if provider == "openrouter" and llm_endpoint:
                    base_url = llm_endpoint
                headers = {"Authorization": f"Bearer {llm_api_key}"}
                async with httpx.AsyncClient(
                    base_url=base_url,
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
