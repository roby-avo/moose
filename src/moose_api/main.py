from __future__ import annotations

import uuid
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from moose.config import Settings, get_settings
from moose.llm import create_client
from moose_api.queue import (
    JobRecord,
    WorkerPool,
    build_backends,
    payload_hash,
    utc_now,
)


class LLMOverrides(BaseModel):
    provider: Literal["ollama", "openrouter"] | None = None
    model: str | None = None
    ollama_host: str | None = None
    ollama_token: str | None = None
    openrouter_api_key: str | None = None
    openrouter_base_url: str | None = None


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


app = FastAPI(title="Moose API", version="0.1.0")


@app.on_event("startup")
async def startup() -> None:
    settings = get_settings()
    job_store, queue_backend, info = await build_backends(settings)
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


async def _enqueue_job(endpoint_type: str, payload: dict, idempotency_key: str | None):
    settings: Settings = app.state.settings
    queue_backend = app.state.queue_backend
    job_store = app.state.job_store

    if idempotency_key:
        existing = await job_store.get_idempotency(idempotency_key)
        if existing:
            if existing.get("payload_hash") == payload_hash(payload):
                return existing.get("job_id")
            raise HTTPException(status_code=409, detail="Idempotency key conflict")

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
    if idempotency_key:
        await job_store.set_idempotency(
            idempotency_key,
            {"payload_hash": payload_hash(payload), "job_id": job_id},
        )
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


@app.post("/v1/ner")
async def submit_ner(
    request: NERRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    payload = {
        "schema": request.schema,
        "tasks": [task.model_dump() for task in request.tasks],
        "include_scores": request.include_scores,
        "llm": request.llm.model_dump(exclude_none=True) if request.llm else None,
    }
    job_id = await _enqueue_job("ner", payload, idempotency_key)
    return {"job_id": job_id, "status": "queued"}


@app.post("/v1/tabular/annotate")
async def submit_tabular(
    request: TabularRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    payload = {
        "schema": request.schema,
        "tasks": [task.model_dump() for task in request.tasks],
        "include_scores": request.include_scores,
        "llm": request.llm.model_dump(exclude_none=True) if request.llm else None,
    }
    job_id = await _enqueue_job("tabular", payload, idempotency_key)
    return {"job_id": job_id, "status": "queued"}


@app.get("/v1/jobs/{job_id}")
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


@app.get("/health")
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
