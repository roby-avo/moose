from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Security
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from moose.config import Settings, get_settings
from moose.llm import create_client
from moose.pipelines import load_privacy_profiles
from moose.schema import get_schema_config, list_schema_names
from moose.privacy import list_policy_packs
from moose_api.queue import JobRecord, WorkerPool, build_backends, utc_now


class LLMOverrides(BaseModel):
    provider: Literal["openrouter", "ollama", "deepinfra", "deepseek"]
    model: str


# -----------------------
# Text NER request models
# -----------------------
class NERTaskIn(BaseModel):
    task_id: str
    text: str


class BaseNERRequest(BaseModel):
    tasks: list[NERTaskIn] = Field(min_length=1)
    include_scores: bool = False
    strict_offsets: bool = False
    llm: LLMOverrides


class NERRequest(BaseNERRequest):
    schema: str = Field(
        description="Schema/vocabulary name to annotate against.",
        examples=["coarse", "fine", "dpv", "dpv_pd"],
    )

    @field_validator("schema")
    @classmethod
    def validate_schema(cls, value: str) -> str:
        try:
            get_schema_config(value)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return value


# -----------------------------
# Tabular typing request models
# -----------------------------
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
        examples=["dpv", "dpv_pd", "sti", "schemaorg_cta_v1"],
    )

    @field_validator("schema")
    @classmethod
    def validate_schema(cls, value: str) -> str:
        try:
            get_schema_config(value)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return value


# -----------------------------
# Tabular cell NER models
# -----------------------------
class TabularNerTaskIn(BaseModel):
    task_id: str
    table_id: str
    sampled_rows: list[dict[str, Any]] = Field(min_length=1)

    target_columns: list[str] = Field(
        min_length=1,
        description="Columns to run cell-level NER on.",
    )

    strings_only: bool = True
    skip_structured_literals: bool = True


class BaseTabularNERRequest(BaseModel):
    tasks: list[TabularNerTaskIn] = Field(min_length=1)
    include_scores: bool = False
    strict_offsets: bool = False
    llm: LLMOverrides


class TabularNERRequest(BaseTabularNERRequest):
    schema: str = Field(
        description="Schema/vocabulary name to annotate against (must support text NER).",
        examples=["coarse", "fine", "dpv", "dpv_pd"],
    )

    @field_validator("schema")
    @classmethod
    def validate_schema(cls, value: str) -> str:
        try:
            get_schema_config(value)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return value


# -----------------------------
# CPA (column relationship) request models
# -----------------------------
class CPATaskIn(BaseModel):
    task_id: str
    table_id: str
    sampled_rows: list[dict[str, Any]] = Field(min_length=1)

    subject_column: str = Field(min_length=1, description="Subject column name (required).")

    # OPTIONAL: if known, helps filter schema.org predicates using domainIncludes + class hierarchy.
    # Example: "schema:Book"
    subject_class: str | None = Field(
        default=None,
        description="Optional schema.org class CURIE for the subject column (e.g., schema:Book). "
                    "If omitted and the CPA schema is schema.org-based, Moose will infer it using schemaorg_cta_v1 by default.",
    )

    target_columns: list[str] | None = None

    # Optional performance knob for deep CPA:
    # If enabled, Moose runs STI typing once and caches selection by STI type signature across columns.
    use_sti_signature_cache: bool = True

    # OPTIONAL: debug output (per task)
    debug: bool = False
    debug_preview_limit: int = Field(default=20, ge=0, le=200)

class BaseCPARequest(BaseModel):
    tasks: list[CPATaskIn] = Field(min_length=1)
    include_scores: bool = False
    llm: LLMOverrides


class CPARequest(BaseCPARequest):
    schema: str = Field(
        default="cpa",
        description="CPA relationship schema name (must support CPA).",
        examples=["cpa", "schemaorg_cpa_v1"],
    )

    @field_validator("schema")
    @classmethod
    def validate_schema(cls, value: str) -> str:
        try:
            get_schema_config(value)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return value


# -----------------------------
# Privacy analysis request models
# -----------------------------
class PrivacyTextTaskIn(BaseModel):
    kind: Literal["text"] = "text"
    task_id: str
    text: str
    context: dict[str, Any] | None = None


class PrivacyTableTaskIn(BaseModel):
    kind: Literal["table"] = "table"
    task_id: str
    table_id: str
    sampled_rows: list[dict[str, Any]] = Field(min_length=1)
    scan_columns: list[str] | None = None
    context: dict[str, Any] | None = None


PrivacyTaskIn = Annotated[Union[PrivacyTextTaskIn, PrivacyTableTaskIn], Field(discriminator="kind")]


class PrivacyAnalyzeRequest(BaseModel):
    # NEW: profile support (fast/balanced/deep)
    profile: str | None = None

    # These are now OPTIONAL; if omitted, profile defaults will apply in moose/privacy.py.
    policy_pack: str | None = None
    analysis_mode: Literal["rules", "hybrid"] | None = None
    text_schema: str | None = None
    table_schema: str | None = None
    scan_schema: str | None = None
    include_extraction: bool | None = None

    tasks: list[PrivacyTaskIn] = Field(min_length=1)
    llm: LLMOverrides


# -----------------------
# Schema-specific wrappers
# -----------------------
class SchemaNERRequest(BaseNERRequest):
    pass


class SchemaTabularRequest(BaseTabularRequest):
    pass


class SchemaTabularNERRequest(BaseTabularNERRequest):
    pass


class SchemaCPARequest(BaseCPARequest):
    pass


class DpvNERRequest(BaseNERRequest):
    pass


class DpvTabularRequest(BaseTabularRequest):
    pass


# -------------
# Job responses
# -------------
class JobQueuedResponse(BaseModel):
    job_id: str
    status: Literal["queued"]


STATIC_DIR = Path(__file__).resolve().parent / "static"

TAG_METADATA = [
    {"name": "NER", "description": "Named entity recognition endpoints."},
    {"name": "Tabular", "description": "Tabular semantic typing endpoints."},
    {"name": "Tabular NER", "description": "Cell-level NER over selected columns of sampled rows."},
    {"name": "CPA", "description": "Column Property Annotation (CPA) / column relationship prediction."},
    {"name": "Schemas", "description": "Schema-specific annotation endpoints."},
    {"name": "DPV", "description": "DPV classification endpoints."},
    {"name": "Privacy", "description": "Privacy analysis orchestration endpoint."},
    {"name": "Metadata", "description": "Metadata endpoints used by frontends (schemas, policy packs, profiles, assets)."},
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


def _require_llm_overrides(request_llm: LLMOverrides, llm_api_key: str | None) -> None:
    provider = request_llm.provider.lower()
    if provider in {"openrouter", "deepinfra", "deepseek"} and not llm_api_key:
        raise HTTPException(
            status_code=400,
            detail="LLM API key is required via X-LLM-API-Key for this provider.",
        )


def _ensure_schema_supported(
    schema: str,
    require_text: bool = False,
    require_table: bool = False,
    require_cpa: bool = False,
) -> None:
    try:
        config = get_schema_config(schema)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if require_text and not config.supports_text:
        raise HTTPException(status_code=400, detail=f"Schema '{schema}' does not support text annotation.")
    if require_table and not config.supports_table:
        raise HTTPException(status_code=400, detail=f"Schema '{schema}' does not support tabular annotation.")
    if require_cpa and not getattr(config, "supports_cpa", False):
        raise HTTPException(status_code=400, detail=f"Schema '{schema}' does not support CPA annotation.")


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
    if llm_client is not None:
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
    provider = request_llm.provider.lower()

    if llm_api_key:
        if provider == "openrouter":
            llm_payload["openrouter_api_key"] = llm_api_key
        elif provider == "ollama":
            llm_payload["ollama_token"] = llm_api_key
        elif provider == "deepinfra":
            llm_payload["deepinfra_api_key"] = llm_api_key
        elif provider == "deepseek":
            llm_payload["deepseek_api_key"] = llm_api_key

    if llm_endpoint:
        llm_payload["endpoint"] = llm_endpoint

    return llm_payload


# -------------
# Text NER APIs
# -------------
@app.post("/ner", dependencies=[Depends(require_api_key)], tags=["NER"], response_model=JobQueuedResponse)
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
        "strict_offsets": request.strict_offsets,
        "llm": _build_llm_payload(request.llm, llm_api_key, llm_endpoint),
    }
    job_id = await _enqueue_job("ner", payload)
    return JobQueuedResponse(job_id=job_id, status="queued")


@app.post("/schemas/{schema}/ner", dependencies=[Depends(require_api_key)], tags=["Schemas"], response_model=JobQueuedResponse)
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
        "strict_offsets": request.strict_offsets,
        "llm": _build_llm_payload(request.llm, llm_api_key, llm_endpoint),
    }
    job_id = await _enqueue_job("ner", payload)
    return JobQueuedResponse(job_id=job_id, status="queued")


@app.post("/dpv/ner", dependencies=[Depends(require_api_key)], tags=["DPV"], response_model=JobQueuedResponse, deprecated=True)
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
        "strict_offsets": request.strict_offsets,
        "llm": _build_llm_payload(request.llm, llm_api_key, llm_endpoint),
    }
    job_id = await _enqueue_job("ner", payload)
    return JobQueuedResponse(job_id=job_id, status="queued")


# -----------------
# Tabular typing APIs
# -----------------
@app.post("/tabular/annotate", dependencies=[Depends(require_api_key)], tags=["Tabular"], response_model=JobQueuedResponse)
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


@app.post("/schemas/{schema}/tabular/annotate", dependencies=[Depends(require_api_key)], tags=["Schemas"], response_model=JobQueuedResponse)
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


@app.post("/dpv/tabular/annotate", dependencies=[Depends(require_api_key)], tags=["DPV"], response_model=JobQueuedResponse, deprecated=True)
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


# ------------------------
# Tabular cell NER APIs
# ------------------------
@app.post("/tabular/ner", dependencies=[Depends(require_api_key)], tags=["Tabular NER"], response_model=JobQueuedResponse)
async def submit_tabular_ner(
    request: TabularNERRequest,
    llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
    llm_endpoint: str | None = Header(default=None, alias="X-LLM-Endpoint"),
):
    _require_llm_overrides(request.llm, llm_api_key)
    _ensure_schema_supported(request.schema, require_text=True)

    payload = {
        "schema": request.schema,
        "tasks": [task.model_dump() for task in request.tasks],
        "include_scores": request.include_scores,
        "strict_offsets": request.strict_offsets,
        "llm": _build_llm_payload(request.llm, llm_api_key, llm_endpoint),
    }
    job_id = await _enqueue_job("tabular_ner", payload)
    return JobQueuedResponse(job_id=job_id, status="queued")


@app.post("/schemas/{schema}/tabular/ner", dependencies=[Depends(require_api_key)], tags=["Schemas"], response_model=JobQueuedResponse)
async def submit_schema_tabular_ner(
    schema: str,
    request: SchemaTabularNERRequest,
    llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
    llm_endpoint: str | None = Header(default=None, alias="X-LLM-Endpoint"),
):
    _require_llm_overrides(request.llm, llm_api_key)
    _ensure_schema_supported(schema, require_text=True)

    payload = {
        "schema": schema,
        "tasks": [task.model_dump() for task in request.tasks],
        "include_scores": request.include_scores,
        "strict_offsets": request.strict_offsets,
        "llm": _build_llm_payload(request.llm, llm_api_key, llm_endpoint),
    }
    job_id = await _enqueue_job("tabular_ner", payload)
    return JobQueuedResponse(job_id=job_id, status="queued")


# ------------------------
# CPA endpoints
# ------------------------
@app.post("/tabular/cpa", dependencies=[Depends(require_api_key)], tags=["CPA"], response_model=JobQueuedResponse)
async def submit_tabular_cpa(
    request: CPARequest,
    llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
    llm_endpoint: str | None = Header(default=None, alias="X-LLM-Endpoint"),
):
    _require_llm_overrides(request.llm, llm_api_key)
    _ensure_schema_supported(request.schema, require_cpa=True)

    payload = {
        "schema": request.schema,
        "tasks": [task.model_dump() for task in request.tasks],
        "include_scores": request.include_scores,
        "llm": _build_llm_payload(request.llm, llm_api_key, llm_endpoint),
    }
    # Note: subject_class and use_sti_signature_cache are now part of task.model_dump()
    job_id = await _enqueue_job("cpa", payload)
    return JobQueuedResponse(job_id=job_id, status="queued")


@app.post("/schemas/{schema}/tabular/cpa", dependencies=[Depends(require_api_key)], tags=["Schemas"], response_model=JobQueuedResponse)
async def submit_schema_tabular_cpa(
    schema: str,
    request: SchemaCPARequest,
    llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
    llm_endpoint: str | None = Header(default=None, alias="X-LLM-Endpoint"),
):
    _require_llm_overrides(request.llm, llm_api_key)
    _ensure_schema_supported(schema, require_cpa=True)

    payload = {
        "schema": schema,
        "tasks": [task.model_dump() for task in request.tasks],
        "include_scores": request.include_scores,
        "llm": _build_llm_payload(request.llm, llm_api_key, llm_endpoint),
    }
    # Note: subject_class and use_sti_signature_cache are now part of task.model_dump()
    job_id = await _enqueue_job("cpa", payload)
    return JobQueuedResponse(job_id=job_id, status="queued")


# ------------------------
# Privacy analysis API
# ------------------------
@app.post("/privacy/analyze", dependencies=[Depends(require_api_key)], tags=["Privacy"], response_model=JobQueuedResponse)
async def submit_privacy_analyze(
    request: PrivacyAnalyzeRequest,
    llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
    llm_endpoint: str | None = Header(default=None, alias="X-LLM-Endpoint"),
):
    """
    NOTE: We do not validate text_schema/table_schema/scan_schema here anymore,
    because they can be omitted and resolved by `profile` defaults in moose/privacy.py.
    """
    _require_llm_overrides(request.llm, llm_api_key)

    payload = {
        "profile": request.profile,
        "policy_pack": request.policy_pack,
        "analysis_mode": request.analysis_mode,
        "text_schema": request.text_schema,
        "table_schema": request.table_schema,
        "scan_schema": request.scan_schema,
        "include_extraction": request.include_extraction,
        "tasks": [t.model_dump() for t in request.tasks],
        "llm": _build_llm_payload(request.llm, llm_api_key, llm_endpoint),
    }
    job_id = await _enqueue_job("privacy_analyze", payload)
    return JobQueuedResponse(job_id=job_id, status="queued")


# ----------
# Metadata / jobs
# ----------
@app.get("/jobs/{job_id}", dependencies=[Depends(require_api_key)], tags=["Metadata"])
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


@app.get("/schemas", dependencies=[Depends(require_api_key)], tags=["Metadata"])
async def list_schemas(include_type_count: bool = False) -> dict[str, Any]:
    schemas = []
    for name in list_schema_names():
        cfg = get_schema_config(name)
        item: dict[str, Any] = {
            "name": cfg.name,
            "label": cfg.label,
            "description": cfg.description,
            "supports_text": bool(cfg.supports_text),
            "supports_table": bool(cfg.supports_table),
            "supports_cpa": bool(getattr(cfg, "supports_cpa", False)),
            "prefilter_types": bool(getattr(cfg, "prefilter_types", False)),
            "score_mode": "dense" if cfg.require_all_scores else "sparse",
        }
        if include_type_count:
            try:
                item["type_count"] = len(cfg.load_type_ids())
            except Exception:  # noqa: BLE001
                item["type_count"] = None
        schemas.append(item)
    return {"schemas": schemas}


@app.get("/policy-packs", dependencies=[Depends(require_api_key)], tags=["Metadata"])
async def list_policy_packs_endpoint() -> dict[str, Any]:
    return {"policy_packs": list_policy_packs()}


@app.get("/privacy/profiles", dependencies=[Depends(require_api_key)], tags=["Metadata"])
async def get_privacy_profiles_endpoint() -> dict[str, Any]:
    """
    Return pipelines/privacy_profiles.json so frontends can show available profiles.
    """
    return load_privacy_profiles()


@app.get("/health", dependencies=[Depends(require_api_key)], tags=["Metadata"])
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


@app.get("/models", dependencies=[Depends(require_api_key)], tags=["Metadata"])
async def list_models(
    provider: Literal["ollama", "openrouter", "deepinfra", "deepseek", "all"] = "openrouter",
    llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
    llm_endpoint: str | None = Header(default=None, alias="X-LLM-Endpoint"),
):
    """
    List available models from specified provider(s).

    Notes:
    - OpenRouter: supported via /models
    - Ollama: supported via /api/tags
    - DeepInfra/DeepSeek: model listing is not currently supported by Moose (provide model name manually)
    """
    settings: Settings = app.state.settings
    results: dict[str, Any] = {}

    if provider in {"deepinfra", "all"}:
        results["deepinfra"] = {
            "error": "Model listing is not supported for DeepInfra in this API. Provide model manually (e.g. Qwen/Qwen3-Next-80B-A3B-Instruct)."
        }

    if provider in {"deepseek", "all"}:
        results["deepseek"] = {
            "error": "Model listing is not supported for DeepSeek in this API. Provide model manually (e.g. deepseek-chat)."
        }

    if provider in {"ollama", "all"}:
        headers = {}
        if llm_api_key:
            headers["Authorization"] = f"Bearer {llm_api_key}"

        base_url = settings.MOOSE_OLLAMA_HOST
        if provider == "ollama" and llm_endpoint:
            base_url = llm_endpoint

        try:
            async with httpx.AsyncClient(base_url=base_url, timeout=settings.MOOSE_TIMEOUT_SECS) as client:
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
                async with httpx.AsyncClient(base_url=base_url, timeout=settings.MOOSE_TIMEOUT_SECS) as client:
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


@app.get("/assets", dependencies=[Depends(require_api_key)], tags=["Metadata"])
async def get_assets_index() -> dict:
    from moose.schema import DATA_DIR
    path = DATA_DIR / "assets_index.json"
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"assets_index.json not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="Moose API Docs",
        swagger_favicon_url="/static/moose-logo.png",
        swagger_css_url="/static/docs.css",
    )