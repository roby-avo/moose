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
from pydantic import BaseModel, ConfigDict, Field, field_validator

from moose.config import Settings, get_settings
from moose.ingest import delete_user_schema, preview_schema_payload, update_user_schema_metadata
from moose.llm import create_client
from moose.pipelines import load_privacy_profiles
from moose.schema import get_schema_config, list_schema_names, reload_schema_registry
from moose.privacy import get_machine_report_json_schema, list_policy_packs
from moose_api.queue import JobRecord, WorkerPool, build_backends, utc_now


class LLMOverrides(BaseModel):
    provider: Literal["openrouter", "ollama", "deepinfra", "deepseek"]
    model: str


# -----------------------
# Text NER request models
# -----------------------
class BaseNERRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    llm: LLMOverrides

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must be non-empty.")
        return value


class NERRequest(BaseNERRequest):
    schema_name: str = Field(
        alias="schema",
        serialization_alias="schema",
        description="Schema/vocabulary name to annotate against.",
        examples=["coarse", "fine", "dpv", "dpv_pd"],
    )

    @field_validator("schema_name")
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
class BaseTabularRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_id: str | None = None
    sampled_rows: list[dict[str, Any]] = Field(min_length=1)
    llm: LLMOverrides


class TabularRequest(BaseTabularRequest):
    schema_name: str = Field(
        alias="schema",
        serialization_alias="schema",
        description="Schema/vocabulary name to annotate against.",
        examples=["dpv", "dpv_pd", "sti", "schemaorg_cta_v1"],
    )

    @field_validator("schema_name")
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
class BaseCPARequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_id: str | None = None
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

    llm: LLMOverrides

    @field_validator("subject_column")
    @classmethod
    def validate_subject_column(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("subject_column must be non-empty.")
        return value


class CPARequest(BaseCPARequest):
    schema_name: str = Field(
        alias="schema",
        serialization_alias="schema",
        default="cpa",
        description="CPA relationship schema name (must support CPA).",
        examples=["cpa", "schemaorg_cpa_v1"],
    )

    @field_validator("schema_name")
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


class SchemaCPARequest(BaseCPARequest):
    pass


class DpvNERRequest(BaseNERRequest):
    pass


class DpvTabularRequest(BaseTabularRequest):
    pass


class SchemaIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Schema registry name.")
    label: str | None = None
    description: str | None = None

    source_type: Literal["url", "github", "file"] = "url"
    source_url: str | None = None
    source_path: str | None = None
    github_repo: str | None = None
    github_ref: str | None = "main"
    github_path: str | None = None

    use_llm_fallback: bool = False
    llm: LLMOverrides | None = None

    score_mode: Literal["dense", "sparse"] = "sparse"
    prefilter_types: bool = False
    supports_text: bool = True
    supports_table: bool = True
    supports_cpa: bool = False
    text_intro: str | None = None
    table_intro: str | None = None
    cpa_intro: str | None = None
    expected_source_sha256: str | None = None
    expected_mapping_sha256: str | None = None


class SchemaPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str | None = None
    description: str | None = None
    score_mode: Literal["dense", "sparse"] | None = None
    prefilter_types: bool | None = None
    supports_text: bool | None = None
    supports_table: bool | None = None
    supports_cpa: bool | None = None
    text_intro: str | None = None
    table_intro: str | None = None
    cpa_intro: str | None = None


# -------------
# Job responses
# -------------
class JobQueuedResponse(BaseModel):
    job_id: str
    status: Literal["queued"]


class SchemaIngestPreviewResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_name: str = Field(alias="schema", serialization_alias="schema")
    label: str
    description: str
    type_count: int
    sample_type_ids: list[str]
    source: dict[str, Any]
    source_sha256: str
    mapping_sha256: str
    mapping_strategy: Literal["deterministic", "llm_fallback"]
    mapping_confidence: float | None = None
    mapping: dict[str, Any]
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    guardrails: dict[str, Any] = Field(default_factory=dict)
    can_activate: bool = True
    activated: bool = False


STATIC_DIR = Path(__file__).resolve().parent / "static"

TAG_METADATA = [
    {"name": "NER", "description": "Named entity recognition endpoints."},
    {"name": "Tabular", "description": "Tabular semantic typing endpoints."},
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
llm_api_key_header = APIKeyHeader(
    name="X-LLM-API-Key",
    auto_error=False,
    scheme_name="X-LLM-API-Key",
    description="Provider API key used by LLM backends (OpenRouter, DeepInfra, DeepSeek, optional for Ollama).",
)


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


def _new_job_id() -> str:
    # Compact UUID form for easier copy/paste while keeping enough entropy.
    return uuid.uuid4().hex


async def _enqueue_job(endpoint_type: str, payload: dict):
    settings: Settings = app.state.settings
    queue_backend = app.state.queue_backend
    job_store = app.state.job_store

    queue_size = await queue_backend.size()
    if queue_size >= settings.MOOSE_QUEUE_MAXSIZE:
        raise HTTPException(status_code=429, detail="Queue is full, try again later")

    job_id = _new_job_id()
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


def _settings_with_llm_payload(settings: Settings, llm_payload: dict[str, Any]) -> Settings:
    update: dict[str, Any] = {}
    provider = str(llm_payload.get("provider") or settings.MOOSE_LLM_PROVIDER).lower()

    if llm_payload.get("provider"):
        update["MOOSE_LLM_PROVIDER"] = provider
    if llm_payload.get("model"):
        update["MOOSE_MODEL"] = llm_payload["model"]

    if llm_payload.get("ollama_token"):
        update["MOOSE_OLLAMA_TOKEN"] = llm_payload["ollama_token"]
    if llm_payload.get("openrouter_api_key"):
        update["MOOSE_OPENROUTER_API_KEY"] = llm_payload["openrouter_api_key"]
    if llm_payload.get("deepinfra_api_key"):
        update["MOOSE_DEEPINFRA_API_KEY"] = llm_payload["deepinfra_api_key"]
    if llm_payload.get("deepseek_api_key"):
        update["MOOSE_DEEPSEEK_API_KEY"] = llm_payload["deepseek_api_key"]

    endpoint = llm_payload.get("endpoint")
    if endpoint:
        if provider == "openrouter":
            update["MOOSE_OPENROUTER_BASE_URL"] = endpoint
        elif provider == "ollama":
            update["MOOSE_OLLAMA_HOST"] = endpoint
        elif provider == "deepinfra":
            update["MOOSE_DEEPINFRA_BASE_URL"] = endpoint
        elif provider == "deepseek":
            update["MOOSE_DEEPSEEK_BASE_URL"] = endpoint

    if not update:
        return settings
    return settings.model_copy(update=update)


def _validate_schema_ingest_request(request: SchemaIngestRequest) -> None:
    if request.source_type == "url" and not request.source_url:
        raise HTTPException(status_code=400, detail="source_url is required when source_type='url'.")
    if request.source_type == "file" and not request.source_path:
        raise HTTPException(status_code=400, detail="source_path is required when source_type='file'.")
    if request.source_type == "github" and (not request.github_repo or not request.github_path):
        raise HTTPException(
            status_code=400,
            detail="github_repo and github_path are required when source_type='github'.",
        )


def _build_schema_ingest_payload(request: SchemaIngestRequest) -> dict[str, Any]:
    return {
        "name": request.name,
        "label": request.label,
        "description": request.description,
        "source_type": request.source_type,
        "source_url": request.source_url,
        "source_path": request.source_path,
        "github_repo": request.github_repo,
        "github_ref": request.github_ref,
        "github_path": request.github_path,
        "use_llm_fallback": request.use_llm_fallback,
        "score_mode": request.score_mode,
        "prefilter_types": request.prefilter_types,
        "supports_text": request.supports_text,
        "supports_table": request.supports_table,
        "supports_cpa": request.supports_cpa,
        "text_intro": request.text_intro,
        "table_intro": request.table_intro,
        "cpa_intro": request.cpa_intro,
        "expected_source_sha256": request.expected_source_sha256,
        "expected_mapping_sha256": request.expected_mapping_sha256,
    }


# -------------
# Text NER APIs
# -------------
@app.post("/ner", dependencies=[Depends(require_api_key)], tags=["NER"], response_model=JobQueuedResponse)
async def submit_ner(
    request: NERRequest,
    llm_api_key: str | None = Security(llm_api_key_header),
    llm_endpoint: str | None = Header(default=None, alias="X-LLM-Endpoint"),
):
    _require_llm_overrides(request.llm, llm_api_key)
    _ensure_schema_supported(request.schema_name, require_text=True)

    payload = {
        "schema": request.schema_name,
        "text": request.text,
        "llm": _build_llm_payload(request.llm, llm_api_key, llm_endpoint),
    }
    job_id = await _enqueue_job("ner", payload)
    return JobQueuedResponse(job_id=job_id, status="queued")


@app.post("/schemas/{schema}/ner", dependencies=[Depends(require_api_key)], tags=["Schemas"], response_model=JobQueuedResponse)
async def submit_schema_ner(
    schema: str,
    request: SchemaNERRequest,
    llm_api_key: str | None = Security(llm_api_key_header),
    llm_endpoint: str | None = Header(default=None, alias="X-LLM-Endpoint"),
):
    _require_llm_overrides(request.llm, llm_api_key)
    _ensure_schema_supported(schema, require_text=True)

    payload = {
        "schema": schema,
        "text": request.text,
        "llm": _build_llm_payload(request.llm, llm_api_key, llm_endpoint),
    }
    job_id = await _enqueue_job("ner", payload)
    return JobQueuedResponse(job_id=job_id, status="queued")


@app.post("/dpv/ner", dependencies=[Depends(require_api_key)], tags=["DPV"], response_model=JobQueuedResponse, deprecated=True)
async def submit_dpv_ner(
    request: DpvNERRequest,
    llm_api_key: str | None = Security(llm_api_key_header),
    llm_endpoint: str | None = Header(default=None, alias="X-LLM-Endpoint"),
):
    _require_llm_overrides(request.llm, llm_api_key)
    _ensure_schema_supported("dpv", require_text=True)

    payload = {
        "schema": "dpv",
        "text": request.text,
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
    llm_api_key: str | None = Security(llm_api_key_header),
    llm_endpoint: str | None = Header(default=None, alias="X-LLM-Endpoint"),
):
    _require_llm_overrides(request.llm, llm_api_key)
    _ensure_schema_supported(request.schema_name, require_table=True)

    payload = {
        "schema": request.schema_name,
        "table_id": request.table_id,
        "sampled_rows": request.sampled_rows,
        "llm": _build_llm_payload(request.llm, llm_api_key, llm_endpoint),
    }
    job_id = await _enqueue_job("tabular", payload)
    return JobQueuedResponse(job_id=job_id, status="queued")


@app.post("/schemas/{schema}/tabular/annotate", dependencies=[Depends(require_api_key)], tags=["Schemas"], response_model=JobQueuedResponse)
async def submit_schema_tabular(
    schema: str,
    request: SchemaTabularRequest,
    llm_api_key: str | None = Security(llm_api_key_header),
    llm_endpoint: str | None = Header(default=None, alias="X-LLM-Endpoint"),
):
    _require_llm_overrides(request.llm, llm_api_key)
    _ensure_schema_supported(schema, require_table=True)

    payload = {
        "schema": schema,
        "table_id": request.table_id,
        "sampled_rows": request.sampled_rows,
        "llm": _build_llm_payload(request.llm, llm_api_key, llm_endpoint),
    }
    job_id = await _enqueue_job("tabular", payload)
    return JobQueuedResponse(job_id=job_id, status="queued")


@app.post("/dpv/tabular/annotate", dependencies=[Depends(require_api_key)], tags=["DPV"], response_model=JobQueuedResponse, deprecated=True)
async def submit_dpv_tabular(
    request: DpvTabularRequest,
    llm_api_key: str | None = Security(llm_api_key_header),
    llm_endpoint: str | None = Header(default=None, alias="X-LLM-Endpoint"),
):
    _require_llm_overrides(request.llm, llm_api_key)
    _ensure_schema_supported("dpv", require_table=True)

    payload = {
        "schema": "dpv",
        "table_id": request.table_id,
        "sampled_rows": request.sampled_rows,
        "llm": _build_llm_payload(request.llm, llm_api_key, llm_endpoint),
    }
    job_id = await _enqueue_job("tabular", payload)
    return JobQueuedResponse(job_id=job_id, status="queued")


# ------------------------
# CPA endpoints
# ------------------------
@app.post("/tabular/cpa", dependencies=[Depends(require_api_key)], tags=["CPA"], response_model=JobQueuedResponse)
async def submit_tabular_cpa(
    request: CPARequest,
    llm_api_key: str | None = Security(llm_api_key_header),
    llm_endpoint: str | None = Header(default=None, alias="X-LLM-Endpoint"),
):
    _require_llm_overrides(request.llm, llm_api_key)
    _ensure_schema_supported(request.schema_name, require_cpa=True)

    payload = {
        "schema": request.schema_name,
        "table_id": request.table_id,
        "sampled_rows": request.sampled_rows,
        "subject_column": request.subject_column,
        "subject_class": request.subject_class,
        "target_columns": request.target_columns,
        "use_sti_signature_cache": request.use_sti_signature_cache,
        "debug": request.debug,
        "debug_preview_limit": request.debug_preview_limit,
        "llm": _build_llm_payload(request.llm, llm_api_key, llm_endpoint),
    }
    job_id = await _enqueue_job("cpa", payload)
    return JobQueuedResponse(job_id=job_id, status="queued")


@app.post("/schemas/{schema}/tabular/cpa", dependencies=[Depends(require_api_key)], tags=["Schemas"], response_model=JobQueuedResponse)
async def submit_schema_tabular_cpa(
    schema: str,
    request: SchemaCPARequest,
    llm_api_key: str | None = Security(llm_api_key_header),
    llm_endpoint: str | None = Header(default=None, alias="X-LLM-Endpoint"),
):
    _require_llm_overrides(request.llm, llm_api_key)
    _ensure_schema_supported(schema, require_cpa=True)

    payload = {
        "schema": schema,
        "table_id": request.table_id,
        "sampled_rows": request.sampled_rows,
        "subject_column": request.subject_column,
        "subject_class": request.subject_class,
        "target_columns": request.target_columns,
        "use_sti_signature_cache": request.use_sti_signature_cache,
        "debug": request.debug,
        "debug_preview_limit": request.debug_preview_limit,
        "llm": _build_llm_payload(request.llm, llm_api_key, llm_endpoint),
    }
    job_id = await _enqueue_job("cpa", payload)
    return JobQueuedResponse(job_id=job_id, status="queued")


# ------------------------
# Privacy analysis API
# ------------------------
@app.post("/privacy/analyze", dependencies=[Depends(require_api_key)], tags=["Privacy"], response_model=JobQueuedResponse)
async def submit_privacy_analyze(
    request: PrivacyAnalyzeRequest,
    llm_api_key: str | None = Security(llm_api_key_header),
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


@app.post("/schemas/ingest", dependencies=[Depends(require_api_key)], tags=["Schemas"], response_model=JobQueuedResponse)
async def submit_schema_ingest(
    request: SchemaIngestRequest,
    llm_api_key: str | None = Security(llm_api_key_header),
    llm_endpoint: str | None = Header(default=None, alias="X-LLM-Endpoint"),
):
    _validate_schema_ingest_request(request)

    llm_payload: dict[str, Any] | None = None
    if request.llm is not None:
        _require_llm_overrides(request.llm, llm_api_key)
        llm_payload = _build_llm_payload(request.llm, llm_api_key, llm_endpoint)
    elif request.use_llm_fallback:
        raise HTTPException(status_code=400, detail="llm is required when use_llm_fallback=true.")

    payload = _build_schema_ingest_payload(request)
    if llm_payload:
        payload["llm"] = llm_payload

    job_id = await _enqueue_job("schema_ingest", payload)
    return JobQueuedResponse(job_id=job_id, status="queued")


@app.post(
    "/schemas/ingest/preview",
    dependencies=[Depends(require_api_key)],
    tags=["Schemas"],
    response_model=SchemaIngestPreviewResponse,
)
async def preview_schema_ingest(
    request: SchemaIngestRequest,
    llm_api_key: str | None = Security(llm_api_key_header),
    llm_endpoint: str | None = Header(default=None, alias="X-LLM-Endpoint"),
):
    _validate_schema_ingest_request(request)

    llm_payload: dict[str, Any] | None = None
    if request.llm is not None:
        _require_llm_overrides(request.llm, llm_api_key)
        llm_payload = _build_llm_payload(request.llm, llm_api_key, llm_endpoint)
    elif request.use_llm_fallback:
        raise HTTPException(status_code=400, detail="llm is required when use_llm_fallback=true.")

    payload = _build_schema_ingest_payload(request)
    llm_client = None
    try:
        if llm_payload:
            settings = _settings_with_llm_payload(app.state.settings, llm_payload)
            llm_client = create_client(settings)
        preview = await preview_schema_payload(payload, llm_client=llm_client)
        return SchemaIngestPreviewResponse.model_validate(preview)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if llm_client is not None:
            await llm_client.close()


@app.patch("/schemas/{schema_name}", dependencies=[Depends(require_api_key)], tags=["Schemas"])
async def patch_user_schema(schema_name: str, request: SchemaPatchRequest) -> dict[str, Any]:
    updates = request.model_dump()
    if not any(value is not None for value in updates.values()):
        raise HTTPException(status_code=400, detail="At least one field must be provided.")
    try:
        return {"status": "ok", **update_user_schema_metadata(schema_name, updates)}
    except ValueError as exc:
        detail = str(exc)
        status = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code=status, detail=detail) from exc


@app.delete("/schemas/{schema_name}", dependencies=[Depends(require_api_key)], tags=["Schemas"])
async def delete_schema(schema_name: str, remove_files: bool = True) -> dict[str, Any]:
    try:
        result = delete_user_schema(schema_name, remove_files=remove_files)
        return {"status": "ok", **result}
    except ValueError as exc:
        detail = str(exc)
        status = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code=status, detail=detail) from exc


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


@app.get("/jobs/{job_id}/privacy-report", dependencies=[Depends(require_api_key)], tags=["Metadata"])
async def get_job_privacy_report(job_id: str) -> dict[str, Any]:
    job_store = app.state.job_store
    job = await job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Job is not completed (status={job.status}).")

    result = job.result if isinstance(job.result, dict) else {}
    reports = result.get("reports")
    if not isinstance(reports, dict):
        raise HTTPException(status_code=404, detail="No reports found for this job.")

    machine = reports.get("machine_readable")
    if not isinstance(machine, dict):
        raise HTTPException(status_code=404, detail="Machine-readable report is not available for this job.")

    content = machine.get("content")
    if not isinstance(content, dict):
        raise HTTPException(status_code=404, detail="Machine-readable report content is missing or invalid.")

    return content


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


@app.post("/schemas/reload", dependencies=[Depends(require_api_key)], tags=["Metadata"])
async def reload_schemas() -> dict[str, Any]:
    names = reload_schema_registry()
    return {"status": "ok", "schema_count": len(names), "schemas": names}


@app.get("/policy-packs", dependencies=[Depends(require_api_key)], tags=["Metadata"])
async def list_policy_packs_endpoint() -> dict[str, Any]:
    return {"policy_packs": list_policy_packs()}


@app.get("/privacy/profiles", dependencies=[Depends(require_api_key)], tags=["Metadata"])
async def get_privacy_profiles_endpoint() -> dict[str, Any]:
    """
    Return pipelines/privacy_profiles.json so frontends can show available profiles.
    """
    return load_privacy_profiles()


@app.get("/privacy/reports/schema", dependencies=[Depends(require_api_key)], tags=["Metadata"])
async def get_privacy_report_schema_endpoint() -> dict[str, Any]:
    return get_machine_report_json_schema()


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


def _load_user_schema_assets(data_dir: Path) -> list[dict[str, Any]]:
    registry_path = data_dir / "user_vocabularies.json"
    if not registry_path.exists():
        return []
    try:
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(raw, list):
        return []

    out: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()

        type_source = entry.get("type_source")
        type_count: int | None = None
        if isinstance(type_source, str) and type_source.strip():
            type_path = Path(type_source.strip())
            if not type_path.is_absolute():
                type_path = data_dir / type_path
            try:
                payload = json.loads(type_path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    type_count = len(payload)
            except Exception:  # noqa: BLE001
                type_count = None

        manifest_path = data_dir / "user" / name / "manifest.json"
        manifest: dict[str, Any] | None = None
        if manifest_path.exists():
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    manifest = loaded
            except Exception:  # noqa: BLE001
                manifest = None

        item: dict[str, Any] = {
            "name": name,
            "label": entry.get("label"),
            "description": entry.get("description"),
            "type_source": entry.get("type_source"),
            "score_mode": entry.get("score_mode"),
            "supports_text": bool(entry.get("supports_text", True)),
            "supports_table": bool(entry.get("supports_table", True)),
            "supports_cpa": bool(entry.get("supports_cpa", False)),
            "prefilter_types": bool(entry.get("prefilter_types", False)),
            "type_count": type_count,
            "manifest": str(manifest_path.relative_to(data_dir).as_posix()) if manifest_path.exists() else None,
        }
        if manifest:
            item["source"] = manifest.get("source")
            item["generated_at"] = manifest.get("generated_at")
            counts = manifest.get("counts")
            if isinstance(counts, dict) and isinstance(counts.get("types"), int):
                item["type_count"] = counts["types"]
        out.append(item)

    return sorted(out, key=lambda x: x["name"])


@app.get("/models", dependencies=[Depends(require_api_key)], tags=["Metadata"])
async def list_models(
    provider: Literal["ollama", "openrouter", "deepinfra", "deepseek", "all"] = "openrouter",
    llm_api_key: str | None = Security(llm_api_key_header),
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
    assets = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(assets, dict):
        raise HTTPException(status_code=500, detail=f"assets_index.json is invalid at {path}")

    assets.setdefault("registries", {})
    if isinstance(assets["registries"], dict):
        assets["registries"]["user_vocabularies"] = "user_vocabularies.json"

    assets.setdefault("assets", {})
    if isinstance(assets["assets"], dict):
        assets["assets"]["user_schemas"] = _load_user_schema_assets(DATA_DIR)

    return assets


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="Moose API Docs",
        swagger_favicon_url="/static/moose-logo.png",
        swagger_css_url="/static/docs.css",
    )
