from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from moose.config import Settings
from moose.cpa import run_table_cpa
from moose.ingest import ingest_schema_payload
from moose.llm import LLMClient, create_client
from moose.ner import run_table_annotate, run_tabular_ner, run_text_ner
from moose.privacy import run_privacy_analyze


@dataclass
class JobRecord:
    job_id: str
    endpoint_type: str
    payload: dict
    status: str
    created_at: str
    updated_at: str
    retries: int
    result: dict | None = None
    error: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_SINGLE_TASK_ID = "__single__"
_SINGLE_TABLE_ID = "__single_table__"


def _strip_warning_ids(warnings: Any, keys: set[str]) -> list[Any]:
    if not isinstance(warnings, list):
        return []
    out: list[Any] = []
    for warning in warnings:
        if isinstance(warning, dict):
            out.append({k: v for k, v in warning.items() if k not in keys})
        else:
            out.append(warning)
    return out


def _unwrap_single_ner_result(result: dict[str, Any]) -> dict[str, Any]:
    item = (result.get("results") or [{}])[0]
    out: dict[str, Any] = {"entities": item.get("entities", [])}
    warnings = _strip_warning_ids(result.get("warnings"), {"task_id"})
    if warnings:
        out["warnings"] = warnings
    return out


def _unwrap_single_tabular_result(result: dict[str, Any]) -> dict[str, Any]:
    item = (result.get("results") or [{}])[0]
    out: dict[str, Any] = {"columns": item.get("columns", [])}
    table_id = item.get("table_id")
    if isinstance(table_id, str) and table_id and table_id != _SINGLE_TABLE_ID:
        out["table_id"] = table_id
    warnings = _strip_warning_ids(result.get("warnings"), {"task_id"})
    if warnings:
        out["warnings"] = warnings
    return out


def _unwrap_single_tabular_ner_result(result: dict[str, Any]) -> dict[str, Any]:
    item = (result.get("results") or [{}])[0]
    out: dict[str, Any] = {"rows": item.get("rows", [])}
    table_id = item.get("table_id")
    if isinstance(table_id, str) and table_id and table_id != _SINGLE_TABLE_ID:
        out["table_id"] = table_id
    warnings = _strip_warning_ids(result.get("warnings"), {"task_id", "table_task_id"})
    if warnings:
        out["warnings"] = warnings
    return out


def _unwrap_single_cpa_result(result: dict[str, Any]) -> dict[str, Any]:
    item = (result.get("results") or [{}])[0]
    out: dict[str, Any] = {
        "subject_column": item.get("subject_column"),
        "relationships": item.get("relationships", []),
    }
    table_id = item.get("table_id")
    if isinstance(table_id, str) and table_id and table_id != _SINGLE_TABLE_ID:
        out["table_id"] = table_id
    if isinstance(item.get("debug"), dict):
        out["debug"] = item["debug"]
    warnings = _strip_warning_ids(result.get("warnings"), {"task_id"})
    if warnings:
        out["warnings"] = warnings
    return out


class JobStore:
    async def get_job(self, job_id: str) -> JobRecord | None:
        raise NotImplementedError

    async def put_job(self, job: JobRecord) -> None:
        raise NotImplementedError

    async def update_job(self, job_id: str, **fields: Any) -> None:
        raise NotImplementedError


class InMemoryJobStore(JobStore):
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()

    async def get_job(self, job_id: str) -> JobRecord | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def put_job(self, job: JobRecord) -> None:
        async with self._lock:
            self._jobs[job.job_id] = job

    async def update_job(self, job_id: str, **fields: Any) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for key, value in fields.items():
                setattr(job, key, value)
            self._jobs[job_id] = job


class MongoJobStore(JobStore):
    def __init__(self, db) -> None:
        self._jobs = db["jobs"]

    async def ensure_indexes(self) -> None:
        await self._jobs.create_index("status")
        await self._jobs.create_index("created_at")
        await self._jobs.create_index("job_id", unique=True)

    def _doc_to_record(self, doc: dict) -> JobRecord:
        if not doc:
            raise ValueError("Empty job document")
        job_id = doc.get("job_id") or doc.get("_id")
        return JobRecord(
            job_id=job_id,
            endpoint_type=doc["endpoint_type"],
            payload=doc["payload"],
            status=doc["status"],
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
            retries=doc.get("retries", 0),
            result=doc.get("result"),
            error=doc.get("error"),
        )

    async def get_job(self, job_id: str) -> JobRecord | None:
        doc = await self._jobs.find_one({"_id": job_id})
        if not doc:
            return None
        return self._doc_to_record(doc)

    async def put_job(self, job: JobRecord) -> None:
        doc = {"_id": job.job_id, **job.__dict__}
        await self._jobs.replace_one({"_id": job.job_id}, doc, upsert=True)

    async def update_job(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        await self._jobs.update_one({"_id": job_id}, {"$set": fields})


class QueueBackend:
    async def enqueue(self, job_id: str) -> None:
        raise NotImplementedError

    async def dequeue(self) -> str:
        raise NotImplementedError

    async def size(self) -> int:
        raise NotImplementedError

    def set_shutdown_event(self, event: asyncio.Event) -> None:
        return None


class InMemoryQueue(QueueBackend):
    def __init__(self, maxsize: int) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=maxsize)

    async def enqueue(self, job_id: str) -> None:
        self._queue.put_nowait(job_id)

    async def dequeue(self) -> str:
        return await self._queue.get()

    async def size(self) -> int:
        return self._queue.qsize()


class MongoQueue(QueueBackend):
    def __init__(self, db, poll_interval: float = 0.2) -> None:
        self._queue = db["queue"]
        self._poll_interval = poll_interval
        self._shutdown: asyncio.Event | None = None

    async def ensure_indexes(self) -> None:
        await self._queue.create_index([("created_at", 1)])

    def set_shutdown_event(self, event: asyncio.Event) -> None:
        self._shutdown = event

    async def enqueue(self, job_id: str) -> None:
        from pymongo.errors import DuplicateKeyError

        doc = {"_id": job_id, "job_id": job_id, "created_at": utc_now()}
        try:
            await self._queue.insert_one(doc)
        except DuplicateKeyError:
            return None

    async def dequeue(self) -> str:
        while True:
            if self._shutdown and self._shutdown.is_set():
                raise asyncio.CancelledError
            doc = await self._queue.find_one_and_delete({}, sort=[("created_at", 1)])
            if doc:
                return doc["job_id"]
            await asyncio.sleep(self._poll_interval)

    async def size(self) -> int:
        return int(await self._queue.count_documents({}))


def _settings_with_overrides(settings: Settings, overrides: dict) -> Settings:
    update: dict[str, Any] = {}
    provider = overrides.get("provider", settings.MOOSE_LLM_PROVIDER).lower()

    if overrides.get("provider"):
        update["MOOSE_LLM_PROVIDER"] = overrides["provider"]
    if overrides.get("model"):
        update["MOOSE_MODEL"] = overrides["model"]

    # Credentials
    if overrides.get("ollama_token"):
        update["MOOSE_OLLAMA_TOKEN"] = overrides["ollama_token"]
    if overrides.get("openrouter_api_key"):
        update["MOOSE_OPENROUTER_API_KEY"] = overrides["openrouter_api_key"]
    if overrides.get("deepinfra_api_key"):
        update["MOOSE_DEEPINFRA_API_KEY"] = overrides["deepinfra_api_key"]
    if overrides.get("deepseek_api_key"):
        update["MOOSE_DEEPSEEK_API_KEY"] = overrides["deepseek_api_key"]

    # Endpoint override
    if overrides.get("endpoint"):
        if provider == "openrouter":
            update["MOOSE_OPENROUTER_BASE_URL"] = overrides["endpoint"]
        elif provider == "ollama":
            update["MOOSE_OLLAMA_HOST"] = overrides["endpoint"]
        elif provider == "deepinfra":
            update["MOOSE_DEEPINFRA_BASE_URL"] = overrides["endpoint"]
        elif provider == "deepseek":
            update["MOOSE_DEEPSEEK_BASE_URL"] = overrides["endpoint"]

    if not update:
        return settings
    return settings.model_copy(update=update)


class WorkerPool:
    def __init__(
        self,
        queue: QueueBackend,
        store: JobStore,
        llm_client: LLMClient | None,
        settings: Settings,
    ) -> None:
        self._queue = queue
        self._store = store
        self._llm_client = llm_client
        self._settings = settings
        self._tasks: list[asyncio.Task] = []
        self._shutdown = asyncio.Event()
        self._queue.set_shutdown_event(self._shutdown)

    async def start(self) -> None:
        for idx in range(self._settings.MOOSE_WORKER_COUNT):
            task = asyncio.create_task(self._worker_loop(idx))
            self._tasks.append(task)

    async def stop(self) -> None:
        self._shutdown.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _worker_loop(self, worker_id: int) -> None:
        while not self._shutdown.is_set():
            job_id = await self._queue.dequeue()
            job = await self._store.get_job(job_id)
            if not job:
                continue

            await self._store.update_job(job_id, status="processing", updated_at=utc_now())

            llm_client = self._llm_client
            owns_client = False

            try:
                overrides = job.payload.get("llm") or {}
                if overrides:
                    job_settings = _settings_with_overrides(self._settings, overrides)
                    llm_client = create_client(job_settings)
                    owns_client = True

                if job.endpoint_type == "ner":
                    if llm_client is None:
                        raise RuntimeError(
                            "LLM client not configured. Provide per-request llm overrides with credentials "
                            "(X-LLM-API-Key) or configure default provider credentials in env."
                        )
                    single_task = {"task_id": _SINGLE_TASK_ID, "text": job.payload["text"]}
                    raw = await run_text_ner(
                        [single_task],
                        job.payload["schema"],
                        llm_client,
                        settings=self._settings,
                    )
                    result = _unwrap_single_ner_result(raw)

                elif job.endpoint_type == "tabular":
                    if llm_client is None:
                        raise RuntimeError(
                            "LLM client not configured. Provide per-request llm overrides with credentials "
                            "(X-LLM-API-Key) or configure default provider credentials in env."
                        )
                    single_task = {
                        "task_id": _SINGLE_TASK_ID,
                        "table_id": job.payload.get("table_id") or _SINGLE_TABLE_ID,
                        "sampled_rows": job.payload["sampled_rows"],
                    }
                    raw = await run_table_annotate(
                        [single_task],
                        job.payload["schema"],
                        llm_client,
                        settings=self._settings,
                    )
                    result = _unwrap_single_tabular_result(raw)

                elif job.endpoint_type == "tabular_ner":
                    if llm_client is None:
                        raise RuntimeError(
                            "LLM client not configured. Provide per-request llm overrides with credentials "
                            "(X-LLM-API-Key) or configure default provider credentials in env."
                        )
                    single_task = {
                        "task_id": _SINGLE_TASK_ID,
                        "table_id": job.payload.get("table_id") or _SINGLE_TABLE_ID,
                        "sampled_rows": job.payload["sampled_rows"],
                        "target_columns": job.payload["target_columns"],
                        "strings_only": job.payload.get("strings_only", True),
                        "skip_structured_literals": job.payload.get("skip_structured_literals", True),
                    }
                    raw = await run_tabular_ner(
                        [single_task],
                        job.payload["schema"],
                        llm_client,
                        settings=self._settings,
                    )
                    result = _unwrap_single_tabular_ner_result(raw)

                elif job.endpoint_type == "privacy_analyze":
                    if llm_client is None:
                        raise RuntimeError(
                            "LLM client not configured. Provide per-request llm overrides with credentials "
                            "(X-LLM-API-Key) or configure default provider credentials in env."
                        )
                    result = await run_privacy_analyze(
                        tasks=job.payload["tasks"],
                        policy_pack=job.payload.get("policy_pack"),
                        profile=job.payload.get("profile"),
                        analysis_mode=job.payload.get("analysis_mode"),
                        text_schema=job.payload.get("text_schema"),
                        table_schema=job.payload.get("table_schema"),
                        scan_schema=job.payload.get("scan_schema"),
                        include_extraction=job.payload.get("include_extraction"),
                        llm_client=llm_client,
                        settings=self._settings,
                    )

                elif job.endpoint_type == "cpa":
                    if llm_client is None:
                        raise RuntimeError(
                            "LLM client not configured. Provide per-request llm overrides with credentials "
                            "(X-LLM-API-Key) or configure default provider credentials in env."
                        )
                    single_task = {
                        "task_id": _SINGLE_TASK_ID,
                        "table_id": job.payload.get("table_id") or _SINGLE_TABLE_ID,
                        "sampled_rows": job.payload["sampled_rows"],
                        "subject_column": job.payload["subject_column"],
                        "subject_class": job.payload.get("subject_class"),
                        "target_columns": job.payload.get("target_columns"),
                        "use_sti_signature_cache": job.payload.get("use_sti_signature_cache", True),
                        "debug": job.payload.get("debug", False),
                        "debug_preview_limit": job.payload.get("debug_preview_limit", 20),
                    }
                    raw = await run_table_cpa(
                        [single_task],
                        job.payload["schema"],
                        llm_client,
                        settings=self._settings,
                    )
                    result = _unwrap_single_cpa_result(raw)

                elif job.endpoint_type == "schema_ingest":
                    result = await ingest_schema_payload(job.payload, llm_client=llm_client)

                else:
                    raise ValueError(f"Unknown endpoint_type: {job.endpoint_type}")

                await self._store.update_job(
                    job_id,
                    status="completed",
                    updated_at=utc_now(),
                    result=result,
                )

            except Exception as exc:  # noqa: BLE001
                await self._store.update_job(
                    job_id,
                    status="failed",
                    updated_at=utc_now(),
                    error=str(exc),
                )

            finally:
                if owns_client and llm_client is not None:
                    await llm_client.close()


async def build_backends(settings: Settings) -> tuple[JobStore, QueueBackend, dict]:
    info: dict[str, Any] = {"queue_backend": "memory"}
    if settings.MOOSE_MONGO_URL:
        try:
            import motor.motor_asyncio as motor
        except ImportError:
            return InMemoryJobStore(), InMemoryQueue(settings.MOOSE_QUEUE_MAXSIZE), info

        timeout_ms = max(1, int(settings.MOOSE_MONGO_TIMEOUT_SECS * 1000))
        client = motor.AsyncIOMotorClient(
            settings.MOOSE_MONGO_URL,
            serverSelectionTimeoutMS=timeout_ms,
            connectTimeoutMS=timeout_ms,
        )
        try:
            await client.admin.command("ping")
            db = client[settings.MOOSE_MONGO_DB]
            store = MongoJobStore(db)
            queue = MongoQueue(db)
            await store.ensure_indexes()
            await queue.ensure_indexes()
            info["queue_backend"] = "mongo"
            return store, queue, {"mongo_client": client, **info}
        except Exception:  # noqa: BLE001
            client.close()

    return InMemoryJobStore(), InMemoryQueue(settings.MOOSE_QUEUE_MAXSIZE), info
