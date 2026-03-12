# Moose

![Moose](src/moose_api/static/moose-readme.png)

Moose is an async API + Streamlit demo for:
- text annotation (NER),
- table column typing,
- BYO schema ingestion,
- privacy analysis with human/machine reports.

Use this README as a quick usage guide. For deep technical details, see the docs linked at the end.

## Quick Start

### Option A: Docker (recommended)

```bash
docker compose up -d --build
```

Default local endpoints:
- API: `http://localhost:8000`
- Frontend (Streamlit): `http://localhost:8501`

### Option B: Local API only

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

export MOOSE_API_KEY=test-dev-key
uvicorn moose_api.main:app --host 0.0.0.0 --port 8000
```

## Auth Headers

All API calls require:
- `X-API-Key: <your-key>`

LLM-backed operations also require:
- `X-LLM-API-Key: <provider-key>` (required for OpenRouter/DeepInfra/DeepSeek; optional for Ollama)
- optional `X-LLM-Endpoint: <override-url>`

Default provider in this project is typically **OpenRouter**, but you can use any supported provider per request via:
- `llm.provider`: `openrouter | ollama | deepinfra | deepseek`
- `llm.model`: provider-specific model name

## Core Usage (curl)

### 1) List available schemas

```bash
curl -sS "http://localhost:8000/schemas?include_type_count=true" \
  -H "X-API-Key: test-dev-key"
```

### 2) Text annotation (schema-specific)

```bash
curl -sS -X POST "http://localhost:8000/schemas/coarse/ner" \
  -H "X-API-Key: test-dev-key" \
  -H "X-LLM-API-Key: <llm-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Maria Rossi works at Acme in Milan.",
    "llm": {"provider": "openrouter", "model": "anthropic/claude-3.5-sonnet"}
  }'
```

### 3) Table annotation (schema-specific)

```bash
curl -sS -X POST "http://localhost:8000/schemas/sti/tabular/annotate" \
  -H "X-API-Key: test-dev-key" \
  -H "X-LLM-API-Key: <llm-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "table_id":"patients",
    "sampled_rows":[
      {"patient_name":"Alice Smith","email":"alice@example.com","medical_record_number":"MRN-100245"},
      {"patient_name":"Bob Jones","email":"bob@example.com","medical_record_number":"MRN-100246"}
    ],
    "llm": {"provider": "openrouter", "model": "anthropic/claude-3.5-sonnet"}
  }'
```

### 4) BYO schema preview (dry run)

```bash
curl -sS -X POST "http://localhost:8000/schemas/ingest/preview" \
  -H "X-API-Key: test-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name":"healthcare_pii",
    "label":"Healthcare PII",
    "source_type":"file",
    "source_path":"examples/schema_ingest_samples/schemas/healthcare_pii.json",
    "supports_text":true,
    "supports_table":true,
    "score_mode":"sparse"
  }'
```

### 5) BYO schema ingest (activate)

```bash
curl -sS -X POST "http://localhost:8000/schemas/ingest" \
  -H "X-API-Key: test-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name":"healthcare_pii",
    "label":"Healthcare PII",
    "source_type":"file",
    "source_path":"examples/schema_ingest_samples/schemas/healthcare_pii.json",
    "supports_text":true,
    "supports_table":true,
    "score_mode":"sparse"
  }'
```

### 6) Privacy analysis

```bash
curl -sS -X POST "http://localhost:8000/privacy/analyze" \
  -H "X-API-Key: test-dev-key" \
  -H "X-LLM-API-Key: <llm-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "profile":"balanced",
    "tasks":[
      {
        "kind":"text",
        "task_id":"t1",
        "text":"We collect patient names and emails and share with billing processor."
      }
    ],
    "llm": {"provider": "openrouter", "model": "anthropic/claude-3.5-sonnet"}
  }'
```

### Provider variants (same payload shape)

Use the same endpoints and payload structure; only change `llm.provider` and `llm.model`.

- OpenRouter (default): `{\"provider\":\"openrouter\",\"model\":\"anthropic/claude-3.5-sonnet\"}`
- Ollama: `{\"provider\":\"ollama\",\"model\":\"llama3.1:8b\"}`
- DeepInfra: `{\"provider\":\"deepinfra\",\"model\":\"Qwen/Qwen3-Next-80B-A3B-Instruct\"}`
- DeepSeek: `{\"provider\":\"deepseek\",\"model\":\"deepseek-chat\"}`

### 7) Poll a job

```bash
curl -sS "http://localhost:8000/jobs/<job_id>" \
  -H "X-API-Key: test-dev-key"
```

### 8) Get machine-readable privacy report from a completed privacy job

```bash
curl -sS "http://localhost:8000/jobs/<job_id>/privacy-report" \
  -H "X-API-Key: test-dev-key"
```

## Frontend (Streamlit)

If running with Docker Compose, open:
- `http://localhost:8501`

The demo UI supports:
- text annotation,
- table annotation,
- BYO schema ingestion,
- privacy analysis,
- report downloads (human-readable `.md` and machine-readable `.json`).

## Detailed Documentation

- System blueprint: [docs/BLUEPRINT.md](docs/BLUEPRINT.md)
- Developer deep dive: [docs/DEVELOPER_DEEP_DIVE.md](docs/DEVELOPER_DEEP_DIVE.md)
- Privacy operation details: [docs/PRIVACY_OPERATION.md](docs/PRIVACY_OPERATION.md)
- BYO schema design + guardrails: [docs/BYO_SCHEMA_DESIGN.md](docs/BYO_SCHEMA_DESIGN.md)
