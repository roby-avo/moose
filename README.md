# Moose

![Moose](src/moose_api/static/moose-readme.png)

Moose is a prototype library and API for asynchronous, high-throughput NER and tabular semantic typing. Requests are queued and processed by workers; clients receive a `job_id` immediately and poll for results.

## Quickstart (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

export MOOSE_API_KEY=your_api_key
export MOOSE_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

uvicorn moose_api.main:app --host 0.0.0.0 --port 8000
```

## Quickstart (Docker)

```bash
docker compose up --build
```

If you want to set defaults for the container, copy `.env.example` to `.env` and adjust values (for example, `MOOSE_OPENROUTER_BASE_URL`).
To avoid port conflicts, set `MOOSE_API_PORT` and `MOOSE_DEMO_PORT`.

## Production (Docker Compose)

Use the production compose file to run the API with multiple workers, Mongo persistence, and health checks:

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

This production compose file also exposes the Streamlit demo. Configure ports via `MOOSE_API_PORT` and `MOOSE_DEMO_PORT` in `.env`.

### LLM endpoint configuration

Moose supports OpenRouter and Ollama. Configure base URLs in `.env`:

```bash
MOOSE_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MOOSE_OLLAMA_HOST=http://localhost:11434
```

Provide `llm.provider` and `llm.model` in each request payload, and pass provider credentials via the `X-LLM-API-Key` header.
When `provider=openrouter`, the header is required; for Ollama it is optional.
You can override the provider endpoint per request with `X-LLM-Endpoint`.

## API

All endpoints require `X-API-Key` or `Authorization: Bearer <key>`.
Swagger UI is available at `/docs`.

### DPV endpoints

For convenience (legacy), the API exposes DPV-specific endpoints that fix `schema` to `dpv`:

- `POST /dpv/ner`
- `POST /dpv/tabular/annotate`

Preferred: use the schema-specific endpoints below for any vocabulary.

### Schema endpoints

Use the schema name in the path to target any registered vocabulary:

- `POST /schemas/{schema}/ner`
- `POST /schemas/{schema}/tabular/annotate`

You can also keep using `POST /ner` and `POST /tabular/annotate` with `schema` in the body.

### Runtime LLM configuration (required)

Provide `llm.provider` and `llm.model` per request. Provider credentials are passed via the `X-LLM-API-Key` header.
Set `X-LLM-Endpoint` to override `MOOSE_OPENROUTER_BASE_URL` or `MOOSE_OLLAMA_HOST` for that request.

```json
{
  "schema": "coarse",
  "text": "Roberto founded Moose.",
  "llm": {
    "provider": "openrouter",
    "model": "anthropic/claude-3.5-sonnet"
  }
}
```

### Submit NER job

```bash
curl -s http://localhost:8000/ner \
  -H 'X-API-Key: your_api_key' \
  -H 'X-LLM-API-Key: your_llm_key' \
  -H 'X-LLM-Endpoint: https://your-llm-endpoint' \
  -H 'Content-Type: application/json' \
  -d '{
    "schema":"coarse",
    "text":"Roberto Avogadro founded Moose in 2024.",
    "llm":{
      "provider":"openrouter",
      "model":"anthropic/claude-3.5-sonnet"
    }
  }'
```

### Submit tabular typing job

```bash
curl -s http://localhost:8000/tabular/annotate \
  -H 'X-API-Key: your_api_key' \
  -H 'X-LLM-API-Key: your_llm_key' \
  -H 'X-LLM-Endpoint: https://your-llm-endpoint' \
  -H 'Content-Type: application/json' \
  -d '{
    "schema":"coarse",
    "table_id":"employees",
    "sampled_rows":[
      {"name":"Alice Smith","email":"alice@example.com","age":"29"},
      {"name":"Bob Jones","email":"bob@example.com","age":"41"}
    ],
    "llm":{
      "provider":"openrouter",
      "model":"anthropic/claude-3.5-sonnet"
    }
  }'
```

### Poll job status/results

```bash
curl -s http://localhost:8000/jobs/<job_id> \
  -H 'X-API-Key: your_api_key'
```

For single-resource endpoints, `result` is also single-resource:
- `/ner` -> `{"entities": [...], "warnings": [...]?}`
- `/tabular/annotate` -> `{"table_id": "...", "columns": [...], "warnings": [...]?}`
- `/tabular/ner` -> `{"table_id": "...", "rows": [...], "warnings": [...]?}`
- `/tabular/cpa` -> `{"table_id": "...", "subject_column": "...", "relationships": [...]}`

### List models

```bash
curl -s http://localhost:8000/models \
  -H 'X-API-Key: your_api_key' \
  -H 'X-LLM-API-Key: your_llm_key' \
  -H 'X-LLM-Endpoint: https://your-llm-endpoint'
```

`/models` returns provider models. For OpenRouter, `X-LLM-API-Key` is required. For Ollama, `X-LLM-API-Key` is optional.
You can query a single provider via `?provider=ollama` or `?provider=openrouter`, or use `?provider=all`.

## Schemas

- `coarse`: high-level NER types: PERSON, ORGANIZATION, LOCATION, EVENT, WORK, PRODUCT, CONCEPT, MISC.
- `fine`: detailed NER types (e.g., PERSON, COMPANY, CITY, LAW, DEVICE, etc.) with parent mapping; responses also include `coarse_type_id`.
- `sti`: column type classification inventory for STI (NE:* plus high-level LIT:* and a compact `xsd:*` set, with `ext:*` syntactic patterns like email/IP/phone/UUID; see `src/moose/data/sti_types.json`). Tabular-only; text NER endpoints reject it. For literal subtypes, responses include `coarse_type_id` with the matching LIT:STRING/NUMBER/DATETIME.
- `dpv`: full DPV vocabulary IDs (loaded from `src/moose/data/dpv_full.json` via the registry).

Custom vocabularies are configured in `src/moose/data/vocabularies.json`. Add a new entry
with a `name`, `type_source` (a JSON file in `src/moose/data`), and optional prompt and
score settings. The vocabulary JSON can be a list of string IDs or objects containing an
`id` field.

Example registry entry:

```json
{
  "name": "my-vocab",
  "label": "My Vocab",
  "type_source": "my_vocab.json",
  "score_mode": "sparse",
  "text_intro": "You are a My Vocab annotation engine.",
  "table_intro": "You are a My Vocab classification engine for tabular data."
}
```

For large vocabularies, `"prefilter_types": true` enables two-step type selection where implemented
(currently used in CPA flows). NER/text and table typing use a single prompt per request.

## Confidence

For each entity/column, the model returns unnormalized non-negative scores for the configured label set (including fallback labels where present), and Moose returns the selected type with confidence in `[0,1]`.

## Queue behavior

- Jobs are enqueued and processed asynchronously by workers.
- Backpressure is enforced when the queue size exceeds `MOOSE_QUEUE_MAXSIZE` (HTTP 429).
- If `MOOSE_MONGO_URL` is set and reachable, MongoDB is used for queue + job storage; otherwise Moose falls back to in-memory queues (not durable across restarts).

## Configuration

- `MOOSE_OPENROUTER_BASE_URL`: defaults to `https://openrouter.ai/api/v1`
- `MOOSE_OLLAMA_HOST`: Ollama host URL
- `MOOSE_MONGO_URL`: MongoDB connection string
- `MOOSE_MONGO_DB`: MongoDB database name
- `MOOSE_API_KEY`: required API key for all endpoints
- `MOOSE_WORKER_COUNT`: worker concurrency
- `MOOSE_QUEUE_MAXSIZE`: max queue length for backpressure
- `MOOSE_MAX_RETRIES`: retries for invalid LLM output

### Provider setup examples

```bash
export MOOSE_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
export MOOSE_OLLAMA_HOST=http://localhost:11434
```

## Examples

- `examples/submit_ner.py`
- `examples/submit_tabular.py`
