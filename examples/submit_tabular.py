import os
import time

import httpx


def main():
    api_key = os.environ.get("MOOSE_API_KEY")
    if not api_key:
        raise RuntimeError("MOOSE_API_KEY is required")
    
    provider = os.environ.get("MOOSE_LLM_PROVIDER", "openrouter")
    if provider not in {"openrouter", "ollama", "deepinfra", "deepseek"}:
        raise RuntimeError("MOOSE_LLM_PROVIDER must be openrouter, ollama, deepinfra, or deepseek")
    
    llm_api_key = os.environ.get("LLM_API_KEY")
    llm_endpoint = os.environ.get("LLM_ENDPOINT")
    
    if provider in {"openrouter", "deepinfra", "deepseek"} and not llm_api_key:
        raise RuntimeError("LLM_API_KEY is required for this provider")
    
    # Set default model based on provider
    if provider == "deepseek":
        model = os.environ.get("MOOSE_MODEL", "deepseek-chat")
    elif provider == "deepinfra":
        model = os.environ.get("MOOSE_MODEL", "Qwen/Qwen3-Next-80B-A3B-Instruct")
    elif provider == "ollama":
        model = os.environ.get("MOOSE_MODEL", "llama3")
    else:
        model = os.environ.get("MOOSE_MODEL", "anthropic/claude-3.5-sonnet")

    payload = {
        "schema": "coarse",
        "tasks": [
            {
                "task_id": "table-1",
                "table_id": "employees",
                "sampled_rows": [
                    {"name": "Alice Smith", "email": "alice@example.com", "age": "29"},
                    {"name": "Bob Jones", "email": "bob@example.com", "age": "41"},
                ],
            }
        ],
        "llm": {"provider": provider, "model": model},
    }
    
    headers = {"X-API-Key": api_key}
    if llm_api_key:
        headers["X-LLM-API-Key"] = llm_api_key
    if llm_endpoint:
        headers["X-LLM-Endpoint"] = llm_endpoint
    
    with httpx.Client() as client:
        resp = client.post(
            "http://localhost:8000/tabular/annotate", json=payload, headers=headers
        )
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
        print("job_id:", job_id)

        while True:
            status = client.get(f"http://localhost:8000/jobs/{job_id}", headers=headers)
            status.raise_for_status()
            data = status.json()
            print("status:", data["status"])
            if data["status"] in {"completed", "failed"}:
                print(data)
                break
            time.sleep(1)


if __name__ == "__main__":
    main()