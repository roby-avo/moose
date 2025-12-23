import time

import httpx


def main():
    payload = {
        "schema": "coarse",
        "tasks": [
            {"task_id": "t1", "text": "Roberto Avogadro founded Moose in 2024."},
            {"task_id": "t2", "text": "OpenAI held a conference in San Francisco."},
        ],
    }
    with httpx.Client() as client:
        resp = client.post("http://localhost:8000/v1/ner", json=payload)
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
        print("job_id:", job_id)

        while True:
            status = client.get(f"http://localhost:8000/v1/jobs/{job_id}")
            status.raise_for_status()
            data = status.json()
            print("status:", data["status"])
            if data["status"] in {"completed", "failed"}:
                print(data)
                break
            time.sleep(1)


if __name__ == "__main__":
    main()
