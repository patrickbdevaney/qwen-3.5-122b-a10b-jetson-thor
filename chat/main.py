"""
client/main.py

FastAPI chat proxy — run on your laptop to talk to the vLLM server on Thor.

Usage:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 7860 --reload

Then open: http://localhost:7860

Connects to Thor at: http://100.72.56.36:8000 || locahlhost:8000 on thor || http://<your_ip>:8000 if running vLLM locally
(Update THOR_BASE_URL below if Thor's IP changes)
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI
import asyncio
import json
import httpx
from datetime import datetime
from typing import AsyncGenerator

# ── Configuration ─────────────────────────────────────────────────────────────
THOR_BASE_URL = "http://100.72.56.36:8000/v1"
MODEL_NAME = "/model"  # vLLM serves the model under its mount path

app = FastAPI(title="Thor Chat — Qwen3.5-122B")
templates = Jinja2Templates(directory="templates")

client = AsyncOpenAI(
    base_url=THOR_BASE_URL,
    api_key="not-needed",  # vLLM doesn't require auth by default
    timeout=300.0,  # Long timeout — TTFT can be 10-20s
)

# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/api/status")
async def status():
    """Check if Thor's vLLM server is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            r = await http.get(f"http://100.72.56.36:8000/health")
            return {"status": "online", "code": r.status_code}
    except Exception as e:
        return {"status": "offline", "error": str(e)}


@app.get("/api/metrics")
async def metrics():
    """Pull live metrics from Thor's Prometheus endpoint."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            r = await http.get(f"http://100.72.56.36:8000/metrics")
            text = r.text

        def extract(name: str) -> str:
            for line in text.splitlines():
                if line.startswith(name + "{") or line.startswith(name + " "):
                    parts = line.rsplit(" ", 1)
                    if len(parts) == 2:
                        try:
                            val = float(parts[1])
                            return f"{val:.1f}"
                        except ValueError:
                            pass
            return "—"

        return {
            "requests_running": extract("vllm:num_requests_running"),
            "prompt_tokens": extract("vllm:prompt_tokens_total"),
            "gen_tokens": extract("vllm:generation_tokens_total"),
            "gpu_cache_usage": extract("vllm:gpu_cache_usage_perc"),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/chat")
async def chat(request: Request):
    """
    Streaming chat endpoint. Proxies to Thor's vLLM /v1/chat/completions.
    Returns a Server-Sent Events stream.
    """
    body = await request.json()
    messages = body.get("messages", [])
    system_prompt = body.get("system_prompt", "")
    temperature = float(body.get("temperature", 0.6))
    max_tokens = int(body.get("max_tokens", 4096))
    enable_thinking = body.get("enable_thinking", True)

    # Build message list
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    # Qwen3.5 thinking mode control
    # /think suffix enables extended reasoning; /no_think disables it
    if full_messages and enable_thinking:
        last = full_messages[-1]
        if last["role"] == "user" and not last["content"].endswith("/think"):
            last = dict(last)
            last["content"] = last["content"] + " /think"
            full_messages[-1] = last

    async def generate() -> AsyncGenerator[str, None]:
        start = datetime.now()
        first_token = None
        token_count = 0
        full_response = ""
        in_think_block = False
        think_buffer = ""

        try:
            stream = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=full_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue

                if first_token is None:
                    first_token = (datetime.now() - start).total_seconds()

                token_count += 1
                full_response += delta

                # Detect and handle <think> blocks
                think_start = "<think>"
                think_end = "</think>"

                if think_start in delta or in_think_block:
                    think_buffer += delta
                    if think_start in think_buffer and not in_think_block:
                        in_think_block = True
                    if think_end in think_buffer and in_think_block:
                        in_think_block = False
                        # Emit the complete think block as a special event
                        think_content = think_buffer[
                            think_buffer.find(think_start)
                            + len(think_start) : think_buffer.find(think_end)
                        ]
                        yield f"data: {json.dumps({'type': 'think', 'content': think_content})}\n\n"
                        # Emit any text after </think>
                        after = think_buffer[
                            think_buffer.find(think_end) + len(think_end) :
                        ]
                        think_buffer = ""
                        if after.strip():
                            yield f"data: {json.dumps({'type': 'token', 'content': after})}\n\n"
                    continue

                yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"

            # Final stats
            elapsed = (datetime.now() - start).total_seconds()
            tps = token_count / elapsed if elapsed > 0 else 0
            yield f"data: {json.dumps({'type': 'done', 'stats': {'ttft': round(first_token or 0, 2), 'tps': round(tps, 1), 'tokens': token_count, 'elapsed': round(elapsed, 1)}})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
