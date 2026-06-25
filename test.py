%%writefile /kaggle/working/main.py

# main.py
# Kaggle Benchmarks -> OpenAI-compatible API (Fixed & Robust Version)
# Run with: !python main.py

import re
import time
import json
import uuid
import asyncio
import logging
from typing import Any, Dict, List, Optional, Literal, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

import kaggle_benchmarks as kbench


# ==========================================================
# CONFIG
# ==========================================================

MODEL_ID = "openai/gpt-5.4-nano-2026-03-17"
PORT = 9191

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("\nAvailable LLMs:")
print(list(kbench.llms.keys()))

app = FastAPI()


# ==========================================================
# LOAD MODEL
# ==========================================================

try:
    llm = kbench.llms[MODEL_ID]

    logger.info("Loaded model: %s", getattr(llm, "name", MODEL_ID))

    test = llm.prompt("Reply with OK")
    logger.info("Startup test successful: %s", str(test)[:100])

except Exception:
    logger.exception("Failed loading model")
    llm = None


# ==========================================================
# SCHEMAS
# ==========================================================

class FunctionCall(BaseModel):
    name: str
    arguments: str = "{}"


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunction


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, List[Any]]] = None

    # For tool-result messages
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    # For assistant messages that previously requested tool calls
    tool_calls: Optional[List[ToolCall]] = None


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]

    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = True

    # OpenAI-compatible tool fields
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None


# ==========================================================
# HELPERS
# ==========================================================

def now_ts() -> int:
    return int(time.time())


def make_chatcmpl_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def make_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def normalize_content(content: Optional[Union[str, List[Any]]]) -> str:
    """
    Convert content into plain text for the backend prompt.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    # Basic handling for multimodal/list-style content.
    # This keeps your proxy from crashing if a client sends content parts.
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        else:
            parts.append(str(item))

    return "\n".join(parts)


def extract_text(response: Any) -> str:
    """
    Kaggle models may return:
    - str
    - object.content
    - object.text
    - object.message.content

    Normalize all of them into a string.
    """

    if response is None:
        return ""

    if isinstance(response, str):
        return response

    if hasattr(response, "content"):
        value = response.content

        if isinstance(value, list):
            return "".join(str(x) for x in value)

        return str(value)

    if hasattr(response, "text"):
        return str(response.text)

    if hasattr(response, "message"):
        if hasattr(response.message, "content"):
            return str(response.message.content)

    return str(response)


def safe_json_loads(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


def strip_markdown_json_fence(text: str) -> str:
    """
    Converts:
        ```json
        {...}
        ```
    into:
        {...}
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    return text.strip()


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Tries to recover a JSON object from model output.

    This helps if the model emits extra text around the JSON despite instructions.
    """
    text = strip_markdown_json_fence(text)

    parsed = safe_json_loads(text)
    if isinstance(parsed, dict):
        return parsed

    # Try extracting first {...} block.
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        parsed = safe_json_loads(candidate)
        if isinstance(parsed, dict):
            return parsed

    return None


def normalize_tool_call(raw: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """
    Normalize different possible model JSON outputs into OpenAI-style tool_call.
    """

    tool_call_id = raw.get("id") or make_tool_call_id()
    tool_type = raw.get("type") or "function"

    function = raw.get("function") or {}

    # Allow simplified format:
    # {"name": "...", "arguments": {...}}
    name = function.get("name") or raw.get("name")
    arguments = function.get("arguments", raw.get("arguments", "{}"))

    if not name:
        name = f"tool_{index}"

    if isinstance(arguments, dict) or isinstance(arguments, list):
        arguments = json.dumps(arguments, ensure_ascii=False)

    if arguments is None:
        arguments = "{}"

    if not isinstance(arguments, str):
        arguments = str(arguments)

    return {
        "id": tool_call_id,
        "type": tool_type,
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def parse_model_output(raw_text: str) -> Dict[str, Any]:
    """
    Expected model adapter output:

    Normal message:
    {
      "type": "message",
      "content": "Hello"
    }

    Tool call:
    {
      "type": "tool_calls",
      "tool_calls": [
        {
          "id": "call_abc",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"city\":\"Toronto\"}"
          }
        }
      ]
    }

    Also accepts simplified tool-call shape:
    {
      "tool_calls": [
        {
          "name": "get_weather",
          "arguments": {"city": "Toronto"}
        }
      ]
    }
    """

    raw_text = raw_text or ""
    obj = extract_first_json_object(raw_text)

    if not obj:
        return {
            "type": "message",
            "content": raw_text,
        }

    output_type = obj.get("type")

    if output_type == "tool_calls" or "tool_calls" in obj:
        raw_tool_calls = obj.get("tool_calls") or []

        if not isinstance(raw_tool_calls, list):
            raw_tool_calls = [raw_tool_calls]

        tool_calls = [
            normalize_tool_call(tc, index=i)
            for i, tc in enumerate(raw_tool_calls)
            if isinstance(tc, dict)
        ]

        if tool_calls:
            return {
                "type": "tool_calls",
                "tool_calls": tool_calls,
            }

    if output_type == "message":
        return {
            "type": "message",
            "content": str(obj.get("content", "")),
        }

    # If the model returned arbitrary JSON, treat it as a normal message.
    return {
        "type": "message",
        "content": raw_text,
    }


def format_tools_for_prompt(tools: Optional[List[Tool]]) -> str:
    if not tools:
        return ""

    tool_dicts = [tool.model_dump() for tool in tools]
    return json.dumps(tool_dicts, ensure_ascii=False, indent=2)


def build_prompt(request: ChatRequest) -> str:
    """
    Converts OpenAI chat messages into a prompt suitable for a text-only backend.

    When tools are present, the model is instructed to return a strict JSON
    envelope. The proxy then converts that JSON envelope into OpenAI-compatible
    tool_calls.
    """

    parts: List[str] = []

    for msg in request.messages:
        content = normalize_content(msg.content)

        if msg.role == "system":
            parts.append(f"System:\n{content}\n")

        elif msg.role == "user":
            parts.append(f"User:\n{content}\n")

        elif msg.role == "assistant":
            if msg.tool_calls:
                tool_calls_json = json.dumps(
                    [tc.model_dump() for tc in msg.tool_calls],
                    ensure_ascii=False,
                    indent=2,
                )
                parts.append(f"Assistant tool calls:\n{tool_calls_json}\n")

                if content:
                    parts.append(f"Assistant content:\n{content}\n")
            else:
                parts.append(f"Assistant:\n{content}\n")

        elif msg.role == "tool":
            parts.append(
                "Tool result:\n"
                f"name: {msg.name or ''}\n"
                f"tool_call_id: {msg.tool_call_id or ''}\n"
                f"content:\n{content}\n"
            )

    if request.tools:
        tools_json = format_tools_for_prompt(request.tools)

        tool_choice_text = ""
        if request.tool_choice is not None:
            tool_choice_text = json.dumps(request.tool_choice, ensure_ascii=False)

        parts.append(
            "Available tools:\n"
            f"{tools_json}\n\n"
            "Tool choice instruction:\n"
            f"{tool_choice_text or 'auto'}\n\n"
            "You are behind an OpenAI-compatible proxy.\n"
            "If the user request requires one or more tools, respond ONLY with valid JSON.\n"
            "Do not include markdown fences, comments, or explanatory text.\n\n"
            "For a tool call, use exactly this shape:\n"
            "{\n"
            '  "type": "tool_calls",\n'
            '  "tool_calls": [\n'
            "    {\n"
            '      "id": "call_unique_id",\n'
            '      "type": "function",\n'
            '      "function": {\n'
            '        "name": "tool_name",\n'
            '        "arguments": "{\\"arg\\": \\"value\\"}"\n'
            "      }\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "The function.arguments value MUST be a JSON string, not a JSON object.\n\n"
            "If no tool is required, respond ONLY with valid JSON in this shape:\n"
            "{\n"
            '  "type": "message",\n'
            '  "content": "your answer here"\n'
            "}\n"
        )

    else:
        parts.append(
            "Respond as the assistant. Do not fabricate tool calls because no tools are available.\n"
        )

    parts.append("Assistant:\n")

    return "\n".join(parts)


def chunk_string(text: str, size: int = 64):
    """
    Chunk by character count instead of whitespace.
    This is safer for JSON/tool arguments than regex word splitting.
    """
    text = text or ""

    for i in range(0, len(text), size):
        yield text[i:i + size]


def sse_payload(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def make_stream_chunk(
    stream_id: str,
    created: int,
    model: str,
    delta: Dict[str, Any],
    finish_reason: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def build_nonstream_response(
    request: ChatRequest,
    parsed: Dict[str, Any],
    prompt: str,
) -> Dict[str, Any]:
    completion_id = make_chatcmpl_id()
    created = now_ts()

    prompt_tokens = len(prompt.split())

    if parsed.get("type") == "tool_calls":
        tool_calls = parsed.get("tool_calls", [])

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 0,
                "total_tokens": prompt_tokens,
            },
        }

    content = str(parsed.get("content", ""))
    completion_tokens = len(content.split())

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def stream_openai_chunks(
    request: ChatRequest,
    parsed: Dict[str, Any],
):
    """
    Stream OpenAI-compatible chat.completion.chunk events.

    For normal content:
      delta.content

    For tool calls:
      delta.tool_calls[index].function.arguments

    Final chunk:
      delta={}
      finish_reason="stop" or "tool_calls"
    """

    stream_id = make_chatcmpl_id()
    created = now_ts()

    # First role chunk
    yield sse_payload(
        make_stream_chunk(
            stream_id=stream_id,
            created=created,
            model=request.model,
            delta={"role": "assistant"},
            finish_reason=None,
        )
    )

    if parsed.get("type") == "tool_calls":
        tool_calls = parsed.get("tool_calls", [])

        for tc_index, tc in enumerate(tool_calls):
            function = tc.get("function", {})
            function_name = function.get("name", "")
            arguments = function.get("arguments", "{}")

            # Send initial tool-call metadata.
            yield sse_payload(
                make_stream_chunk(
                    stream_id=stream_id,
                    created=created,
                    model=request.model,
                    delta={
                        "tool_calls": [
                            {
                                "index": tc_index,
                                "id": tc.get("id") or make_tool_call_id(),
                                "type": tc.get("type", "function"),
                                "function": {
                                    "name": function_name,
                                    "arguments": "",
                                },
                            }
                        ]
                    },
                    finish_reason=None,
                )
            )

            # Stream tool arguments incrementally.
            for piece in chunk_string(arguments, size=64):
                yield sse_payload(
                    make_stream_chunk(
                        stream_id=stream_id,
                        created=created,
                        model=request.model,
                        delta={
                            "tool_calls": [
                                {
                                    "index": tc_index,
                                    "function": {
                                        "arguments": piece,
                                    },
                                }
                            ]
                        },
                        finish_reason=None,
                    )
                )
                await asyncio.sleep(0.005)

        # Final chunk for tool calls.
        yield sse_payload(
            make_stream_chunk(
                stream_id=stream_id,
                created=created,
                model=request.model,
                delta={},
                finish_reason="tool_calls",
            )
        )

        yield "data: [DONE]\n\n"
        return

    # Normal text streaming.
    content = str(parsed.get("content", ""))

    for piece in chunk_string(content, size=64):
        yield sse_payload(
            make_stream_chunk(
                stream_id=stream_id,
                created=created,
                model=request.model,
                delta={"content": piece},
                finish_reason=None,
            )
        )
        await asyncio.sleep(0.005)

    # Final chunk for normal message.
    yield sse_payload(
        make_stream_chunk(
            stream_id=stream_id,
            created=created,
            model=request.model,
            delta={},
            finish_reason="stop",
        )
    )

    yield "data: [DONE]\n\n"


async def call_llm(prompt: str, request: ChatRequest) -> str:
    """
    Runs blocking llm.prompt() in a worker thread.

    If your kbench backend supports temperature/max_tokens kwargs,
    you can uncomment the kwargs section.
    """

    try:
        # Conservative default because some simple wrappers only accept prompt.
        response = await asyncio.to_thread(llm.prompt, prompt)

        # If your backend supports these kwargs, use this instead:
        #
        # response = await asyncio.to_thread(
        #     llm.prompt,
        #     prompt,
        #     temperature=request.temperature,
        #     max_tokens=request.max_tokens,
        # )

        return extract_text(response)

    except TypeError:
        # Fallback if the backend signature is strict.
        response = await asyncio.to_thread(llm.prompt, prompt)
        return extract_text(response)


def map_backend_error(e: Exception) -> HTTPException:
    error = str(e).lower()

    if "quota" in error:
        return HTTPException(status_code=429, detail="Kaggle quota exceeded")

    if "permission" in error or "access denied" in error:
        return HTTPException(status_code=403, detail="Access denied")

    return HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


# ==========================================================
# ROUTES
# ==========================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "loaded": llm is not None,
        "model": MODEL_ID,
    }


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "kaggle",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    prompt = build_prompt(request)

    try:
        raw_text = await call_llm(prompt, request)
        parsed = parse_model_output(raw_text)

        logger.info("Raw model output preview: %s", raw_text[:500])
        logger.info("Parsed response type: %s", parsed.get("type"))

    except Exception as e:
        raise map_backend_error(e)

    if request.stream:
        return StreamingResponse(
            stream_openai_chunks(request, parsed),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return build_nonstream_response(request, parsed, prompt)


# ==================== START SERVER ====================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" STARTING KAGGLE BENCHMARKS OPENAI PROXY")
    print("=" * 60)
    print("\nServer is now running...")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_config=None,
    )# import os
# os.system("/benchmarks/.venv/bin/python /kaggle/working/main.py &")

# !curl http://localhost:9191/v1/chat/completions \
#  -H "Content-Type: application/json" \
# -d '{"model": "google/gemini-3-flash-preview", "messages": [{"role": "user", "content": "what is 3+7"}]}'
