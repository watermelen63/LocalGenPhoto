"""
依照 docs/API_CONTRACT.md 實作的 LLM 端（先不接生圖引擎）

啟動：
    cd Background
    uvicorn llm_api:app --host 127.0.0.1 --port 8001

API：
    POST /api/v1/image-generation       主要入口（契約 §3）
    GET  /api/v1/conversations/{id}     除錯用：查看對話狀態與整理後的生圖提示詞

目前進度：
    * flow=true  → 交給本地 Ollama 判斷「資訊是否足夠」
        - 不足 → 200 + status=needs_clarification + question
        - 足夠 → 整理出英文生圖提示詞並存進對話狀態；因為還沒接生圖引擎，
                 回 503 + status=failed + error.code=IMAGE_ENGINE_NOT_CONNECTED
    * flow=false → 一樣回 503（尚未接生圖引擎）
    之後接上生圖引擎時，只要改 run_image_engine() 這一個函式。
"""

import json
import threading
import urllib.error
import urllib.request
import uuid
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen3.5:4b"  # 本機 Ollama 目前裝的模型；UI 那邊寫的是 qwen3:4b

SYSTEM_PROMPT = """你是生圖助手的需求分析器。根據整段對話，判斷使用者的生圖需求是否足以開始生成。
只輸出一個 JSON 物件，不要有其他文字，格式如下：
{"need_more_info": true或false, "question": "要追問的一個問題（不需要追問時為空字串）", "image_prompt": "整理好的英文生圖提示詞（需要追問時為空字串）"}
規則：
- 只在缺少「無法合理猜測」的關鍵資訊時才追問，一次只問一個問題，用使用者的語言。
- 主題、場景大致明確就直接 need_more_info=false，風格等細節自行補得合理即可。
- 若使用者是在修改既有圖片（會標註「[修改]」），image_prompt 要寫成完整的新提示詞，保留先前提示詞中未被要求改動的部分。"""

app = FastAPI(title="Local ImageGen LLM API")


# ---------------------------------------------------------------------------
# Request 模型（對應 schemas/api-contract.schema.json）
# ---------------------------------------------------------------------------
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceImage(_Strict):
    image_id: str = Field(min_length=1)
    version: int = Field(ge=1)


class InputImage(_Strict):
    asset_id: str = Field(min_length=1)


class Size(_Strict):
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class ChatRequest(_Strict):
    flow: Literal[True]
    conversation_id: str = Field(min_length=1)
    operation: Literal["generate", "edit"]
    message: str = Field(min_length=1)
    source_image: SourceImage | None = None
    input_image: InputImage | None = None


class SingleGenerationRequest(_Strict):
    flow: Literal[False]
    prompt: str = Field(min_length=1)
    style: str = Field(min_length=1)
    aspect_ratio: Size
    resolution: Size
    count: int = Field(ge=1)
    input_image: InputImage | None = None


# ---------------------------------------------------------------------------
# 錯誤處理：一律回契約 §6 的錯誤物件
# ---------------------------------------------------------------------------
def new_request_id() -> str:
    return "req_" + uuid.uuid4().hex[:12]


def error_response(status_code: int, code: str, message: str,
                   retryable: bool = False) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "request_id": new_request_id(),
            "status": "failed",
            "error": {"code": code, "message": message, "retryable": retryable},
        },
    )


@app.exception_handler(RequestValidationError)
async def on_validation_error(_: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    return error_response(400, "INVALID_REQUEST", f"{loc}: {first.get('msg', '格式錯誤')}")


# ---------------------------------------------------------------------------
# 對話狀態（Conversation Manager，目前放記憶體）
# ---------------------------------------------------------------------------
_conversations: dict[str, dict] = {}
_lock = threading.Lock()


def get_conversation(conversation_id: str) -> dict:
    return _conversations.setdefault(
        conversation_id, {"messages": [], "image_prompt": None}
    )


# ---------------------------------------------------------------------------
# LLM（Ollama）
# ---------------------------------------------------------------------------
class LLMError(RuntimeError):
    pass


def call_llm(messages: list[dict]) -> dict:
    body = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        "stream": False,
        "format": "json",
        "think": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_ENDPOINT, data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 本地不走代理
    try:
        with opener.open(req, timeout=300) as r:
            content = json.loads(r.read().decode("utf-8"))["message"]["content"]
        result = json.loads(content)
    except urllib.error.URLError as err:
        raise LLMError(f"無法連線到本地 AI（{err.reason}），請確認 Ollama 是否已啟動。") from err
    except (KeyError, json.JSONDecodeError) as err:
        raise LLMError("LLM 回傳的格式無法解析。") from err

    return {
        "need_more_info": bool(result.get("need_more_info")),
        "question": str(result.get("question") or "").strip(),
        "image_prompt": str(result.get("image_prompt") or "").strip(),
    }


def run_image_engine(image_prompt: str, req) -> list[dict]:
    """TODO：接上生圖引擎（Background/server.py 的 diffusers）。
    要回傳契約 §4.3 的 images 陣列；目前尚未接上，所以丟例外。"""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def not_connected() -> JSONResponse:
    return error_response(
        503, "IMAGE_ENGINE_NOT_CONNECTED", "生圖引擎尚未接上（目前只完成 LLM 對話部分）。"
    )


@app.post("/api/v1/image-generation")
def image_generation(req: ChatRequest | SingleGenerationRequest):
    if isinstance(req, SingleGenerationRequest):
        return not_connected()

    user_text = ("[修改] " if req.operation == "edit" else "") + req.message

    with _lock:
        conv = get_conversation(req.conversation_id)
        conv["messages"].append({"role": "user", "content": user_text})
        history = list(conv["messages"])

    try:
        result = call_llm(history)
    except LLMError as err:
        with _lock:
            conv["messages"].pop()  # 這一輪失敗，不把使用者訊息留在紀錄裡
        return error_response(503, "LLM_UNAVAILABLE", str(err), retryable=True)

    if result["need_more_info"] and result["question"]:
        with _lock:
            conv["messages"].append({"role": "assistant", "content": result["question"]})
        return {
            "request_id": new_request_id(),
            "status": "needs_clarification",
            "conversation_id": req.conversation_id,
            "need_more_info": True,
            "question": result["question"],
        }

    # 資訊足夠：保存整理好的提示詞，交給生圖引擎
    image_prompt = result["image_prompt"] or req.message
    with _lock:
        conv["messages"].append({"role": "assistant", "content": f"(已整理提示詞) {image_prompt}"})
        conv["image_prompt"] = image_prompt
    try:
        images = run_image_engine(image_prompt, req)
    except NotImplementedError:
        return not_connected()

    return {
        "request_id": new_request_id(),
        "status": "completed",
        "conversation_id": req.conversation_id,
        "need_more_info": False,
        "images": images,
    }


@app.get("/api/v1/conversations/{conversation_id}")
def show_conversation(conversation_id: str):
    conv = _conversations.get(conversation_id)
    if conv is None:
        return error_response(404, "CONVERSATION_NOT_FOUND", "找不到指定的對話。")
    return conv
