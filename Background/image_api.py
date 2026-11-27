"""
依照 docs/API_CONTRACT.md 實作的完整生圖 API（LLM 對話 + 生圖引擎 + 圖片儲存）

架構：
    前端 ──► image_api.py (本檔, :8002) ──► llm_api.call_llm()      整理提示詞／追問
                                        └─► server.py (:8000) /generate  實際生圖（diffusers）
    生成的圖片存到 data/photo/<image_id>.png，並在 data/photo.json 登記圖片名稱與版本資訊。

啟動（三個都要開）：
    ollama serve
    cd Background && uvicorn server:app --host 127.0.0.1 --port 8000
    cd Background && uvicorn image_api:app --host 127.0.0.1 --port 8002

API：
    POST /api/v1/image-generation       主要入口（契約 §3）
    GET  /api/v1/images/{image_id}      取得圖片檔（契約 §5.1 的 images[].url）
    GET  /api/v1/images                 列出 photo.json 裡所有圖片紀錄
    GET  /api/v1/conversations/{id}     除錯用：查看對話狀態

關於修改圖片（operation=edit）：
    server.py 目前只有文字轉圖片，沒有 img2img。這裡的做法是用 LLM 整理出的「新完整提示詞」
    搭配來源圖片的同一個 seed、同樣尺寸重新生成，讓構圖盡量接近原圖。
    之後 server.py 若加上 img2img，只需要改 run_engine() 這一個函式。
"""

from contextlib import asynccontextmanager
import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from math import gcd
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

try:
    from database import (
        DatabaseUnavailableError,
        find_photo_record,
        get_conversation_record,
        init_db,
        insert_photo_record,
        load_photo_records,
        save_conversation_record,
    )
except ImportError:
    from Background.database import (
        DatabaseUnavailableError,
        find_photo_record,
        get_conversation_record,
        init_db,
        insert_photo_record,
        load_photo_records,
        save_conversation_record,
    )

from llm_api import (
    ChatRequest,
    LLMError,
    SingleGenerationRequest,
    call_llm,
    translate_to_english,
    error_response,
    new_request_id,
)
from llm_api import _lock as _conv_lock

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
ENGINE_URL = "http://127.0.0.1:8000/generate"  # Background/server.py
ENGINE_MODEL = "flux1-schnell"  # server.py MODELS 裡的 id；None = 用 server.py 的預設模型
ENGINE_TIMEOUT = 900         # 第一次載入模型可能很久
DEFAULT_SIZE = (1024, 1024)  # 對話式生圖沒有指定尺寸時使用
MIN_SIDE, MAX_SIDE = 256, 2048  # 與 server.py 的限制一致
SIZE_MULTIPLE = 16           # diffusion 模型的寬高需為 16 的倍數
MAX_COUNT = 4                # 契約 §9 待確認：單次最多張數
RATIO_TOLERANCE = 0.02       # resolution 與 aspect_ratio 允許的誤差（2%）

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PHOTO_DIR = DATA_DIR / "photo"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db(raise_on_error=False)
    yield


app = FastAPI(title="Local ImageGen API (contract v1.0.0)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def on_validation_error(_: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    return error_response(400, "INVALID_REQUEST", f"{loc}: {first.get('msg', '格式錯誤')}")


@app.exception_handler(DatabaseUnavailableError)
async def on_db_error(_: Request, exc: DatabaseUnavailableError):
    return error_response(503, "DATABASE_UNAVAILABLE", str(exc), retryable=True)


# ---------------------------------------------------------------------------
# Image Storage：data/photo/*.png + PostgreSQL（photo_records 表）
# ---------------------------------------------------------------------------
_store_lock = threading.Lock()


def load_records() -> list[dict]:
    return load_photo_records()


def find_record(image_id: str) -> dict | None:
    return find_photo_record(image_id)


def store_image(png: bytes, *, prompt: str, width: int, height: int, seed: int,
                version: int, parent_image_id: str | None,
                conversation_id: str | None) -> dict:
    image_id = "img_" + uuid.uuid4().hex[:12]
    filename = f"{image_id}.png"
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    (PHOTO_DIR / filename).write_bytes(png)

    record = {
        "image_id": image_id,
        "filename": filename,
        "version": version,
        "parent_image_id": parent_image_id,
        "conversation_id": conversation_id,
        "prompt": prompt,
        "width": width,
        "height": height,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with _store_lock:
        return insert_photo_record(record)


def to_image_item(record: dict) -> dict:
    """轉成契約 §4.3 的 images[] 格式。"""
    return {
        "image_id": record["image_id"],
        "version": record["version"],
        "parent_image_id": record["parent_image_id"],
        "url": f"/api/v1/images/{record['image_id']}",
        "width": record["width"],
        "height": record["height"],
    }


# ---------------------------------------------------------------------------
# Image Engine：呼叫 server.py
# ---------------------------------------------------------------------------
class EngineError(RuntimeError):
    def __init__(self, status: int, code: str, message: str, retryable: bool):
        super().__init__(message)
        self.status, self.code, self.retryable = status, code, retryable


def run_engine(prompt: str, width: int, height: int, seed: int = -1) -> tuple[bytes, int]:
    payload = {"prompt": prompt, "width": width, "height": height, "seed": seed}
    if ENGINE_MODEL:
        payload["model"] = ENGINE_MODEL
    req = urllib.request.Request(
        ENGINE_URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=ENGINE_TIMEOUT) as r:
            return r.read(), int(r.headers.get("X-Seed", seed))
    except urllib.error.HTTPError as err:
        try:
            detail = json.loads(err.read().decode("utf-8")).get("detail", "")
        except Exception:
            detail = ""
        if err.code == 400:
            raise EngineError(400, "INVALID_REQUEST", detail or "生圖引擎拒絕此參數。", False)
        if err.code == 507:
            raise EngineError(503, "IMAGE_ENGINE_OOM", detail or "顯存不足。", True)
        raise EngineError(500, "IMAGE_ENGINE_ERROR", detail or f"生圖引擎錯誤（HTTP {err.code}）。", False)
    except (urllib.error.URLError, TimeoutError) as err:
        reason = getattr(err, "reason", err)
        raise EngineError(503, "IMAGE_ENGINE_UNAVAILABLE",
                          f"無法連線到生圖引擎（{reason}），請確認 server.py 是否已啟動。", True)


# ---------------------------------------------------------------------------
# 參數驗證（契約 §8 第 6、7 點）
# ---------------------------------------------------------------------------
def validate_single(req: SingleGenerationRequest) -> str | None:
    w, h = req.resolution.width, req.resolution.height
    for name, v in (("width", w), ("height", h)):
        if not MIN_SIDE <= v <= MAX_SIDE:
            return f"resolution.{name} 必須介於 {MIN_SIDE} 到 {MAX_SIDE} 之間。"
        if v % SIZE_MULTIPLE:
            return f"resolution.{name} 必須是 {SIZE_MULTIPLE} 的倍數。"
    ar = req.aspect_ratio.width / req.aspect_ratio.height
    if abs(w / h - ar) / ar > RATIO_TOLERANCE:
        g = gcd(req.aspect_ratio.width, req.aspect_ratio.height)
        return (f"resolution {w}×{h} 與 aspect_ratio "
                f"{req.aspect_ratio.width // g}:{req.aspect_ratio.height // g} 不一致。")
    return None


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def completed(images: list[dict], conversation_id: str | None = None) -> dict:
    body = {"request_id": new_request_id(), "status": "completed"}
    if conversation_id is not None:
        body |= {"conversation_id": conversation_id, "need_more_info": False}
    body["images"] = images
    return body


def handle_single(req: SingleGenerationRequest):
    if (msg := validate_single(req)) is not None:
        return error_response(400, "INVALID_RESOLUTION", msg)
    if req.count > MAX_COUNT:
        return error_response(400, "INVALID_COUNT", f"count 最多為 {MAX_COUNT}。")
    if req.input_image is not None:
        return error_response(400, "INPUT_IMAGE_NOT_SUPPORTED", "目前尚未支援 input_image 參考圖片。")

    # 多一步：先用 LLM 把提示詞與風格翻成英文，再交給生圖引擎
    try:
        en = translate_to_english(req.prompt, req.style)
    except LLMError as err:
        return error_response(503, "LLM_UNAVAILABLE", str(err), retryable=True)
    style = en["style"]
    if style and not style.lower().endswith("style"):
        style += " style"
    prompt = f"{en['prompt']}, {style}" if style else en["prompt"]
    print(f"[translate] {req.prompt!r} / {req.style!r} -> {prompt!r}")
    w, h = req.resolution.width, req.resolution.height
    images = []
    try:
        for _ in range(req.count):
            png, seed = run_engine(prompt, w, h)
            rec = store_image(png, prompt=prompt, width=w, height=h, seed=seed,
                              version=1, parent_image_id=None, conversation_id=None)
            images.append(to_image_item(rec))
    except EngineError as err:
        return error_response(err.status, err.code, str(err), err.retryable)
    return completed(images)


def handle_chat(req: ChatRequest):
    source = None
    if req.operation == "edit":
        if req.source_image is None:
            return error_response(400, "INVALID_REQUEST", "operation=edit 時必須提供 source_image。")
        source = find_record(req.source_image.image_id)
        if source is None or source["version"] != req.source_image.version:
            return error_response(404, "IMAGE_NOT_FOUND", "找不到指定的圖片版本。")
    if req.input_image is not None:
        return error_response(400, "INPUT_IMAGE_NOT_SUPPORTED", "目前尚未支援 input_image 參考圖片。")

    if source is not None:
        user_text = f"[修改] 來源圖片提示詞：{source['prompt']}\n修改要求：{req.message}"
    else:
        user_text = req.message

    with _conv_lock:
        existing = get_conversation_record(req.conversation_id)
        messages = list(existing["messages"]) if existing else []
        prev_prompt = existing["image_prompt"] if existing else None
        history = [*messages, {"role": "user", "content": user_text}]

    try:
        result = call_llm(history)
    except LLMError as err:
        return error_response(503, "LLM_UNAVAILABLE", str(err), retryable=True)

    if result["need_more_info"] and result["question"]:
        with _conv_lock:
            updated_messages = [*history, {"role": "assistant", "content": result["question"]}]
            save_conversation_record(req.conversation_id, updated_messages, prev_prompt)
        return {
            "request_id": new_request_id(),
            "status": "needs_clarification",
            "conversation_id": req.conversation_id,
            "need_more_info": True,
            "question": result["question"],
        }

    prompt = result["image_prompt"] or req.message
    with _conv_lock:
        updated_messages = [*history, {"role": "assistant", "content": f"(已整理提示詞) {prompt}"}]
        save_conversation_record(req.conversation_id, updated_messages, prompt)

    if source is not None:
        w, h, seed = source["width"], source["height"], source["seed"]
        version, parent = source["version"] + 1, source["image_id"]
    else:
        (w, h), seed, version, parent = DEFAULT_SIZE, -1, 1, None

    try:
        png, used_seed = run_engine(prompt, w, h, seed)
    except EngineError as err:
        return error_response(err.status, err.code, str(err), err.retryable)

    rec = store_image(png, prompt=prompt, width=w, height=h, seed=used_seed,
                      version=version, parent_image_id=parent,
                      conversation_id=req.conversation_id)
    return completed([to_image_item(rec)], req.conversation_id)


@app.post("/api/v1/image-generation")
def image_generation(req: ChatRequest | SingleGenerationRequest):
    t0 = time.time()
    resp = handle_single(req) if isinstance(req, SingleGenerationRequest) else handle_chat(req)
    print(f"[image-generation] flow={req.flow} done in {time.time() - t0:.1f}s")
    return resp


@app.get("/api/v1/images/{image_id}")
def get_image(image_id: str):
    rec = find_record(image_id)
    path = PHOTO_DIR / rec["filename"] if rec else None
    if path is None or not path.is_file():
        return error_response(404, "IMAGE_NOT_FOUND", "找不到指定的圖片。")
    return FileResponse(path, media_type="image/png")


@app.get("/api/v1/images")
def list_images():
    with _store_lock:
        return load_records()


@app.get("/api/v1/conversations/{conversation_id}")
def show_conversation(conversation_id: str):
    conv = get_conversation_record(conversation_id)
    if conv is None:
        return error_response(404, "CONVERSATION_NOT_FOUND", "找不到指定的對話。")
    return conv
