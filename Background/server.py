"""
本地端生圖後端（FastAPI + diffusers）

啟動：
    uvicorn server:app --host 127.0.0.1 --port 8000

API：
    GET  /models                  列出可用模型與目前載入的模型
    POST /generate                生圖，儲存圖片與資料庫紀錄並回傳 PNG
    GET  /progress                目前生圖進度（前端用來更新進度條）
    POST /unload                  卸載目前模型，釋放顯存
    GET  /history                 依照時間由新到舊讀取生成歷史紀錄
    GET  /history/{id}/image      讀取歷史生成圖片檔案
"""

import gc
import io
import json
import math
import sys
import threading
import time
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import diffusers
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from LocalGenPhoto.database import (  # noqa: E402
    add_generation,
    get_generation_by_id,
    get_generations,
    init_db,
    resolve_image_path,
    save_generation_image,
)

# ---------------------------------------------------------------------------
# 模型設定表：要加新模型，只要在這裡加一筆
#   cls      : diffusers 裡的 Pipeline 類別名稱
#   repo     : Hugging Face 上的 repo
#   steps    : 預設步數
#   guidance : 預設 guidance_scale
#   offload  : True = 用 enable_model_cpu_offload()（16GB 顯存跑較大模型時用）
#   extra    : 額外傳給 pipeline 的參數
# ---------------------------------------------------------------------------
MODELS = {
    "z-image-turbo": {
        "label": "Z-Image Turbo (6B, Apache 2.0)",
        "cls": "ZImagePipeline",
        "repo": "Tongyi-MAI/Z-Image-Turbo",
        "steps": 9,
        "guidance": 0.0,
        "offload": False,
        "extra": {},
    },
    "flux2-klein-4b": {
        "label": "FLUX.2 klein 4B (Apache 2.0)",
        "cls": "Flux2KleinPipeline",
        "repo": "black-forest-labs/FLUX.2-klein-4B",
        "steps": 4,
        "guidance": 1.0,
        "offload": True,
        "extra": {},
    },
    "ernie-image-turbo": {
        "label": "ERNIE-Image Turbo (8B, Apache 2.0)",
        "cls": "ErnieImagePipeline",
        "repo": "baidu/ERNIE-Image-Turbo",
        "steps": 8,
        "guidance": 1.0,
        "offload": True,
        "extra": {"use_pe": False},  # Prompt Enhancer 會多吃顯存，16GB 先關掉
    },
    "sdxl": {
        "label": "SDXL (3.5B)",
        "cls": "StableDiffusionXLPipeline",
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "steps": 30,
        "guidance": 6.0,
        "offload": False,
        "extra": {},
        "dtype": "float16",
    },
}

DEFAULT_MODEL = "z-image-turbo"
OLLAMA_URL = "http://127.0.0.1:11434"
UNLOAD_OLLAMA_BEFORE_GENERATE = True  # 生圖前先請 Ollama 釋放顯存

@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


init_db()
app = FastAPI(title="Local ImageGen Backend", lifespan=lifespan)

# ---------------------------------------------------------------------------
# 全域狀態
# ---------------------------------------------------------------------------
_pipe = None
_pipe_id = None
_gen_lock = threading.Lock()  # 確保同一時間只有一張圖在生（GPU 只有一張）
_progress = {"busy": False, "model": None, "step": 0, "total": 0, "started": None}


class GenerateRequest(BaseModel):
    prompt: str
    model: str = DEFAULT_MODEL
    negative_prompt: str | None = None
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    steps: int | None = Field(None, ge=1, le=100)
    guidance: float | None = None
    seed: int = -1
    aspect_ratio: str | None = None
    style: str | None = None


# ---------------------------------------------------------------------------
# 工具函式
# ---------------------------------------------------------------------------
def unload_ollama_models():
    """請 Ollama 把目前載入的模型從顯存卸載（keep_alive=0）。Ollama 沒開就略過。"""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/ps", timeout=2) as r:
            running = json.load(r).get("models", [])
        for m in running:
            body = json.dumps({"model": m["name"], "keep_alive": 0}).encode()
            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/generate",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10).read()
            print(f"[ollama] unloaded {m['name']}")
    except Exception:
        pass  # Ollama 沒在跑或連不到都沒關係


def free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_pipe():
    global _pipe, _pipe_id
    if _pipe is not None:
        print(f"[model] unloading {_pipe_id}")
        _pipe = None
        _pipe_id = None
        free_vram()


def load_pipe(model_id: str):
    """載入指定模型；如果已經是目前的模型就直接回傳。"""
    global _pipe, _pipe_id
    if _pipe_id == model_id:
        return _pipe

    cfg = MODELS[model_id]
    unload_pipe()  # 16GB 只放得下一個模型，先清掉舊的

    pipe_cls = getattr(diffusers, cfg["cls"], None)
    if pipe_cls is None:
        raise HTTPException(
            500,
            f"你的 diffusers 沒有 {cfg['cls']}，請用 "
            "`pip install -U git+https://github.com/huggingface/diffusers` 更新",
        )

    dtype = torch.float16 if cfg.get("dtype") == "float16" else torch.bfloat16
    print(f"[model] loading {model_id} from {cfg['repo']} ...")
    t0 = time.time()
    pipe = pipe_cls.from_pretrained(cfg["repo"], torch_dtype=dtype)

    if cfg["offload"]:
        pipe.enable_model_cpu_offload()  # 用到哪個元件才搬上 GPU
    else:
        pipe.to("cuda")

    # 生大圖時避免 VAE 解碼爆顯存
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    print(f"[model] {model_id} ready in {time.time() - t0:.1f}s")
    _pipe, _pipe_id = pipe, model_id
    return pipe


def make_progress_callback(total: int):
    def _cb(pipe, step, timestep, callback_kwargs):
        _progress["step"] = step + 1
        _progress["total"] = total
        return callback_kwargs

    return _cb


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/models")
def list_models():
    return {
        "loaded": _pipe_id,
        "default": DEFAULT_MODEL,
        "models": [{"id": k, "label": v["label"]} for k, v in MODELS.items()],
    }


@app.get("/progress")
def progress():
    return _progress


@app.post("/unload")
def unload():
    with _gen_lock:
        unload_pipe()
    return {"ok": True}


def _format_aspect_ratio(width: int, height: int, explicit: str | None) -> str:
    if explicit:
        return explicit
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


@app.post("/generate")
def generate(req: GenerateRequest):
    # FastAPI 會把一般 def 放進執行緒池跑，所以 /progress 可以同時被查詢
    if req.model not in MODELS:
        raise HTTPException(400, f"未知的模型：{req.model}")

    with _gen_lock:  # 排隊：前一張沒生完，這裡會等
        cfg = MODELS[req.model]
        steps = req.steps or cfg["steps"]
        guidance = cfg["guidance"] if req.guidance is None else req.guidance

        _progress.update(busy=True, model=req.model, step=0, total=steps,
                         started=time.time())
        try:
            if UNLOAD_OLLAMA_BEFORE_GENERATE:
                unload_ollama_models()

            pipe = load_pipe(req.model)

            seed = req.seed if req.seed >= 0 else int(torch.seed() % (2**32))
            generator = torch.Generator("cpu").manual_seed(seed)

            kwargs = dict(
                prompt=req.prompt,
                width=req.width,
                height=req.height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                callback_on_step_end=make_progress_callback(steps),
                **cfg["extra"],
            )
            if req.negative_prompt:
                kwargs["negative_prompt"] = req.negative_prompt

            try:
                image = pipe(**kwargs).images[0]
            except TypeError:
                # 少數 pipeline 不支援某些參數（例如 negative_prompt），去掉再試一次
                kwargs.pop("negative_prompt", None)
                image = pipe(**kwargs).images[0]

        except torch.cuda.OutOfMemoryError:
            unload_pipe()
            raise HTTPException(
                507,
                "顯存不足（OOM）。請降低解析度，或在 MODELS 裡把這個模型的 offload 設為 True。",
            )
        finally:
            _progress.update(busy=False)

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        image_rel_path = save_generation_image(png_bytes)
        parameters = {
            "seed": seed,
            "width": req.width,
            "height": req.height,
            "aspect_ratio": _format_aspect_ratio(req.width, req.height, req.aspect_ratio),
            "steps": steps,
            "guidance": guidance,
        }
        if req.style:
            parameters["style"] = req.style

        generation_id = add_generation(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            model_name=req.model,
            parameters=parameters,
            image_path=image_rel_path,
        )

        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={
                "X-Seed": str(seed),
                "X-Model": req.model,
                "X-Elapsed": f"{time.time() - _progress['started']:.2f}",
                "X-Generation-Id": str(generation_id),
                "X-Image-Path": image_rel_path,
            },
        )


@app.get("/history")
def list_history(limit: int = 50, offset: int = 0):
    """依照建立時間由新到舊回傳生成歷史紀錄。"""
    if limit < 1 or limit > 200:
        raise HTTPException(400, "limit 必須介於 1 到 200 之間")
    if offset < 0:
        raise HTTPException(400, "offset 不可為負數")

    items = get_generations(limit=limit, offset=offset)
    for item in items:
        item["image_url"] = f"/history/{item['id']}/image"
    return {"items": items, "count": len(items)}


@app.get("/history/{generation_id}/image")
def get_history_image(generation_id: int):
    """取得指定生成紀錄對應的 PNG 圖片。"""
    record = get_generation_by_id(generation_id)
    if record is None:
        raise HTTPException(404, "找不到指定的生成紀錄")

    file_path = resolve_image_path(record["image_path"])
    if not file_path.is_file():
        raise HTTPException(404, "找不到對應的圖片檔案")

    return FileResponse(file_path, media_type="image/png")
