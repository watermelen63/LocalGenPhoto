"""
本地端生圖後端（FastAPI + diffusers）

啟動：
    uvicorn server:app --host 127.0.0.1 --port 8000

API：
    GET  /models      列出可用模型與目前載入的模型
    POST /generate    生圖，回傳 PNG（一次只跑一張，其他請求會排隊）
    GET  /progress    目前生圖進度（前端用來更新進度條）
    POST /unload      卸載目前模型，釋放顯存
"""

from contextlib import asynccontextmanager
import gc
import io
import json
import threading
import time
import urllib.request

import torch
import diffusers
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

try:
    from database import (
        DatabaseUnavailableError,
        get_all_sessions,
        get_generation_history,
        init_db,
        log_generation,
        replace_all_sessions,
    )
except ImportError:
    from Background.database import (
        DatabaseUnavailableError,
        get_all_sessions,
        get_generation_history,
        init_db,
        log_generation,
        replace_all_sessions,
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
    # FLUX.1（12B）原始 bf16 約 34GB，8GB 顯存 + 16GB RAM 放不下，所以：
    #   transformer 用預先量化好的 GGUF Q4（約 6.8GB），T5 文字編碼器載入時量化成 NF4（約 3GB）
    #   顯存不夠時可把 Q4_K_S 換成 Q3_K_S / Q2_K（品質會下降）
    "flux1-schnell": {
        "label": "FLUX.1 schnell (12B GGUF Q4, Apache 2.0)",
        "cls": "FluxPipeline",
        "repo": "black-forest-labs/FLUX.1-schnell",
        "steps": 4,
        "guidance": 0.0,
        "offload": True,
        "extra": {"max_sequence_length": 256},  # schnell 最多 256
        "gguf_transformer": "https://huggingface.co/city96/FLUX.1-schnell-gguf/blob/main/flux1-schnell-Q4_K_S.gguf",
        "nf4_text_encoder": "text_encoder_2",
    },
    "flux1-dev": {
        # 非商用授權，需先在 Hugging Face 同意條款並 `hf auth login`
        "label": "FLUX.1 dev (12B GGUF Q4, non-commercial)",
        "cls": "FluxPipeline",
        "repo": "black-forest-labs/FLUX.1-dev",
        "steps": 28,
        "guidance": 3.5,
        "offload": True,
        "extra": {"max_sequence_length": 512},
        "gguf_transformer": "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q4_K_S.gguf",
        "nf4_text_encoder": "text_encoder_2",
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
    init_db(raise_on_error=False)
    yield


app = FastAPI(title="Local ImageGen Backend", lifespan=lifespan)

# 前端 HTML 若是直接用檔案開啟（file://）也要能呼叫 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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


def load_quantized_components(cfg: dict, dtype) -> dict:
    """依 MODELS 設定先載入量化過的元件，再當作參數傳給 from_pretrained（這些元件就不會重複下載）。
        gguf_transformer : GGUF 檔網址 → transformer
        nf4_text_encoder : 子資料夾名稱 → 用 bitsandbytes 量化成 4-bit 的 T5 文字編碼器
    """
    components = {}
    # 順序很重要：T5 原檔 9.5GB，要趁 RAM 還空的時候先載入並量化到 GPU（約 3GB 顯存），
    # 之後才載 GGUF transformer（6.8GB 放 RAM）。反過來的話 16GB RAM 會不夠（os error 1455）。
    if cfg.get("nf4_text_encoder"):
        from transformers import BitsAndBytesConfig, T5EncoderModel

        name = cfg["nf4_text_encoder"]
        print(f"[model] loading {name} as NF4")
        components[name] = T5EncoderModel.from_pretrained(
            cfg["repo"],
            subfolder=name,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            ),
            torch_dtype=dtype,
        )
        gc.collect()
    if cfg.get("gguf_transformer"):
        from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig

        print(f"[model] loading GGUF transformer {cfg['gguf_transformer']}")
        components["transformer"] = FluxTransformer2DModel.from_single_file(
            cfg["gguf_transformer"],
            config=cfg["repo"],
            subfolder="transformer",
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
        )
    return components


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
    pipe = pipe_cls.from_pretrained(cfg["repo"], torch_dtype=dtype,
                                    **load_quantized_components(cfg, dtype))

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

        elapsed = time.time() - _progress["started"]
        try:
            log_generation(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                model_id=req.model,
                width=req.width,
                height=req.height,
                steps=steps,
                guidance=guidance,
                seed=seed,
                elapsed_sec=elapsed,
            )
        except DatabaseUnavailableError as exc:
            print(f"[db] 略過寫入生圖歷史：{exc}")

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={
                "X-Seed": str(seed),
                "X-Model": req.model,
                "X-Elapsed": f"{elapsed:.2f}",
            },
        )


# ---------------------------------------------------------------------------
# 生圖歷史紀錄（PostgreSQL generation_records 表）
# ---------------------------------------------------------------------------
@app.get("/history")
def list_history(limit: int = 20, offset: int = 0):
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    try:
        return get_generation_history(limit=limit, offset=offset)
    except DatabaseUnavailableError as exc:
        raise HTTPException(503, str(exc))


# ---------------------------------------------------------------------------
# 創作紀錄：前端把整份 sessions 送來，寫進 PostgreSQL（chat_sessions 表）
# ---------------------------------------------------------------------------
@app.get("/sessions")
def get_sessions():
    try:
        return get_all_sessions()
    except DatabaseUnavailableError as exc:
        raise HTTPException(503, str(exc))


@app.put("/sessions")
def save_sessions(sessions: list[dict]):
    try:
        count = replace_all_sessions(sessions)
    except DatabaseUnavailableError as exc:
        raise HTTPException(503, str(exc))
    return {"ok": True, "count": count}
