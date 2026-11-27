"""
PostgreSQL 資料庫模組（SQLAlchemy 2.0 + psycopg v3）

管理四張主要資料表：
1. chat_sessions       取代 data/chat_sessions.json（前端創作工作區紀錄）
2. photo_records       取代 data/photo.json（API 契約圖片與版本紀錄，實體 PNG 仍存於 data/photo/）
3. generation_records  紀錄 server.py /generate 單次推論參數與耗時
4. conversations       保存 llm_api.py / image_api.py 多輪對話上下文（依 conversation_id）

連線字串讀取順序：
    環境變數 DATABASE_URL -> 專案根目錄 .env -> 預設本機連線字串
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    func,
    select,
    text,
)
from sqlalchemy.engine import Engine, URL, make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LEGACY_SESSIONS_JSON = DATA_DIR / "chat_sessions.json"
LEGACY_PHOTO_JSON = DATA_DIR / "photo.json"

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    env_file = ROOT / ".env"
    if env_file.is_file():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip("'\""))

DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://postgres:postgres@127.0.0.1:5432/localgenphoto"
)


def normalize_database_url(raw_url: str) -> str:
    """確保連線字串使用 psycopg (v3) driver，避免誤用 psycopg2。"""
    url = raw_url.strip()
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://") :]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def get_database_url() -> str:
    return normalize_database_url(os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL))


class Base(DeclarativeBase):
    pass


class ChatSessionModel(Base):
    """對應前端 LocalGenPhoto_UI.html 的創作紀錄（原 data/chat_sessions.json）。"""

    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0, index=True)
    mode: Mapped[str] = mapped_column(String(32), nullable=False, default="single")
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="單次生成圖片")
    title_edited: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False, default="")
    style: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    proportion: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    resolution: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    num: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    example: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    extra_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class PhotoRecordModel(Base):
    """對應 image_api.py 的圖片與版本紀錄（原 data/photo.json）。"""

    __tablename__ = "photo_records"

    image_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    parent_image_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    conversation_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False, default="")
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    seed: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class GenerationRecordModel(Base):
    """對應 server.py /generate 的推論歷史紀錄。"""

    __tablename__ = "generation_records"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    negative_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_id: Mapped[str] = mapped_column(String(64), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    steps: Mapped[int] = mapped_column(Integer, nullable=False)
    guidance: Mapped[float] = mapped_column(Float, nullable=False)
    seed: Mapped[int] = mapped_column(BigInteger, nullable=False)
    elapsed_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class ConversationModel(Base):
    """對應 llm_api.py / image_api.py 的多輪對話上下文（依 conversation_id 管理）。"""

    __tablename__ = "conversations"

    conversation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    messages_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    image_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


_init_lock = threading.Lock()
_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
_initialized = False


class DatabaseUnavailableError(RuntimeError):
    """當 PostgreSQL 無法連線或尚未設定正確密碼時拋出。"""


def _ensure_database_exists(url: URL) -> None:
    """若目標 PostgreSQL 資料庫尚未建立，自動連線至 postgres 預設庫建立之。"""
    if not url.drivername.startswith("postgresql"):
        return
    target_db = url.database
    if not target_db or not re.fullmatch(r"[A-Za-z0-9_-]+", target_db):
        return

    admin_url = url.set(database="postgres")
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT", pool_pre_ping=True)
    try:
        with admin_engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
                {"dbname": target_db},
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{target_db}" ENCODING \'UTF8\''))
                print(f"[db] 已自動建立 PostgreSQL 資料庫：{target_db}")
    finally:
        admin_engine.dispose()


def _get_engine_and_session() -> tuple[Engine, sessionmaker[Session]]:
    global _engine, _session_factory
    if _engine is None or _session_factory is None:
        db_url = get_database_url()
        _engine = create_engine(db_url, pool_pre_ping=True)
        _session_factory = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine, _session_factory


def init_db(*, raise_on_error: bool = True) -> bool:
    """初始化資料庫（自動建庫、建表，並在空表時匯入舊有 JSON 紀錄）。"""
    global _initialized, _engine, _session_factory
    with _init_lock:
        if _initialized:
            return True
        try:
            db_url = get_database_url()
            parsed_url = make_url(db_url)
            _ensure_database_exists(parsed_url)

            engine, session_factory = _get_engine_and_session()
            Base.metadata.create_all(bind=engine)
            _migrate_legacy_json(session_factory)
            _initialized = True
            safe_url = parsed_url.render_as_string(hide_password=True)
            print(f"[db] PostgreSQL 已就緒：{safe_url}")
            return True
        except Exception as exc:
            _engine = None
            _session_factory = None
            msg = (
                f"PostgreSQL 連線或初始化失敗：{exc}。"
                "請確認 PostgreSQL 服務已啟動，並在專案根目錄 .env 設定正確的 DATABASE_URL。"
            )
            print(f"[db] 警告：{msg}")
            if raise_on_error:
                raise DatabaseUnavailableError(msg) from exc
            return False


def get_session() -> Session:
    if not _initialized:
        init_db(raise_on_error=True)
    _, session_factory = _get_engine_and_session()
    return session_factory()


# ---------------------------------------------------------------------------
# 舊版 JSON 單次自動匯入
# ---------------------------------------------------------------------------
_KNOWN_SESSION_KEYS = {
    "id",
    "mode",
    "title",
    "titleEdited",
    "prompt",
    "style",
    "proportion",
    "resolution",
    "num",
    "example",
}


def _dict_to_session_model(item: dict[str, Any], order: int) -> ChatSessionModel:
    extra = {k: v for k, v in item.items() if k not in _KNOWN_SESSION_KEYS}
    raw_title = item.get("title")
    title = raw_title if isinstance(raw_title, str) and raw_title.strip() else "單次生成圖片"
    return ChatSessionModel(
        id=str(item.get("id") or f"{int(datetime.now().timestamp() * 1000)}-{order}"),
        sort_order=order,
        mode=str(item.get("mode") or "single"),
        title=title,
        title_edited=bool(item.get("titleEdited", False)),
        prompt=str(item.get("prompt") or ""),
        style=str(item.get("style") or ""),
        proportion=str(item.get("proportion") or ""),
        resolution=str(item.get("resolution") or ""),
        num=str(item.get("num") or ""),
        example=str(item.get("example") or ""),
        extra_json=json.dumps(extra, ensure_ascii=False) if extra else None,
        updated_at=datetime.now(timezone.utc),
    )


def _session_model_to_dict(row: ChatSessionModel) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if row.extra_json:
        try:
            parsed = json.loads(row.extra_json)
            if isinstance(parsed, dict):
                result.update(parsed)
        except json.JSONDecodeError:
            pass
    result.update(
        {
            "id": row.id,
            "mode": row.mode,
            "title": row.title,
            "titleEdited": row.title_edited,
            "prompt": row.prompt,
            "style": row.style,
            "proportion": row.proportion,
            "resolution": row.resolution,
            "num": row.num,
            "example": row.example,
        }
    )
    return result


def _parse_iso_datetime(value: Any) -> datetime:
    if isinstance(value, str) and value.strip():
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _photo_model_to_dict(row: PhotoRecordModel) -> dict[str, Any]:
    created = row.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return {
        "image_id": row.image_id,
        "filename": row.filename,
        "version": row.version,
        "parent_image_id": row.parent_image_id,
        "conversation_id": row.conversation_id,
        "prompt": row.prompt,
        "width": row.width,
        "height": row.height,
        "seed": row.seed,
        "created_at": created.astimezone(timezone.utc).isoformat(timespec="seconds"),
    }


def _migrate_legacy_json(session_factory: sessionmaker[Session]) -> None:
    """若 PostgreSQL 資料表為空且舊有 JSON 檔有資料，自動匯入一次。"""
    with session_factory() as db:
        session_count = db.scalar(select(func.count()).select_from(ChatSessionModel)) or 0
        if session_count == 0 and LEGACY_SESSIONS_JSON.is_file():
            try:
                raw = json.loads(LEGACY_SESSIONS_JSON.read_text(encoding="utf-8").strip() or "[]")
                if isinstance(raw, list) and raw:
                    seen_ids: set[str] = set()
                    for idx, item in enumerate(raw):
                        if not isinstance(item, dict):
                            continue
                        model = _dict_to_session_model(item, idx)
                        if model.id in seen_ids:
                            continue
                        seen_ids.add(model.id)
                        db.add(model)
                    db.commit()
                    print(f"[db] 已從 chat_sessions.json 匯入 {len(seen_ids)} 筆創作紀錄")
            except Exception as exc:
                db.rollback()
                print(f"[db] 略過 chat_sessions.json 匯入：{exc}")

        photo_count = db.scalar(select(func.count()).select_from(PhotoRecordModel)) or 0
        if photo_count == 0 and LEGACY_PHOTO_JSON.is_file():
            try:
                raw = json.loads(LEGACY_PHOTO_JSON.read_text(encoding="utf-8").strip() or "[]")
                if isinstance(raw, list) and raw:
                    seen_imgs: set[str] = set()
                    for item in raw:
                        if not isinstance(item, dict) or not item.get("image_id"):
                            continue
                        img_id = str(item["image_id"])
                        if img_id in seen_imgs:
                            continue
                        seen_imgs.add(img_id)
                        db.add(
                            PhotoRecordModel(
                                image_id=img_id,
                                filename=str(item.get("filename") or f"{img_id}.png"),
                                version=int(item.get("version") or 1),
                                parent_image_id=item.get("parent_image_id"),
                                conversation_id=item.get("conversation_id"),
                                prompt=str(item.get("prompt") or ""),
                                width=int(item.get("width") or 1024),
                                height=int(item.get("height") or 1024),
                                seed=int(item.get("seed") or 0),
                                created_at=_parse_iso_datetime(item.get("created_at")),
                            )
                        )
                    db.commit()
                    print(f"[db] 已從 photo.json 匯入 {len(seen_imgs)} 筆圖片紀錄")
            except Exception as exc:
                db.rollback()
                print(f"[db] 略過 photo.json 匯入：{exc}")


# ---------------------------------------------------------------------------
# Chat Sessions CRUD（供 Background/server.py 使用）
# ---------------------------------------------------------------------------
def get_all_sessions() -> list[dict[str, Any]]:
    try:
        with get_session() as db:
            rows = db.scalars(
                select(ChatSessionModel).order_by(
                    ChatSessionModel.sort_order.asc(), ChatSessionModel.updated_at.desc()
                )
            ).all()
            return [_session_model_to_dict(r) for r in rows]
    except SQLAlchemyError as exc:
        raise DatabaseUnavailableError(f"讀取創作紀錄失敗：{exc}") from exc


def replace_all_sessions(sessions: list[dict[str, Any]]) -> int:
    """以單一 Transaction 同步整份前端創作紀錄清單並保留排列順序。"""
    try:
        with get_session() as db:
            incoming_ids: set[str] = set()
            deduped: list[tuple[int, dict[str, Any], str]] = []
            for idx, item in enumerate(sessions):
                if not isinstance(item, dict):
                    continue
                sid = str(item.get("id") or f"{int(datetime.now().timestamp() * 1000)}-{idx}")
                if sid in incoming_ids:
                    continue
                incoming_ids.add(sid)
                deduped.append((idx, item, sid))

            if incoming_ids:
                db.execute(
                    delete(ChatSessionModel).where(ChatSessionModel.id.notin_(incoming_ids))
                )
            else:
                db.execute(delete(ChatSessionModel))

            existing_map = {
                row.id: row
                for row in db.scalars(
                    select(ChatSessionModel).where(ChatSessionModel.id.in_(incoming_ids))
                ).all()
            } if incoming_ids else {}

            for idx, item, sid in deduped:
                fresh = _dict_to_session_model({**item, "id": sid}, idx)
                target = existing_map.get(sid)
                if target is None:
                    db.add(fresh)
                else:
                    target.sort_order = fresh.sort_order
                    target.mode = fresh.mode
                    target.title = fresh.title
                    target.title_edited = fresh.title_edited
                    target.prompt = fresh.prompt
                    target.style = fresh.style
                    target.proportion = fresh.proportion
                    target.resolution = fresh.resolution
                    target.num = fresh.num
                    target.example = fresh.example
                    target.extra_json = fresh.extra_json
                    target.updated_at = fresh.updated_at

            db.commit()
            return len(deduped)
    except SQLAlchemyError as exc:
        raise DatabaseUnavailableError(f"寫入創作紀錄失敗：{exc}") from exc


# ---------------------------------------------------------------------------
# Photo Records CRUD（供 Background/image_api.py 使用）
# ---------------------------------------------------------------------------
def load_photo_records() -> list[dict[str, Any]]:
    try:
        with get_session() as db:
            rows = db.scalars(
                select(PhotoRecordModel).order_by(PhotoRecordModel.created_at.asc())
            ).all()
            return [_photo_model_to_dict(r) for r in rows]
    except SQLAlchemyError as exc:
        raise DatabaseUnavailableError(f"讀取圖片紀錄失敗：{exc}") from exc


def find_photo_record(image_id: str) -> dict[str, Any] | None:
    try:
        with get_session() as db:
            row = db.get(PhotoRecordModel, image_id)
            return _photo_model_to_dict(row) if row is not None else None
    except SQLAlchemyError as exc:
        raise DatabaseUnavailableError(f"查詢圖片紀錄失敗：{exc}") from exc


def insert_photo_record(record: dict[str, Any]) -> dict[str, Any]:
    try:
        with get_session() as db:
            row = PhotoRecordModel(
                image_id=str(record["image_id"]),
                filename=str(record["filename"]),
                version=int(record["version"]),
                parent_image_id=record.get("parent_image_id"),
                conversation_id=record.get("conversation_id"),
                prompt=str(record["prompt"]),
                width=int(record["width"]),
                height=int(record["height"]),
                seed=int(record["seed"]),
                created_at=_parse_iso_datetime(record.get("created_at")),
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return _photo_model_to_dict(row)
    except SQLAlchemyError as exc:
        raise DatabaseUnavailableError(f"儲存圖片紀錄失敗：{exc}") from exc


# ---------------------------------------------------------------------------
# Generation History CRUD（供 Background/server.py /generate 與 /history 使用）
# ---------------------------------------------------------------------------
def log_generation(
    *,
    prompt: str,
    negative_prompt: str | None,
    model_id: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int,
    elapsed_sec: float,
) -> str:
    try:
        with get_session() as db:
            row = GenerationRecordModel(
                prompt=prompt,
                negative_prompt=negative_prompt,
                model_id=model_id,
                width=width,
                height=height,
                steps=steps,
                guidance=guidance,
                seed=seed,
                elapsed_sec=elapsed_sec,
            )
            db.add(row)
            db.commit()
            return row.id
    except SQLAlchemyError as exc:
        raise DatabaseUnavailableError(f"寫入生圖歷史失敗：{exc}") from exc


def get_generation_history(limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
    try:
        with get_session() as db:
            rows = db.scalars(
                select(GenerationRecordModel)
                .order_by(GenerationRecordModel.created_at.desc())
                .offset(offset)
                .limit(limit)
            ).all()
            return [
                {
                    "id": r.id,
                    "prompt": r.prompt,
                    "negative_prompt": r.negative_prompt,
                    "model_id": r.model_id,
                    "width": r.width,
                    "height": r.height,
                    "steps": r.steps,
                    "guidance": r.guidance,
                    "seed": r.seed,
                    "elapsed_sec": r.elapsed_sec,
                    "created_at": (
                        r.created_at.replace(tzinfo=timezone.utc)
                        if r.created_at.tzinfo is None
                        else r.created_at
                    )
                    .astimezone(timezone.utc)
                    .isoformat(timespec="seconds"),
                }
                for r in rows
            ]
    except SQLAlchemyError as exc:
        raise DatabaseUnavailableError(f"讀取生圖歷史失敗：{exc}") from exc


# ---------------------------------------------------------------------------
# Conversations CRUD（供 Background/llm_api.py 與 Background/image_api.py 使用）
# ---------------------------------------------------------------------------
def _conversation_model_to_dict(row: ConversationModel) -> dict[str, Any]:
    try:
        messages = json.loads(row.messages_json or "[]")
        if not isinstance(messages, list):
            messages = []
    except json.JSONDecodeError:
        messages = []
    return {
        "conversation_id": row.conversation_id,
        "messages": messages,
        "image_prompt": row.image_prompt,
    }


def get_conversation_record(conversation_id: str) -> dict[str, Any] | None:
    try:
        with get_session() as db:
            row = db.get(ConversationModel, conversation_id)
            return _conversation_model_to_dict(row) if row is not None else None
    except SQLAlchemyError as exc:
        raise DatabaseUnavailableError(f"讀取對話紀錄失敗：{exc}") from exc


def save_conversation_record(
    conversation_id: str,
    messages: list[dict[str, Any]],
    image_prompt: str | None = None,
) -> dict[str, Any]:
    try:
        with get_session() as db:
            row = db.get(ConversationModel, conversation_id)
            serialized = json.dumps(messages, ensure_ascii=False)
            now = datetime.now(timezone.utc)
            if row is None:
                row = ConversationModel(
                    conversation_id=conversation_id,
                    messages_json=serialized,
                    image_prompt=image_prompt,
                    created_at=now,
                    updated_at=now,
                )
                db.add(row)
            else:
                row.messages_json = serialized
                row.image_prompt = image_prompt
                row.updated_at = now
            db.commit()
            db.refresh(row)
            return _conversation_model_to_dict(row)
    except SQLAlchemyError as exc:
        raise DatabaseUnavailableError(f"寫入對話紀錄失敗：{exc}") from exc
