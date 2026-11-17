"""SQLite 資料庫管理模組：管理生圖紀錄與圖片檔案儲存。"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
DB_PATH = DATA_DIR / "gallery.db"


def ensure_storage_dirs(
    db_path: Path = DB_PATH,
    images_dir: Path = IMAGES_DIR,
) -> None:
    """確保資料庫所在目錄與圖片儲存資料夾存在。"""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(images_dir).mkdir(parents=True, exist_ok=True)


@contextmanager
def _connect(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db_path), timeout=15.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(
    db_path: Path = DB_PATH,
    images_dir: Path = IMAGES_DIR,
) -> Path:
    """自動建立資料夾、SQLite 資料庫檔案與 generations 資料表。"""
    db_path = Path(db_path)
    images_dir = Path(images_dir)
    ensure_storage_dirs(db_path=db_path, images_dir=images_dir)

    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                negative_prompt TEXT NOT NULL DEFAULT '',
                model_name TEXT NOT NULL,
                parameters TEXT NOT NULL DEFAULT '{}',
                image_path TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
            """
        )
        conn.commit()
    return db_path


def save_generation_image(
    image_bytes: bytes,
    filename: str | None = None,
    images_dir: Path = IMAGES_DIR,
    project_root: Path = PROJECT_ROOT,
) -> str:
    """將圖片二進位資料寫入 data/images/，回傳相對於專案根目錄的路徑字串。"""
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uid = uuid.uuid4().hex[:8]
        filename = f"gen_{stamp}_{short_uid}.png"

    dest = images_dir / filename
    dest.write_bytes(image_bytes)

    try:
        return dest.resolve().relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        return dest.as_posix()


def resolve_image_path(
    image_path: str,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """將資料庫中的相對路徑解析為完整檔案路徑。"""
    path = Path(image_path)
    if path.is_absolute():
        return path
    return Path(project_root) / path


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    raw_params = row["parameters"]
    try:
        parsed_params = json.loads(raw_params) if raw_params else {}
        if not isinstance(parsed_params, dict):
            parsed_params = {"raw": parsed_params}
    except (json.JSONDecodeError, TypeError):
        parsed_params = {}

    return {
        "id": row["id"],
        "prompt": row["prompt"],
        "negative_prompt": row["negative_prompt"],
        "model_name": row["model_name"],
        "parameters": parsed_params,
        "image_path": row["image_path"],
        "created_at": row["created_at"],
    }


def add_generation(
    prompt: str,
    negative_prompt: str | None,
    model_name: str,
    parameters: dict[str, Any] | str,
    image_path: str,
    db_path: Path = DB_PATH,
    created_at: str | None = None,
) -> int:
    """寫入一筆生圖紀錄到 generations 資料表，回傳新增紀錄的 id。"""
    db_path = Path(db_path)
    init_db(db_path=db_path)

    if isinstance(parameters, str):
        params_json = parameters
    else:
        params_json = json.dumps(parameters, ensure_ascii=False)

    timestamp = created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO generations (
                prompt,
                negative_prompt,
                model_name,
                parameters,
                image_path,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                prompt,
                negative_prompt or "",
                model_name,
                params_json,
                image_path,
                timestamp,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_generations(
    limit: int = 50,
    offset: int = 0,
    db_path: Path = DB_PATH,
) -> list[dict[str, Any]]:
    """依照建立時間由新到舊讀取歷史生圖紀錄。"""
    db_path = Path(db_path)
    init_db(db_path=db_path)

    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, prompt, negative_prompt, model_name, parameters, image_path, created_at
            FROM generations
            ORDER BY created_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

    return [_row_to_dict(row) for row in rows]


def get_generation_by_id(
    generation_id: int,
    db_path: Path = DB_PATH,
) -> dict[str, Any] | None:
    """依 id 讀取單筆生圖紀錄。"""
    db_path = Path(db_path)
    init_db(db_path=db_path)

    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, prompt, negative_prompt, model_name, parameters, image_path, created_at
            FROM generations
            WHERE id = ?
            """,
            (generation_id,),
        ).fetchone()

    return _row_to_dict(row) if row is not None else None
