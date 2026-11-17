import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import create_engine, Column, String, Integer, BigInteger, Float, Text, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# 預設連線字串（可透過環境變數 DATABASE_URL 覆蓋）
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://gen_admin:gen_password123@127.0.0.1:5432/localgen_db"
)

# 圖片儲存根目錄：專案根目錄下的 data/images
BASE_IMAGE_DIR = Path(__file__).resolve().parent.parent / "data" / "images"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class GenerationRecord(Base):
    """對應 server.py 的單次生圖紀錄與圖片資產"""
    __tablename__ = "generation_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    prompt = Column(Text, nullable=False)
    negative_prompt = Column(Text, nullable=True)
    model_id = Column(String(64), nullable=False)        # 例如 'flux2-klein-4b'
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    steps = Column(Integer, nullable=False)
    guidance = Column(Float, nullable=False)
    seed = Column(BigInteger, nullable=False)
    elapsed_sec = Column(Float, nullable=True)           # 記錄生成耗時
    file_path = Column(String(512), nullable=False)      # 相對路徑，如 '2026/11/16/uuid.png'
    is_favorite = Column(Boolean, default=False)         # 預留給 UI 做收藏功能
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


def init_db():
    """自動建立資料表與圖片資料夾"""
    BASE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def save_image_and_record(
    png_bytes: bytes,
    prompt: str,
    negative_prompt: str | None,
    model_id: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int,
    elapsed_sec: float,
) -> GenerationRecord:
    """將 PNG 存入硬碟，並把參數寫入 PostgreSQL"""
    now = datetime.now()
    # 依日期建立子資料夾：data/images/2026/11/16/
    rel_dir = Path(str(now.year), f"{now.month:02d}", f"{now.day:02d}")
    abs_dir = BASE_IMAGE_DIR / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)

    record_id = str(uuid.uuid4())
    filename = f"{record_id}.png"
    rel_path = (rel_dir / filename).as_posix()  # 統一用正斜線儲存相對路徑
    abs_path = abs_dir / filename

    # 1. 實體圖片存進硬碟
    with open(abs_path, "wb") as f:
        f.write(png_bytes)

    # 2. 生成參數寫入 PostgreSQL
    db = SessionLocal()
    try:
        record = GenerationRecord(
            id=record_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_id=model_id,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            seed=seed,
            elapsed_sec=elapsed_sec,
            file_path=rel_path,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    finally:
        db.close()
