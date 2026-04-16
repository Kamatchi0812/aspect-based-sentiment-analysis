from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase

from backend.app.config import get_settings


def _normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


class Base(DeclarativeBase):
    pass


settings = get_settings()
database_url = _normalize_database_url(settings.database_url)

connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}

engine = create_engine(
    database_url,
    pool_pre_ping=True,
    future=True,
    connect_args=connect_args,
)


def init_db() -> None:
    from backend.app.models.review import Review  # noqa: F401

    Base.metadata.create_all(bind=engine)
