from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

TESTS_DIR = Path(__file__).resolve().parent
TEST_DB_PATH = TESTS_DIR / "test_reviews.db"

os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH.as_posix()}"
os.environ.setdefault("ENABLE_FAKE_MODE", "true")

from backend.app.db.database import Base, engine
from backend.app.db.session import SessionLocal
from backend.app.main import app


@pytest.fixture(autouse=True)
def reset_database() -> None:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
