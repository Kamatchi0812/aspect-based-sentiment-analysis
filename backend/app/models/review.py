from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.database import Base


class Review(Base):
    __tablename__ = "reviews"
    __table_args__ = (
        Index("ix_reviews_brand_sentiment", "brand", "sentiment"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    product: Mapped[str] = mapped_column(String(255), nullable=False)
    brand: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    review_text: Mapped[str] = mapped_column(Text, nullable=False)
    sentiment: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    aspect: Mapped[str | None] = mapped_column(String(255), nullable=True)
    language: Mapped[str] = mapped_column(String(50), nullable=False, default="unknown")
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="text", index=True)
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
