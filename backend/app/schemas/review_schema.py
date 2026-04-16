from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ReviewSource = Literal["text", "voice", "video"]
ReviewSentiment = Literal["positive", "neutral", "negative"]


class ReviewCreate(BaseModel):
    product: str = Field(min_length=1, max_length=255)
    brand: str = Field(min_length=1, max_length=100)
    review_text: str = Field(min_length=1)
    sentiment: ReviewSentiment
    aspect: str | None = Field(default=None, max_length=255)
    language: str = Field(default="unknown", min_length=2, max_length=50)
    source: ReviewSource = "text"
    rating: float | None = Field(default=None, ge=0.0, le=5.0)

    @field_validator("product", "brand", "review_text", "language")
    @classmethod
    def strip_required_values(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Value cannot be empty.")
        return cleaned

    @field_validator("aspect")
    @classmethod
    def strip_optional_values(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class ReviewResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    product: str
    brand: str
    review_text: str
    sentiment: ReviewSentiment
    aspect: str | None = None
    language: str
    source: ReviewSource
    rating: float | None = None
    created_at: datetime


class ReviewListResponse(BaseModel):
    total: int
    limit: int
    skip: int
    items: list[ReviewResponse]
