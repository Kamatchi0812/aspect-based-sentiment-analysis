from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SentimentLabel = Literal["positive", "neutral", "negative"]


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    artifacts_ready: bool
    retrieval_backend: Literal["faiss", "cosine", "tfidf"]


class BrandMetric(BaseModel):
    brand: str
    review_count: int
    avg_rating: float
    positive_rate: float
    neutral_rate: float
    negative_rate: float


class AspectMetric(BaseModel):
    aspect: str
    mention_count: int
    positive_rate: float
    neutral_rate: float
    negative_rate: float


class ReviewRecord(BaseModel):
    review_id: int
    brand: str
    product_name: str
    review: str
    rating: int
    sentiment: SentimentLabel
    aspects: list[str]
    similarity: float | None = None


class OverviewResponse(BaseModel):
    total_reviews: int
    total_brands: int
    avg_rating: float
    positive_rate: float
    neutral_rate: float
    negative_rate: float
    model_accuracy: float | None = None
    top_brands: list[BrandMetric]
    top_aspects: list[AspectMetric]


class FiltersResponse(BaseModel):
    brands: list[str]
    aspects: list[str]
    sentiments: list[SentimentLabel]


class ReviewsResponse(BaseModel):
    total: int
    items: list[ReviewRecord]


class AnalyzeReviewRequest(BaseModel):
    review: str = Field(min_length=3)
    product_name: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class AnalyzeReviewResponse(BaseModel):
    normalized_review: str
    predicted_sentiment: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    brand: str | None = None
    aspects: list[str]
    summary: str
    similar_reviews: list[ReviewRecord]


class InsightRequest(BaseModel):
    query: str = Field(min_length=3)
    brand: str | None = None
    aspect: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    use_gemini: bool = True


class InsightResponse(BaseModel):
    query: str
    answer: str
    recommended_actions: list[str]
    dominant_sentiment: str
    brands_mentioned: list[str]
    aspects_mentioned: list[str]
    retrieved_reviews: list[ReviewRecord]
    mode: Literal["gemini", "template"]


class BrandComparisonRecord(BaseModel):
    brand: str
    aspect: str
    mention_count: int
    avg_rating: float
    positive_rate: float
    neutral_rate: float
    negative_rate: float


class BrandComparisonResponse(BaseModel):
    aspect: str | None = None
    items: list[BrandComparisonRecord]


class AspectSummaryResponse(BaseModel):
    items: list[AspectMetric]


class ExportFile(BaseModel):
    name: str
    path: str


class ExportListResponse(BaseModel):
    files: list[ExportFile]
