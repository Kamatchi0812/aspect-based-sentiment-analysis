from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from backend.app.schemas.review_schema import ReviewSource

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


class VoiceReviewRequest(BaseModel):
    transcript: str = Field(min_length=3)
    product_name: str | None = None
    rating: float | None = Field(default=None, ge=0.0, le=5.0)
    language: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class VideoReviewRequest(BaseModel):
    product_name: str | None = None
    transcript: str | None = None
    review_text: str | None = None
    gesture_sentiment: SentimentLabel
    rating: float | None = Field(default=None, ge=0.0, le=5.0)
    language: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def ensure_text_content(self) -> "VideoReviewRequest":
        if not (self.transcript or self.review_text):
            raise ValueError("Either transcript or review_text is required.")
        return self


class StoredAnalysisResponse(BaseModel):
    analysis: AnalyzeReviewResponse
    stored_review_id: int
    source: ReviewSource


class AnalyticsSummaryResponse(BaseModel):
    total_reviews: int
    sentiment_distribution: dict[str, int]
    language_distribution: dict[str, int]
    source_distribution: dict[str, int]
