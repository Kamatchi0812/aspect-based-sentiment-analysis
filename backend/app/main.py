from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.config import Settings, get_settings
from backend.app.crud.review_crud import (
    create_review,
    get_reviews as get_reviews_from_db,
    get_reviews_by_brand,
    get_reviews_by_sentiment,
)
from backend.app.db.database import Base, engine
from backend.app.db.session import get_db
from backend.app.models.review import Review
from backend.app.schemas.api_schema import (
    AnalyzeReviewRequest,
    AnalyzeReviewResponse,
    AnalyticsSummaryResponse,
    AspectSummaryResponse,
    BrandComparisonResponse,
    ExportListResponse,
    FiltersResponse,
    HealthResponse,
    InsightRequest,
    InsightResponse,
    OverviewResponse,
    ReviewsResponse,
    StoredAnalysisResponse,
    VideoReviewRequest,
    VoiceReviewRequest,
)
from backend.app.schemas.review_schema import ReviewListResponse, ReviewResponse
from backend.app.services.artifact_builder import ArtifactBuilder
from backend.app.services.review_intelligence import ReviewIntelligenceService

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Multilingual Laptop Review Intelligence System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_review_service_cached() -> ReviewIntelligenceService:
    return ReviewIntelligenceService(get_settings())


def get_review_service() -> ReviewIntelligenceService:
    return get_review_service_cached()


def _store_review_from_analysis(
    *,
    db: Session,
    service: ReviewIntelligenceService,
    product_name: str | None,
    review_text: str,
    analysis: AnalyzeReviewResponse,
    source: str,
    rating: float | None = None,
    language: str | None = None,
) -> Review:
    payload = service.build_review_create(
        product_name=product_name,
        review_text=review_text,
        predicted_sentiment=analysis.predicted_sentiment,
        aspects=analysis.aspects,
        source=source,
        rating=rating,
        language=language,
    )
    stored_review = create_review(db, payload)
    service.ingest_review(stored_review)
    return stored_review


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Multilingual Laptop Review Intelligence API",
        "status": "ok",
        "health": "/health",
        "docs": "/docs",
        "overview": "/api/v1/overview",
    }


@app.get("/health", response_model=HealthResponse)
def health(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    artifact_builder = ArtifactBuilder(settings)
    retrieval_backend = "tfidf"
    metrics_path = settings.artifact_dir / "metrics.json"
    if metrics_path.exists():
        try:
            import json

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            retrieval_backend = metrics.get("runtime_retrieval_backend", "tfidf")
        except Exception:
            retrieval_backend = "tfidf"

    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        artifacts_ready=artifact_builder.artifacts_ready(),
        retrieval_backend=retrieval_backend,  # type: ignore[arg-type]
    )


@app.get("/api/v1/overview", response_model=OverviewResponse)
def get_overview(
    service: Annotated[ReviewIntelligenceService, Depends(get_review_service)],
) -> OverviewResponse:
    return service.get_overview()


@app.get("/api/v1/filters", response_model=FiltersResponse)
def get_filters(
    service: Annotated[ReviewIntelligenceService, Depends(get_review_service)],
) -> FiltersResponse:
    return service.get_filters()


@app.get("/api/v1/brand-comparison", response_model=BrandComparisonResponse)
def get_brand_comparison(
    service: Annotated[ReviewIntelligenceService, Depends(get_review_service)],
    aspect: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 15,
) -> BrandComparisonResponse:
    return service.get_brand_comparison(aspect=aspect, limit=limit)


@app.get("/api/v1/aspect-summary", response_model=AspectSummaryResponse)
def get_aspect_summary(
    service: Annotated[ReviewIntelligenceService, Depends(get_review_service)],
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> AspectSummaryResponse:
    return service.get_aspect_summary(limit=limit)


@app.get("/api/v1/reviews", response_model=ReviewsResponse)
def get_reviews(
    service: Annotated[ReviewIntelligenceService, Depends(get_review_service)],
    query: Annotated[str | None, Query()] = None,
    brand: Annotated[str | None, Query()] = None,
    aspect: Annotated[str | None, Query()] = None,
    sentiment: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> ReviewsResponse:
    return service.search_reviews(
        query=query,
        brand=brand,
        aspect=aspect,
        sentiment=sentiment,
        limit=limit,
    )


@app.post("/api/v1/analyze-review", response_model=AnalyzeReviewResponse)
def analyze_review(
    request: AnalyzeReviewRequest,
    service: Annotated[ReviewIntelligenceService, Depends(get_review_service)],
    db: Annotated[Session, Depends(get_db)],
) -> AnalyzeReviewResponse:
    analysis = service.analyze_review(request)
    _store_review_from_analysis(
        db=db,
        service=service,
        product_name=request.product_name,
        review_text=request.review,
        analysis=analysis,
        source="text",
    )
    return analysis


@app.post("/api/v1/voice-review", response_model=StoredAnalysisResponse)
def analyze_voice_review(
    request: VoiceReviewRequest,
    service: Annotated[ReviewIntelligenceService, Depends(get_review_service)],
    db: Annotated[Session, Depends(get_db)],
) -> StoredAnalysisResponse:
    analysis = service.analyze_content(
        review_text=request.transcript,
        product_name=request.product_name,
        top_k=request.top_k,
    )
    stored_review = _store_review_from_analysis(
        db=db,
        service=service,
        product_name=request.product_name,
        review_text=request.transcript,
        analysis=analysis,
        source="voice",
        rating=request.rating,
        language=request.language,
    )
    return StoredAnalysisResponse(analysis=analysis, stored_review_id=stored_review.id, source="voice")


@app.post("/api/v1/video-review", response_model=StoredAnalysisResponse)
def analyze_video_review(
    request: VideoReviewRequest,
    service: Annotated[ReviewIntelligenceService, Depends(get_review_service)],
    db: Annotated[Session, Depends(get_db)],
) -> StoredAnalysisResponse:
    review_text = request.review_text or request.transcript
    if not review_text:
        raise HTTPException(status_code=400, detail="review_text or transcript is required.")

    analysis = service.analyze_content(
        review_text=review_text,
        product_name=request.product_name,
        top_k=request.top_k,
        forced_sentiment=request.gesture_sentiment,
    )
    stored_review = _store_review_from_analysis(
        db=db,
        service=service,
        product_name=request.product_name,
        review_text=review_text,
        analysis=analysis,
        source="video",
        rating=request.rating,
        language=request.language,
    )
    return StoredAnalysisResponse(analysis=analysis, stored_review_id=stored_review.id, source="video")


@app.post("/api/v1/insights", response_model=InsightResponse)
def generate_insight(
    request: InsightRequest,
    service: Annotated[ReviewIntelligenceService, Depends(get_review_service)],
) -> InsightResponse:
    return service.generate_insight(request)


@app.get("/api/v1/live-reviews", response_model=ReviewListResponse)
def list_live_reviews(
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    skip: Annotated[int, Query(ge=0)] = 0,
    brand: Annotated[str | None, Query()] = None,
    sentiment: Annotated[str | None, Query()] = None,
) -> ReviewListResponse:
    if brand:
        items, total = get_reviews_by_brand(db, brand, limit=limit, skip=skip)
    elif sentiment:
        items, total = get_reviews_by_sentiment(db, sentiment, limit=limit, skip=skip)
    else:
        items, total = get_reviews_from_db(db, limit=limit, skip=skip)

    return ReviewListResponse(
        total=total,
        limit=limit,
        skip=skip,
        items=[ReviewResponse.model_validate(item) for item in items],
    )


@app.get("/api/v1/analytics/summary", response_model=AnalyticsSummaryResponse)
def analytics_summary(
    db: Annotated[Session, Depends(get_db)],
) -> AnalyticsSummaryResponse:
    total_reviews = db.scalar(select(func.count()).select_from(Review)) or 0
    sentiment_rows = db.execute(
        select(Review.sentiment, func.count()).group_by(Review.sentiment).order_by(func.count().desc())
    ).all()
    language_rows = db.execute(
        select(Review.language, func.count()).group_by(Review.language).order_by(func.count().desc())
    ).all()
    source_rows = db.execute(
        select(Review.source, func.count()).group_by(Review.source).order_by(func.count().desc())
    ).all()
    return AnalyticsSummaryResponse(
        total_reviews=int(total_reviews),
        sentiment_distribution={str(key): int(value) for key, value in sentiment_rows},
        language_distribution={str(key): int(value) for key, value in language_rows},
        source_distribution={str(key): int(value) for key, value in source_rows},
    )


@app.get("/api/v1/powerbi/exports", response_model=ExportListResponse)
def list_powerbi_exports(
    service: Annotated[ReviewIntelligenceService, Depends(get_review_service)],
) -> ExportListResponse:
    return service.list_powerbi_exports()
