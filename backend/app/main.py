from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import Settings, get_settings
from backend.app.models import (
    AnalyzeReviewRequest,
    AnalyzeReviewResponse,
    AspectSummaryResponse,
    BrandComparisonResponse,
    ExportListResponse,
    FiltersResponse,
    HealthResponse,
    InsightRequest,
    InsightResponse,
    OverviewResponse,
    ReviewsResponse,
)
from backend.app.services.artifact_builder import ArtifactBuilder
from backend.app.services.review_intelligence import ReviewIntelligenceService

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
    aspect: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 15,
    service: Annotated[ReviewIntelligenceService, Depends(get_review_service)] = None,
) -> BrandComparisonResponse:
    return service.get_brand_comparison(aspect=aspect, limit=limit)


@app.get("/api/v1/aspect-summary", response_model=AspectSummaryResponse)
def get_aspect_summary(
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
    service: Annotated[ReviewIntelligenceService, Depends(get_review_service)] = None,
) -> AspectSummaryResponse:
    return service.get_aspect_summary(limit=limit)


@app.get("/api/v1/reviews", response_model=ReviewsResponse)
def get_reviews(
    query: Annotated[str | None, Query()] = None,
    brand: Annotated[str | None, Query()] = None,
    aspect: Annotated[str | None, Query()] = None,
    sentiment: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
    service: Annotated[ReviewIntelligenceService, Depends(get_review_service)] = None,
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
) -> AnalyzeReviewResponse:
    return service.analyze_review(request)


@app.post("/api/v1/insights", response_model=InsightResponse)
def generate_insight(
    request: InsightRequest,
    service: Annotated[ReviewIntelligenceService, Depends(get_review_service)],
) -> InsightResponse:
    return service.generate_insight(request)


@app.get("/api/v1/powerbi/exports", response_model=ExportListResponse)
def list_powerbi_exports(
    service: Annotated[ReviewIntelligenceService, Depends(get_review_service)],
) -> ExportListResponse:
    return service.list_powerbi_exports()
