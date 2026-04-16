from __future__ import annotations

from sqlalchemy import select

from backend.app.main import app, get_review_service
from backend.app.models.review import Review
from backend.app.schemas.api_schema import AnalyzeReviewResponse
from backend.app.schemas.review_schema import ReviewCreate
from backend.app.services.preprocessing import detect_language, extract_brand


class FakeReviewService:
    def analyze_review(self, request) -> AnalyzeReviewResponse:
        return AnalyzeReviewResponse(
            normalized_review="battery backup excellent performance smooth",
            predicted_sentiment="positive",
            confidence=0.97,
            brand="Dell",
            aspects=["battery", "performance"],
            summary="The review is mostly positive and highlights battery and performance.",
            similar_reviews=[],
        )

    def build_review_create(
        self,
        *,
        product_name,
        review_text,
        predicted_sentiment,
        aspects,
        source,
        rating=None,
        language=None,
    ) -> ReviewCreate:
        product = product_name or "User Submitted Review"
        return ReviewCreate(
            product=product,
            brand=extract_brand(product),
            review_text=review_text,
            sentiment=predicted_sentiment,
            aspect="|".join(aspects),
            language=language or detect_language(review_text),
            source=source,
            rating=rating,
        )

    def ingest_review(self, review: Review) -> None:
        self.last_review_id = review.id


def test_analyze_review_saves_review_to_database(client, db_session) -> None:
    app.dependency_overrides[get_review_service] = lambda: FakeReviewService()

    response = client.post(
        "/api/v1/analyze-review",
        json={
            "review": "Battery backup is excellent and performance is smooth.",
            "product_name": "Dell Inspiron 15",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_sentiment"] == "positive"
    assert payload["aspects"] == ["battery", "performance"]

    stored_review = db_session.scalar(select(Review))
    assert stored_review is not None
    assert stored_review.product == "Dell Inspiron 15"
    assert stored_review.source == "text"
    assert stored_review.sentiment == "positive"


def test_live_reviews_and_analytics_summary_use_database_records(client, db_session) -> None:
    db_session.add_all(
        [
            Review(
                product="HP Pavilion 14",
                brand="HP",
                review_text="Display is good and battery is decent.",
                sentiment="positive",
                aspect="display|battery",
                language="english",
                source="text",
                rating=4.0,
            ),
            Review(
                product="Lenovo IdeaPad 3",
                brand="Lenovo",
                review_text="Performance is average in Tamil and English mix.",
                sentiment="neutral",
                aspect="performance",
                language="tanglish",
                source="voice",
                rating=3.0,
            ),
        ]
    )
    db_session.commit()

    reviews_response = client.get("/api/v1/live-reviews", params={"limit": 10, "skip": 0})
    summary_response = client.get("/api/v1/analytics/summary")

    assert reviews_response.status_code == 200
    assert summary_response.status_code == 200

    reviews_payload = reviews_response.json()
    summary_payload = summary_response.json()

    assert reviews_payload["total"] == 2
    assert len(reviews_payload["items"]) == 2
    assert summary_payload["total_reviews"] == 2
    assert summary_payload["sentiment_distribution"]["positive"] == 1
    assert summary_payload["source_distribution"]["voice"] == 1
