from __future__ import annotations

from backend.app.crud.review_crud import (
    create_review,
    get_reviews,
    get_reviews_by_brand,
    get_reviews_by_sentiment,
)
from backend.app.schemas.review_schema import ReviewCreate


def test_create_review_persists_record(db_session) -> None:
    review = create_review(
        db_session,
        ReviewCreate(
            product="Dell Inspiron 15",
            brand="Dell",
            review_text="Battery backup is excellent and performance is smooth.",
            sentiment="positive",
            aspect="battery|performance",
            language="english",
            source="text",
            rating=4.5,
        ),
    )

    assert review.id is not None
    assert review.brand == "Dell"
    assert review.sentiment == "positive"
    assert review.source == "text"


def test_get_reviews_supports_pagination_and_filters(db_session) -> None:
    payloads = [
        ReviewCreate(
            product="Dell Inspiron 15",
            brand="Dell",
            review_text="Good battery and good display.",
            sentiment="positive",
            aspect="battery|display",
            language="english",
            source="text",
            rating=4.0,
        ),
        ReviewCreate(
            product="HP Pavilion 14",
            brand="HP",
            review_text="Average performance but okay for office use.",
            sentiment="neutral",
            aspect="performance",
            language="english",
            source="voice",
            rating=3.0,
        ),
        ReviewCreate(
            product="Dell Vostro 14",
            brand="Dell",
            review_text="Keyboard quality is poor.",
            sentiment="negative",
            aspect="keyboard",
            language="english",
            source="video",
            rating=2.0,
        ),
    ]

    for payload in payloads:
        create_review(db_session, payload)

    page_items, total = get_reviews(db_session, limit=2, skip=1)
    dell_items, dell_total = get_reviews_by_brand(db_session, "dell", limit=10, skip=0)
    negative_items, negative_total = get_reviews_by_sentiment(
        db_session,
        "negative",
        limit=10,
        skip=0,
    )

    assert total == 3
    assert len(page_items) == 2
    assert dell_total == 2
    assert {item.brand for item in dell_items} == {"Dell"}
    assert negative_total == 1
    assert negative_items[0].sentiment == "negative"
