from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.models.review import Review
from backend.app.schemas.review_schema import ReviewCreate


def create_review(db: Session, review_data: ReviewCreate) -> Review:
    review = Review(**review_data.model_dump())
    db.add(review)
    db.commit()
    db.refresh(review)
    return review


def get_reviews(db: Session, limit: int = 20, skip: int = 0) -> tuple[list[Review], int]:
    return _get_review_page(db, limit=limit, skip=skip)


def get_reviews_by_brand(
    db: Session,
    brand: str,
    limit: int = 20,
    skip: int = 0,
) -> tuple[list[Review], int]:
    return _get_review_page(
        db,
        limit=limit,
        skip=skip,
        where_clause=func.lower(Review.brand) == brand.lower(),
    )


def get_reviews_by_sentiment(
    db: Session,
    sentiment: str,
    limit: int = 20,
    skip: int = 0,
) -> tuple[list[Review], int]:
    return _get_review_page(
        db,
        limit=limit,
        skip=skip,
        where_clause=func.lower(Review.sentiment) == sentiment.lower(),
    )


def _get_review_page(
    db: Session,
    *,
    limit: int,
    skip: int,
    where_clause: object | None = None,
) -> tuple[list[Review], int]:
    base_query = select(Review)
    count_query = select(func.count()).select_from(Review)
    if where_clause is not None:
        base_query = base_query.where(where_clause)
        count_query = count_query.where(where_clause)

    items = db.scalars(
        base_query.order_by(Review.created_at.desc(), Review.id.desc()).offset(skip).limit(limit)
    ).all()
    total = db.scalar(count_query) or 0
    return items, int(total)
