from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel
from sqlalchemy import func, select

from backend.app.config import Settings
from backend.app.db.session import SessionLocal
from backend.app.models.review import Review
from backend.app.schemas.api_schema import (
    AnalyzeReviewRequest,
    AnalyzeReviewResponse,
    AspectMetric,
    AspectSummaryResponse,
    BrandComparisonRecord,
    BrandComparisonResponse,
    BrandMetric,
    ExportFile,
    ExportListResponse,
    FiltersResponse,
    InsightRequest,
    InsightResponse,
    OverviewResponse,
    ReviewRecord,
    ReviewsResponse,
    SentimentLabel,
)
from backend.app.schemas.review_schema import ReviewCreate
from backend.app.services.artifact_builder import ArtifactBuilder
from backend.app.services.gemini_client import GeminiClient
from backend.app.services.preprocessing import (
    build_aspect_contexts,
    build_rag_text,
    detect_language,
    extract_aspects,
    extract_brand,
    join_aspects,
    normalize_aspect_string,
    preprocess_review,
    split_aspects,
)

SENTIMENT_TO_RATING = {"positive": 5, "neutral": 3, "negative": 1}


class ReviewIntelligenceService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.artifact_builder = ArtifactBuilder(settings)
        self.artifact_builder.ensure()
        self.artifacts_ready = self.artifact_builder.artifacts_ready()

        artifact_dir = Path(settings.artifact_dir)
        self.processed_reviews = pd.read_csv(artifact_dir / "processed_reviews.csv")
        self.aspect_mentions = pd.read_csv(artifact_dir / "aspect_mentions.csv")
        self.rag_metadata = pd.read_csv(artifact_dir / "rag_metadata.csv")
        self.vectorizer = joblib.load(artifact_dir / "sentiment_vectorizer.joblib")
        self.model = joblib.load(artifact_dir / "sentiment_model.joblib")
        self.rag_vectorizer = joblib.load(artifact_dir / "rag_vectorizer.joblib")
        self.rag_matrix = sparse.load_npz(artifact_dir / "rag_matrix.npz")
        self.metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
        self.gemini_client = GeminiClient(settings)
        self.retrieval_backend = "tfidf"

        self.processed_reviews["aspects_list"] = self.processed_reviews["aspects"].apply(split_aspects)
        self.rag_metadata["aspects_list"] = self.rag_metadata["aspects"].apply(split_aspects)
        if "review_id" in self.aspect_mentions.columns:
            self.aspect_mentions["review_id"] = self.aspect_mentions["review_id"].astype(int)
        self._known_review_ids = {int(value) for value in self.processed_reviews["review_id"].tolist()}
        self._recompute_aggregates()
        self._sync_runtime_delta()

    def get_overview(self) -> OverviewResponse:
        self._sync_runtime_delta()
        total_reviews = int(len(self.processed_reviews))
        sentiment_counts = self.processed_reviews["sentiment"].value_counts()
        denominator = total_reviews or 1

        top_brands = [
            BrandMetric(
                brand=str(row.brand),
                review_count=int(row.review_count),
                avg_rating=round(float(row.avg_rating), 2),
                positive_rate=round(float(row.positive_rate), 2),
                neutral_rate=round(float(row.neutral_rate), 2),
                negative_rate=round(float(row.negative_rate), 2),
            )
            for row in self.brand_overview.head(10).itertuples(index=False)
        ]
        top_aspects = [
            AspectMetric(
                aspect=str(row.aspect),
                mention_count=int(row.mention_count),
                positive_rate=round(float(row.positive_rate), 2),
                neutral_rate=round(float(row.neutral_rate), 2),
                negative_rate=round(float(row.negative_rate), 2),
            )
            for row in self.aspect_summary.head(10).itertuples(index=False)
        ]

        return OverviewResponse(
            total_reviews=total_reviews,
            total_brands=int(self.processed_reviews["brand"].nunique()) if total_reviews else 0,
            avg_rating=round(float(self.processed_reviews["rating"].mean()), 2) if total_reviews else 0.0,
            positive_rate=round(float(sentiment_counts.get("positive", 0) / denominator * 100), 2),
            neutral_rate=round(float(sentiment_counts.get("neutral", 0) / denominator * 100), 2),
            negative_rate=round(float(sentiment_counts.get("negative", 0) / denominator * 100), 2),
            model_accuracy=self.metrics.get("accuracy"),
            top_brands=top_brands,
            top_aspects=top_aspects,
        )

    def get_filters(self) -> FiltersResponse:
        self._sync_runtime_delta()
        return FiltersResponse(
            brands=sorted(self.processed_reviews["brand"].dropna().unique().tolist()),
            aspects=sorted(self.aspect_summary["aspect"].dropna().unique().tolist()),
            sentiments=["positive", "neutral", "negative"],
        )

    def get_brand_comparison(self, *, aspect: str | None, limit: int) -> BrandComparisonResponse:
        self._sync_runtime_delta()
        frame = self.brand_aspect_summary.copy()
        if aspect:
            frame = frame[frame["aspect"].str.lower() == aspect.lower()]
        frame = frame.sort_values(
            ["positive_rate", "mention_count", "avg_rating"],
            ascending=[False, False, False],
        ).head(limit)
        return BrandComparisonResponse(
            aspect=aspect,
            items=[
                BrandComparisonRecord(
                    brand=str(row.brand),
                    aspect=str(row.aspect),
                    mention_count=int(row.mention_count),
                    avg_rating=round(float(row.avg_rating), 2),
                    positive_rate=round(float(row.positive_rate), 2),
                    neutral_rate=round(float(row.neutral_rate), 2),
                    negative_rate=round(float(row.negative_rate), 2),
                )
                for row in frame.itertuples(index=False)
            ],
        )

    def get_aspect_summary(self, *, limit: int) -> AspectSummaryResponse:
        self._sync_runtime_delta()
        frame = self.aspect_summary.sort_values(
            ["mention_count", "positive_rate"],
            ascending=[False, False],
        ).head(limit)
        return AspectSummaryResponse(
            items=[
                AspectMetric(
                    aspect=str(row.aspect),
                    mention_count=int(row.mention_count),
                    positive_rate=round(float(row.positive_rate), 2),
                    neutral_rate=round(float(row.neutral_rate), 2),
                    negative_rate=round(float(row.negative_rate), 2),
                )
                for row in frame.itertuples(index=False)
            ]
        )

    def search_reviews(
        self,
        *,
        query: str | None,
        brand: str | None,
        aspect: str | None,
        sentiment: str | None,
        limit: int,
    ) -> ReviewsResponse:
        self._sync_runtime_delta()
        if query:
            items = self.semantic_search(
                query=query,
                top_k=max(limit, self.settings.default_top_k),
                brand=brand,
                aspect=aspect,
                sentiment=sentiment,
            )
            return ReviewsResponse(total=len(items), items=items[:limit])

        frame = self._filter_reviews(self.processed_reviews.copy(), brand, aspect, sentiment)
        total = int(len(frame))
        return ReviewsResponse(total=total, items=self._records_from_frame(frame.head(limit)))

    def analyze_review(self, request: AnalyzeReviewRequest) -> AnalyzeReviewResponse:
        self._sync_runtime_delta()
        return self._analyze_text_input(
            review_text=request.review,
            product_name=request.product_name,
            top_k=request.top_k,
        )

    def analyze_content(
        self,
        *,
        review_text: str,
        product_name: str | None,
        top_k: int,
        forced_sentiment: SentimentLabel | None = None,
    ) -> AnalyzeReviewResponse:
        self._sync_runtime_delta()
        return self._analyze_text_input(
            review_text=review_text,
            product_name=product_name,
            top_k=top_k,
            forced_sentiment=forced_sentiment,
        )

    def generate_insight(self, request: InsightRequest) -> InsightResponse:
        self._sync_runtime_delta()
        inferred_aspect = request.aspect or self._infer_query_aspect(request.query)
        retrieved_reviews = self.semantic_search(
            query=request.query,
            top_k=request.top_k,
            brand=request.brand,
            aspect=inferred_aspect,
        )
        aggregate_stats = self._aggregate_retrieved(retrieved_reviews, inferred_aspect)
        aggregate_stats["aspect_ranking"] = self._brand_ranking_for_aspect(inferred_aspect)

        if request.use_gemini and self.gemini_client.enabled and retrieved_reviews:
            try:
                payload = self.gemini_client.generate_business_insight(
                    query=request.query,
                    filters={"brand": request.brand, "aspect": inferred_aspect},
                    aggregate_stats=aggregate_stats,
                    retrieved_reviews=[item.model_dump() for item in retrieved_reviews],
                )
                answer = str(payload.get("answer", "")).strip()
                recommended_actions = [
                    str(item).strip()
                    for item in payload.get("recommended_actions", [])[:3]
                    if str(item).strip()
                ]
                dominant_sentiment = str(
                    payload.get("dominant_sentiment", aggregate_stats["dominant_sentiment"])
                )
                mode = "gemini"
            except Exception:
                answer, recommended_actions = self._template_answer(
                    request.query,
                    aggregate_stats,
                    request.brand,
                    inferred_aspect,
                )
                dominant_sentiment = str(aggregate_stats["dominant_sentiment"])
                mode = "template"
        else:
            answer, recommended_actions = self._template_answer(
                request.query,
                aggregate_stats,
                request.brand,
                inferred_aspect,
            )
            dominant_sentiment = str(aggregate_stats["dominant_sentiment"])
            mode = "template"

        return InsightResponse(
            query=request.query,
            answer=answer,
            recommended_actions=recommended_actions,
            dominant_sentiment=dominant_sentiment,
            brands_mentioned=list(aggregate_stats["brands_mentioned"]),
            aspects_mentioned=list(aggregate_stats["aspects_mentioned"]),
            retrieved_reviews=retrieved_reviews,
            mode=mode,  # type: ignore[arg-type]
        )

    def list_powerbi_exports(self) -> ExportListResponse:
        self._sync_runtime_delta()
        export_dir = Path(self.settings.powerbi_export_dir)
        files = [
            ExportFile(name=path.name, path=str(path.resolve()))
            for path in sorted(export_dir.glob("*.csv"))
        ]
        return ExportListResponse(files=files)

    def build_review_create(
        self,
        *,
        product_name: str | None,
        review_text: str,
        predicted_sentiment: SentimentLabel,
        aspects: list[str],
        source: str,
        rating: float | None = None,
        language: str | None = None,
    ) -> ReviewCreate:
        product = product_name.strip() if product_name and product_name.strip() else "User Submitted Review"
        return ReviewCreate(
            product=product,
            brand=extract_brand(product),
            review_text=review_text,
            sentiment=predicted_sentiment,
            aspect=normalize_aspect_string(aspects),
            language=language or detect_language(review_text),
            source=source,  # type: ignore[arg-type]
            rating=rating,
        )

    def semantic_search(
        self,
        *,
        query: str,
        top_k: int,
        brand: str | None = None,
        aspect: str | None = None,
        sentiment: str | None = None,
    ) -> list[ReviewRecord]:
        self._sync_runtime_delta()
        if not query.strip() or self.rag_metadata.empty:
            return []

        query_vector = self.rag_vectorizer.transform([query])
        similarities = linear_kernel(query_vector, self.rag_matrix).ravel()
        best_indices = np.argsort(similarities)[::-1][: min(len(self.rag_metadata), max(top_k * 12, 60))]

        results: list[ReviewRecord] = []
        seen_review_ids: set[int] = set()
        for idx in best_indices:
            row = self.rag_metadata.iloc[int(idx)]
            review_id = int(row.review_id)
            if review_id in seen_review_ids:
                continue
            aspects = split_aspects(row.aspects)
            if brand and str(row.brand).lower() != brand.lower():
                continue
            if aspect and aspect.lower() not in [item.lower() for item in aspects]:
                continue
            if sentiment and str(row.sentiment).lower() != sentiment.lower():
                continue
            results.append(
                ReviewRecord(
                    review_id=review_id,
                    brand=str(row.brand),
                    product_name=str(row.product_name),
                    review=str(row.review),
                    rating=int(row.rating),
                    sentiment=str(row.sentiment),  # type: ignore[arg-type]
                    aspects=aspects,
                    similarity=round(float(similarities[int(idx)]), 4),
                )
            )
            seen_review_ids.add(review_id)
            if len(results) >= top_k:
                break
        return results

    def ingest_review(self, review: Review) -> None:
        if int(review.id) in self._known_review_ids:
            return

        row = self._review_to_runtime_row(review)
        self.processed_reviews = pd.concat([self.processed_reviews, pd.DataFrame([row])], ignore_index=True)
        self.processed_reviews["review_id"] = self.processed_reviews["review_id"].astype(int)
        self.processed_reviews["rating"] = self.processed_reviews["rating"].astype(int)
        self.processed_reviews["aspects_list"] = self.processed_reviews["aspects"].apply(split_aspects)

        rag_row = {
            "review_id": row["review_id"],
            "brand": row["brand"],
            "product_name": row["product_name"],
            "review": row["review"],
            "rating": row["rating"],
            "sentiment": row["sentiment"],
            "aspects": row["aspects"],
            "rag_text": row["rag_text"],
            "language": row["language"],
            "source": row["source"],
            "created_at": row["created_at"],
            "aspects_list": split_aspects(str(row["aspects"])),
        }
        self.rag_metadata = pd.concat([self.rag_metadata, pd.DataFrame([rag_row])], ignore_index=True)
        self.rag_matrix = sparse.vstack(
            [self.rag_matrix, self.rag_vectorizer.transform([str(row["rag_text"])])],
            format="csr",
        )

        self.aspect_mentions = pd.concat(
            [self.aspect_mentions, pd.DataFrame(self._build_aspect_rows(row))],
            ignore_index=True,
        )
        if not self.aspect_mentions.empty:
            self.aspect_mentions["review_id"] = self.aspect_mentions["review_id"].astype(int)
        self._known_review_ids.add(int(review.id))
        self._recompute_aggregates()
        self._write_runtime_exports()

    def _analyze_text_input(
        self,
        *,
        review_text: str,
        product_name: str | None,
        top_k: int,
        forced_sentiment: SentimentLabel | None = None,
    ) -> AnalyzeReviewResponse:
        normalized_review = preprocess_review(review_text)
        features = self.vectorizer.transform([normalized_review])
        probabilities = self.model.predict_proba(features)[0]
        predicted_sentiment = str(forced_sentiment or self.model.predict(features)[0])
        confidence = float(np.max(probabilities))
        aspects = extract_aspects(normalized_review)
        brand = extract_brand(product_name) if product_name else None
        similar_reviews = self.semantic_search(
            query=f"{product_name or ''} {review_text}".strip(),
            top_k=top_k,
            brand=brand,
        )
        summary = (
            f"The review is mostly {predicted_sentiment} with strongest discussion around "
            f"{', '.join(aspects[:4])}. Confidence: {confidence:.2f}. "
            f"{'Inferred brand: ' + brand + '. ' if brand else ''}"
            f"Retrieved {len(similar_reviews)} similar historical reviews for context."
        )
        return AnalyzeReviewResponse(
            normalized_review=normalized_review,
            predicted_sentiment=predicted_sentiment,  # type: ignore[arg-type]
            confidence=confidence,
            brand=brand,
            aspects=aspects,
            summary=summary,
            similar_reviews=similar_reviews,
        )

    def _sync_runtime_delta(self) -> None:
        with SessionLocal() as db:
            latest_review_id = db.scalar(select(func.max(Review.id))) or 0
            current_max = max(self._known_review_ids, default=0)
            if latest_review_id <= current_max:
                return
            missing_reviews = db.scalars(
                select(Review).where(Review.id > current_max).order_by(Review.id.asc())
            ).all()
        for review in missing_reviews:
            self.ingest_review(review)

    def _review_to_runtime_row(self, review: Review) -> dict[str, object]:
        normalized_review = preprocess_review(review.review_text)
        aspects = split_aspects(review.aspect) or extract_aspects(normalized_review)
        sentiment = str(review.sentiment).lower()
        rating = int(round(float(review.rating))) if review.rating is not None else SENTIMENT_TO_RATING.get(sentiment, 3)
        features = self.vectorizer.transform([normalized_review])
        probabilities = self.model.predict_proba(features)[0]
        classes = list(self.model.classes_)
        return {
            "review_id": int(review.id),
            "product_name": review.product,
            "brand": review.brand or extract_brand(review.product),
            "review": review.review_text,
            "rating": rating,
            "sentiment": sentiment,
            "predicted_sentiment": str(self.model.predict(features)[0]),
            "prediction_confidence": float(np.max(probabilities)),
            "normalized_review": normalized_review,
            "aspects": join_aspects(aspects),
            "rag_text": build_rag_text(
                brand=review.brand or extract_brand(review.product),
                product_name=review.product,
                review=review.review_text,
                sentiment=sentiment,
                aspects=aspects,
                rating=rating,
            ),
            "negative_score": self._score_for_class(probabilities, classes, "negative"),
            "neutral_score": self._score_for_class(probabilities, classes, "neutral"),
            "positive_score": self._score_for_class(probabilities, classes, "positive"),
            "language": review.language or detect_language(review.review_text),
            "source": review.source,
            "created_at": review.created_at.isoformat() if review.created_at else "",
        }

    def _build_aspect_rows(self, row: dict[str, object]) -> list[dict[str, object]]:
        aspects = split_aspects(str(row["aspects"])) or ["generic"]
        contexts = build_aspect_contexts(
            review_text=str(row["review"]),
            normalized_review=str(row["normalized_review"]),
            aspects=aspects,
        )
        items: list[dict[str, object]] = []
        for aspect, context in contexts.items():
            features = self.vectorizer.transform([preprocess_review(context)])
            probabilities = self.model.predict_proba(features)[0]
            items.append(
                {
                    "review_id": int(row["review_id"]),
                    "brand": row["brand"],
                    "product_name": row["product_name"],
                    "rating": int(row["rating"]),
                    "review_sentiment": row["sentiment"],
                    "aspect": aspect,
                    "aspect_text": context,
                    "normalized_aspect_text": preprocess_review(context),
                    "aspect_sentiment": str(self.model.predict(features)[0]),
                    "confidence": float(np.max(probabilities)),
                }
            )
        return items

    def _recompute_aggregates(self) -> None:
        self.brand_overview = pd.DataFrame()
        self.brand_aspect_summary = pd.DataFrame()
        self.aspect_summary = pd.DataFrame()
        if self.processed_reviews.empty:
            return

        self.brand_overview = (
            self.processed_reviews.groupby("brand")
            .agg(
                review_count=("review_id", "count"),
                avg_rating=("rating", "mean"),
                positive_count=("sentiment", lambda series: int((series == "positive").sum())),
                neutral_count=("sentiment", lambda series: int((series == "neutral").sum())),
                negative_count=("sentiment", lambda series: int((series == "negative").sum())),
            )
            .reset_index()
        )
        for label in ("positive", "neutral", "negative"):
            self.brand_overview[f"{label}_rate"] = (
                self.brand_overview[f"{label}_count"] / self.brand_overview["review_count"] * 100
            )
        self.brand_overview = self.brand_overview.sort_values(
            ["positive_rate", "avg_rating", "review_count"],
            ascending=[False, False, False],
        )

        if self.aspect_mentions.empty:
            return

        self.brand_aspect_summary = (
            self.aspect_mentions.groupby(["brand", "aspect"])
            .agg(
                mention_count=("review_id", "count"),
                avg_rating=("rating", "mean"),
                positive_count=("aspect_sentiment", lambda series: int((series == "positive").sum())),
                neutral_count=("aspect_sentiment", lambda series: int((series == "neutral").sum())),
                negative_count=("aspect_sentiment", lambda series: int((series == "negative").sum())),
            )
            .reset_index()
        )
        for label in ("positive", "neutral", "negative"):
            self.brand_aspect_summary[f"{label}_rate"] = (
                self.brand_aspect_summary[f"{label}_count"]
                / self.brand_aspect_summary["mention_count"]
                * 100
            )

        self.aspect_summary = (
            self.aspect_mentions.groupby("aspect")
            .agg(
                mention_count=("review_id", "count"),
                positive_count=("aspect_sentiment", lambda series: int((series == "positive").sum())),
                neutral_count=("aspect_sentiment", lambda series: int((series == "neutral").sum())),
                negative_count=("aspect_sentiment", lambda series: int((series == "negative").sum())),
            )
            .reset_index()
        )
        for label in ("positive", "neutral", "negative"):
            self.aspect_summary[f"{label}_rate"] = (
                self.aspect_summary[f"{label}_count"] / self.aspect_summary["mention_count"] * 100
            )
        self.aspect_summary = self.aspect_summary.sort_values(
            ["mention_count", "positive_rate"],
            ascending=[False, False],
        )

    def _write_runtime_exports(self) -> None:
        export_dir = Path(self.settings.powerbi_export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        self.processed_reviews[
            ["review_id", "product_name", "brand", "review", "sentiment", "language", "source", "rating", "created_at"]
        ].to_csv(export_dir / "reviews.csv", index=False)
        (
            self.processed_reviews.groupby("sentiment")
            .agg(review_count=("review_id", "count"))
            .reset_index()
        ).to_csv(export_dir / "sentiment_summary.csv", index=False)
        (
            self.processed_reviews.groupby("language")
            .agg(review_count=("review_id", "count"))
            .reset_index()
        ).to_csv(export_dir / "language_distribution.csv", index=False)
        self.brand_overview.to_csv(export_dir / "brand_overview.csv", index=False)
        self.brand_aspect_summary.to_csv(export_dir / "brand_aspect_summary.csv", index=False)
        self.aspect_summary.to_csv(export_dir / "aspect_summary.csv", index=False)
        self.processed_reviews.drop(columns=["aspects_list"], errors="ignore").to_csv(
            export_dir / "processed_reviews.csv",
            index=False,
        )

    def _filter_reviews(
        self,
        frame: pd.DataFrame,
        brand: str | None,
        aspect: str | None,
        sentiment: str | None,
    ) -> pd.DataFrame:
        if brand:
            frame = frame[frame["brand"].str.lower() == brand.lower()]
        if aspect:
            frame = frame[
                frame["aspects_list"].apply(
                    lambda items: aspect.lower() in [item.lower() for item in items]
                )
            ]
        if sentiment:
            frame = frame[frame["sentiment"].str.lower() == sentiment.lower()]
        return frame

    def _records_from_frame(self, frame: pd.DataFrame) -> list[ReviewRecord]:
        return [
            ReviewRecord(
                review_id=int(row.review_id),
                brand=str(row.brand),
                product_name=str(row.product_name),
                review=str(row.review),
                rating=int(row.rating),
                sentiment=str(row.sentiment),  # type: ignore[arg-type]
                aspects=list(row.aspects_list),
                similarity=round(float(getattr(row, "similarity")), 4)
                if getattr(row, "similarity", None) is not None
                else None,
            )
            for row in frame.itertuples(index=False)
        ]

    def _aggregate_retrieved(self, reviews: list[ReviewRecord], requested_aspect: str | None) -> dict[str, object]:
        if not reviews:
            return {
                "review_count": 0,
                "avg_rating": 0.0,
                "brands_mentioned": [],
                "aspects_mentioned": [],
                "dominant_sentiment": "neutral",
                "positive_rate": 0.0,
                "negative_rate": 0.0,
                "top_negative_aspects": [],
            }
        review_ids = [item.review_id for item in reviews]
        subset = self.aspect_mentions[self.aspect_mentions["review_id"].isin(review_ids)].copy()
        if requested_aspect:
            subset = subset[subset["aspect"].str.lower() == requested_aspect.lower()]
        sentiments = Counter(item.sentiment for item in reviews)
        aspects_counter = Counter(aspect for item in reviews for aspect in item.aspects)
        brands_counter = Counter(item.brand for item in reviews)
        negative_aspects = []
        if not subset.empty:
            negative_aspects = (
                subset[subset["aspect_sentiment"] == "negative"]["aspect"].value_counts().head(3).index.tolist()
            )
        return {
            "review_count": len(reviews),
            "avg_rating": round(float(np.mean([item.rating for item in reviews])), 2),
            "brands_mentioned": [brand for brand, _ in brands_counter.most_common(5)],
            "aspects_mentioned": [aspect for aspect, _ in aspects_counter.most_common(6)],
            "dominant_sentiment": sentiments.most_common(1)[0][0],
            "positive_rate": round(float(sentiments.get("positive", 0) / len(reviews) * 100), 2),
            "negative_rate": round(float(sentiments.get("negative", 0) / len(reviews) * 100), 2),
            "top_negative_aspects": negative_aspects,
        }

    def _infer_query_aspect(self, query: str) -> str | None:
        aspects = [aspect for aspect in extract_aspects(preprocess_review(query)) if aspect != "generic"]
        return aspects[0] if aspects else None

    def _brand_ranking_for_aspect(self, aspect: str | None) -> list[dict[str, float | int | str]]:
        if not aspect or self.brand_aspect_summary.empty:
            return []
        frame = self.brand_aspect_summary[
            self.brand_aspect_summary["aspect"].str.lower() == aspect.lower()
        ].copy()
        frame = frame.sort_values(
            ["positive_rate", "mention_count", "avg_rating"],
            ascending=[False, False, False],
        )
        return [
            {
                "brand": str(row.brand),
                "positive_rate": round(float(row.positive_rate), 2),
                "mention_count": int(row.mention_count),
                "avg_rating": round(float(row.avg_rating), 2),
            }
            for row in frame.head(5).itertuples(index=False)
        ]

    def _template_answer(
        self,
        query: str,
        aggregate_stats: dict[str, object],
        brand: str | None,
        aspect: str | None,
    ) -> tuple[str, list[str]]:
        review_count = int(aggregate_stats.get("review_count", 0))
        if review_count == 0:
            return (
                "No matching reviews were retrieved for this query. Try a more specific brand or aspect.",
                [
                    "Use a clearer product aspect such as battery, display, or performance.",
                    "Add a brand filter to narrow the search space.",
                    "Refresh the artifact build if you recently added new reviews.",
                ],
            )
        avg_rating = float(aggregate_stats.get("avg_rating", 0.0))
        positive_rate = float(aggregate_stats.get("positive_rate", 0.0))
        negative_rate = float(aggregate_stats.get("negative_rate", 0.0))
        brands = aggregate_stats.get("brands_mentioned", [])
        aspects = aggregate_stats.get("aspects_mentioned", [])
        negative_aspects = aggregate_stats.get("top_negative_aspects", [])
        aspect_ranking = aggregate_stats.get("aspect_ranking", [])
        aspect_text = ", ".join(aspects[:4]) if aspects else "overall laptop experience"
        brand_text = ", ".join(brands[:3]) if brands else "multiple brands"
        ranking_text = ""
        if aspect_ranking:
            leaders = ", ".join(
                f"{item['brand']} ({item['positive_rate']:.1f}% positive)"
                for item in aspect_ranking[:3]
            )
            ranking_text = (
                f" In the full dataset, the strongest brands for {aspect or aspect_text.split(',')[0]} "
                f"are {leaders}."
            )
        answer = (
            f"For '{query}', the retrieved review set covers {review_count} similar reviews with an average rating "
            f"of {avg_rating:.2f}/5. Discussions are strongest around {aspect_text}, and the dominant evidence "
            f"comes from {brand_text}. Positive sentiment is {positive_rate:.1f}% versus {negative_rate:.1f}% "
            f"negative{' for ' + brand if brand else ''}{' on ' + aspect if aspect else ''}.{ranking_text}"
        )
        actions = [
            f"Promote the strongest proof points around {aspect or aspect_text.split(',')[0]} in product messaging.",
            (
                f"Investigate recurring negative signals in {', '.join(negative_aspects[:3])}."
                if negative_aspects
                else "Review the lowest-rated retrieved comments for recurring complaints and support gaps."
            ),
            "Use the exported Power BI tables to track brand and aspect movement after each dataset refresh.",
        ]
        return answer, actions

    @staticmethod
    def _score_for_class(probabilities: np.ndarray, classes: list[str], label: str) -> float:
        return float(probabilities[classes.index(label)]) if label in classes else 0.0
