from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel

from backend.app.config import Settings
from backend.app.models import (
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
)
from backend.app.services.artifact_builder import ArtifactBuilder
from backend.app.services.gemini_client import GeminiClient
from backend.app.services.preprocessing import (
    extract_aspects,
    extract_brand,
    preprocess_review,
    split_aspects,
)


class ReviewIntelligenceService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.artifact_builder = ArtifactBuilder(settings)
        self.artifact_builder.ensure()
        self.artifacts_ready = self.artifact_builder.artifacts_ready()

        artifact_dir = Path(settings.artifact_dir)
        self.processed_reviews = pd.read_csv(artifact_dir / "processed_reviews.csv")
        self.aspect_mentions = pd.read_csv(artifact_dir / "aspect_mentions.csv")
        self.brand_overview = pd.read_csv(artifact_dir / "brand_overview.csv")
        self.brand_aspect_summary = pd.read_csv(artifact_dir / "brand_aspect_summary.csv")
        self.aspect_summary = pd.read_csv(artifact_dir / "aspect_summary.csv")
        self.rag_metadata = pd.read_csv(artifact_dir / "rag_metadata.csv")
        self.vectorizer = joblib.load(artifact_dir / "sentiment_vectorizer.joblib")
        self.model = joblib.load(artifact_dir / "sentiment_model.joblib")
        self.rag_vectorizer = joblib.load(artifact_dir / "rag_vectorizer.joblib")
        self.rag_matrix = sparse.load_npz(artifact_dir / "rag_matrix.npz")
        self.metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
        self.gemini_client = GeminiClient(settings)

        self.processed_reviews["aspects_list"] = self.processed_reviews["aspects"].apply(split_aspects)
        self.rag_metadata["aspects_list"] = self.rag_metadata["aspects"].apply(split_aspects)
        self.aspect_mentions["review_id"] = self.aspect_mentions["review_id"].astype(int)
        self.retrieval_backend = "tfidf"

    def get_overview(self) -> OverviewResponse:
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

        sentiment_counts = self.processed_reviews["sentiment"].value_counts()
        total_reviews = int(len(self.processed_reviews))

        return OverviewResponse(
            total_reviews=total_reviews,
            total_brands=int(self.processed_reviews["brand"].nunique()),
            avg_rating=round(float(self.processed_reviews["rating"].mean()), 2),
            positive_rate=round(float(sentiment_counts.get("positive", 0) / total_reviews * 100), 2),
            neutral_rate=round(float(sentiment_counts.get("neutral", 0) / total_reviews * 100), 2),
            negative_rate=round(float(sentiment_counts.get("negative", 0) / total_reviews * 100), 2),
            model_accuracy=self.metrics.get("accuracy"),
            top_brands=top_brands,
            top_aspects=top_aspects,
        )

    def get_filters(self) -> FiltersResponse:
        brands = sorted(self.processed_reviews["brand"].dropna().unique().tolist())
        aspects = sorted(self.aspect_summary["aspect"].dropna().unique().tolist())
        return FiltersResponse(
            brands=brands,
            aspects=aspects,
            sentiments=["positive", "neutral", "negative"],
        )

    def get_brand_comparison(self, *, aspect: str | None, limit: int) -> BrandComparisonResponse:
        if aspect:
            frame = self.brand_aspect_summary[
                self.brand_aspect_summary["aspect"].str.lower() == aspect.lower()
            ].copy()
        else:
            frame = self.brand_aspect_summary.copy()

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
        frame = frame.head(limit)
        return ReviewsResponse(total=total, items=self._records_from_frame(frame))

    def analyze_review(self, request: AnalyzeReviewRequest) -> AnalyzeReviewResponse:
        normalized_review = preprocess_review(request.review)
        features = self.vectorizer.transform([normalized_review])
        probabilities = self.model.predict_proba(features)[0]
        predicted_sentiment = str(self.model.predict(features)[0])
        confidence = float(np.max(probabilities))
        aspects = extract_aspects(normalized_review)
        brand = extract_brand(request.product_name) if request.product_name else None
        similar_reviews = self.semantic_search(
            query=f"{request.product_name or ''} {request.review}".strip(),
            top_k=request.top_k,
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

    def generate_insight(self, request: InsightRequest) -> InsightResponse:
        inferred_aspect = request.aspect or self._infer_query_aspect(request.query)
        retrieved_reviews = self.semantic_search(
            query=request.query,
            top_k=request.top_k,
            brand=request.brand,
            aspect=inferred_aspect,
        )
        aggregate_stats = self._aggregate_retrieved(retrieved_reviews, inferred_aspect)
        aggregate_stats["aspect_ranking"] = self._brand_ranking_for_aspect(inferred_aspect)
        brands_mentioned = aggregate_stats["brands_mentioned"]
        aspects_mentioned = aggregate_stats["aspects_mentioned"]
        dominant_sentiment = aggregate_stats["dominant_sentiment"]

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
                mode = "gemini"
                dominant_sentiment = str(payload.get("dominant_sentiment", dominant_sentiment))
            except Exception:
                answer, recommended_actions = self._template_answer(
                    request.query,
                    aggregate_stats,
                    request.brand,
                    inferred_aspect,
                )
                mode = "template"
        else:
            answer, recommended_actions = self._template_answer(
                request.query,
                aggregate_stats,
                request.brand,
                inferred_aspect,
            )
            mode = "template"

        return InsightResponse(
            query=request.query,
            answer=answer,
            recommended_actions=recommended_actions,
            dominant_sentiment=dominant_sentiment,
            brands_mentioned=brands_mentioned,
            aspects_mentioned=aspects_mentioned,
            retrieved_reviews=retrieved_reviews,
            mode=mode,  # type: ignore[arg-type]
        )

    def list_powerbi_exports(self) -> ExportListResponse:
        export_dir = Path(self.settings.powerbi_export_dir)
        files = [
            ExportFile(name=path.name, path=str(path.resolve()))
            for path in sorted(export_dir.glob("*.csv"))
        ]
        return ExportListResponse(files=files)

    def semantic_search(
        self,
        *,
        query: str,
        top_k: int,
        brand: str | None = None,
        aspect: str | None = None,
        sentiment: str | None = None,
    ) -> list[ReviewRecord]:
        if not query.strip():
            return []

        top_search = min(len(self.rag_metadata), max(top_k * 12, 60))
        query_vector = self.rag_vectorizer.transform([query])
        similarities = linear_kernel(query_vector, self.rag_matrix).ravel()
        best_indices = np.argsort(similarities)[::-1][:top_search]
        candidates = [(int(idx), float(similarities[idx])) for idx in best_indices]

        results: list[ReviewRecord] = []
        seen_review_ids: set[int] = set()
        for idx, similarity in candidates:
            row = self.rag_metadata.iloc[idx]
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
                    similarity=round(similarity, 4),
                )
            )
            seen_review_ids.add(review_id)
            if len(results) >= top_k:
                break

        return results

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
        items: list[ReviewRecord] = []
        for row in frame.itertuples(index=False):
            similarity = getattr(row, "similarity", None)
            items.append(
                ReviewRecord(
                    review_id=int(row.review_id),
                    brand=str(row.brand),
                    product_name=str(row.product_name),
                    review=str(row.review),
                    rating=int(row.rating),
                    sentiment=str(row.sentiment),  # type: ignore[arg-type]
                    aspects=list(row.aspects_list),
                    similarity=round(float(similarity), 4) if similarity is not None else None,
                )
            )
        return items

    def _aggregate_retrieved(
        self,
        reviews: list[ReviewRecord],
        requested_aspect: str | None,
    ) -> dict[str, object]:
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
        aspects_counter = Counter(
            aspect
            for item in reviews
            for aspect in item.aspects
        )
        brand_counter = Counter(item.brand for item in reviews)
        dominant_sentiment = sentiments.most_common(1)[0][0]

        top_negative_aspects: list[str] = []
        if not subset.empty:
            negative_subset = subset[subset["aspect_sentiment"] == "negative"]
            top_negative_aspects = negative_subset["aspect"].value_counts().head(3).index.tolist()

        return {
            "review_count": len(reviews),
            "avg_rating": round(float(np.mean([item.rating for item in reviews])), 2),
            "brands_mentioned": [brand for brand, _ in brand_counter.most_common(5)],
            "aspects_mentioned": [aspect for aspect, _ in aspects_counter.most_common(6)],
            "dominant_sentiment": dominant_sentiment,
            "positive_rate": round(float(sentiments.get("positive", 0) / len(reviews) * 100), 2),
            "negative_rate": round(float(sentiments.get("negative", 0) / len(reviews) * 100), 2),
            "top_negative_aspects": top_negative_aspects,
        }

    def _infer_query_aspect(self, query: str) -> str | None:
        aspects = [
            aspect
            for aspect in extract_aspects(preprocess_review(query))
            if aspect != "generic"
        ]
        return aspects[0] if aspects else None

    def _brand_ranking_for_aspect(self, aspect: str | None) -> list[dict[str, float | int | str]]:
        if not aspect:
            return []

        frame = self.brand_aspect_summary[
            self.brand_aspect_summary["aspect"].str.lower() == aspect.lower()
        ].copy()
        if frame.empty:
            return []

        frame = frame[frame["mention_count"] >= 15].sort_values(
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

        brand_text = ", ".join(brands[:3]) if brands else "multiple brands"
        aspect_text = ", ".join(aspects[:4]) if aspects else "overall laptop experience"
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
