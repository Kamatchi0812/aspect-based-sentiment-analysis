from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from backend.app.config import Settings
from backend.app.services.preprocessing import (
    build_aspect_contexts,
    build_rag_text,
    extract_aspects,
    extract_brand,
    join_aspects,
    preprocess_review,
    rating_to_sentiment,
)

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - platform dependent
    faiss = None


class ArtifactBuilder:
    REQUIRED_FILES = [
        "processed_reviews.csv",
        "aspect_mentions.csv",
        "brand_overview.csv",
        "brand_aspect_summary.csv",
        "aspect_summary.csv",
        "rag_metadata.csv",
        "review_embeddings.npy",
        "sentiment_vectorizer.joblib",
        "sentiment_model.joblib",
        "rag_vectorizer.joblib",
        "rag_matrix.npz",
        "metrics.json",
    ]

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.artifact_dir = Path(settings.artifact_dir)
        self.powerbi_dir = Path(settings.powerbi_export_dir)

    def artifacts_ready(self) -> bool:
        return all((self.artifact_dir / name).exists() for name in self.REQUIRED_FILES)

    def ensure(self, *, force: bool = False) -> None:
        if force or not self.artifacts_ready():
            self.build()

    def build(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.powerbi_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = Path(self.settings.raw_dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. "
                "Copy final_dataset.csv into data/raw or set RAW_DATASET_PATH."
            )

        data = pd.read_csv(dataset_path).fillna("")
        data = data.loc[:, ["product_name", "review", "rating"]].copy()
        data["review_id"] = np.arange(len(data))
        data["brand"] = data["product_name"].apply(extract_brand)
        data["normalized_review"] = data["review"].apply(preprocess_review)
        data["sentiment"] = data["rating"].apply(rating_to_sentiment)
        data["aspects_list"] = data["normalized_review"].apply(extract_aspects)
        data["aspects"] = data["aspects_list"].apply(join_aspects)
        data["rag_text"] = data.apply(
            lambda row: build_rag_text(
                brand=row["brand"],
                product_name=row["product_name"],
                review=row["review"],
                sentiment=row["sentiment"],
                aspects=row["aspects_list"],
                rating=int(row["rating"]),
            ),
            axis=1,
        )

        x_train, x_test, y_train, y_test = train_test_split(
            data["normalized_review"],
            data["sentiment"],
            test_size=0.2,
            random_state=42,
            stratify=data["sentiment"],
        )

        vectorizer = TfidfVectorizer(
            max_features=12000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        x_train_tfidf = vectorizer.fit_transform(x_train)
        x_test_tfidf = vectorizer.transform(x_test)

        model = LogisticRegression(
            max_iter=1200,
            class_weight="balanced",
        )
        model.fit(x_train_tfidf, y_train)

        y_pred = model.predict(x_test_tfidf)
        accuracy = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True)

        review_features = vectorizer.transform(data["normalized_review"])
        review_probabilities = model.predict_proba(review_features)
        classes = list(model.classes_)
        data["predicted_sentiment"] = model.predict(review_features)
        data["prediction_confidence"] = review_probabilities.max(axis=1)
        data["negative_score"] = review_probabilities[:, classes.index("negative")]
        data["neutral_score"] = review_probabilities[:, classes.index("neutral")]
        data["positive_score"] = review_probabilities[:, classes.index("positive")]

        aspect_rows: list[dict[str, object]] = []
        for row in data.itertuples(index=False):
            contexts = build_aspect_contexts(
                review_text=row.review,
                normalized_review=row.normalized_review,
                aspects=row.aspects_list,
            )
            for aspect, context in contexts.items():
                aspect_rows.append(
                    {
                        "review_id": int(row.review_id),
                        "brand": row.brand,
                        "product_name": row.product_name,
                        "rating": int(row.rating),
                        "review_sentiment": row.sentiment,
                        "aspect": aspect,
                        "aspect_text": context,
                        "normalized_aspect_text": preprocess_review(context),
                    }
                )

        aspect_mentions = pd.DataFrame(aspect_rows)
        aspect_features = vectorizer.transform(aspect_mentions["normalized_aspect_text"])
        aspect_probabilities = model.predict_proba(aspect_features)
        aspect_mentions["aspect_sentiment"] = model.predict(aspect_features)
        aspect_mentions["confidence"] = aspect_probabilities.max(axis=1)

        processed_reviews = data[
            [
                "review_id",
                "product_name",
                "brand",
                "review",
                "rating",
                "sentiment",
                "predicted_sentiment",
                "prediction_confidence",
                "normalized_review",
                "aspects",
                "rag_text",
                "negative_score",
                "neutral_score",
                "positive_score",
            ]
        ].copy()
        processed_reviews["rating"] = processed_reviews["rating"].astype(int)

        brand_overview = (
            processed_reviews.groupby("brand")
            .agg(
                review_count=("review_id", "count"),
                avg_rating=("rating", "mean"),
                positive_count=("sentiment", lambda series: int((series == "positive").sum())),
                neutral_count=("sentiment", lambda series: int((series == "neutral").sum())),
                negative_count=("sentiment", lambda series: int((series == "negative").sum())),
            )
            .reset_index()
        )
        brand_overview["positive_rate"] = (
            brand_overview["positive_count"] / brand_overview["review_count"] * 100
        )
        brand_overview["neutral_rate"] = (
            brand_overview["neutral_count"] / brand_overview["review_count"] * 100
        )
        brand_overview["negative_rate"] = (
            brand_overview["negative_count"] / brand_overview["review_count"] * 100
        )
        brand_overview = brand_overview.sort_values(
            ["positive_rate", "avg_rating", "review_count"],
            ascending=[False, False, False],
        )

        brand_aspect_summary = (
            aspect_mentions.groupby(["brand", "aspect"])
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
            brand_aspect_summary[f"{label}_rate"] = (
                brand_aspect_summary[f"{label}_count"]
                / brand_aspect_summary["mention_count"]
                * 100
            )

        aspect_summary = (
            aspect_mentions.groupby("aspect")
            .agg(
                mention_count=("review_id", "count"),
                positive_count=("aspect_sentiment", lambda series: int((series == "positive").sum())),
                neutral_count=("aspect_sentiment", lambda series: int((series == "neutral").sum())),
                negative_count=("aspect_sentiment", lambda series: int((series == "negative").sum())),
            )
            .reset_index()
        )
        for label in ("positive", "neutral", "negative"):
            aspect_summary[f"{label}_rate"] = (
                aspect_summary[f"{label}_count"] / aspect_summary["mention_count"] * 100
            )
        aspect_summary = aspect_summary.sort_values(
            ["mention_count", "positive_rate"],
            ascending=[False, False],
        )

        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(self.settings.embedding_model_name)
        embeddings = encoder.encode(
            processed_reviews["rag_text"].tolist(),
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        np.save(self.artifact_dir / "review_embeddings.npy", embeddings)
        rag_vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            dtype=np.float32,
        )
        rag_matrix = rag_vectorizer.fit_transform(processed_reviews["rag_text"])
        sparse.save_npz(self.artifact_dir / "rag_matrix.npz", rag_matrix)
        joblib.dump(rag_vectorizer, self.artifact_dir / "rag_vectorizer.joblib")
        rag_metadata = processed_reviews[
            ["review_id", "brand", "product_name", "review", "rating", "sentiment", "aspects", "rag_text"]
        ].copy()

        if faiss is not None:
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            faiss.write_index(index, str(self.artifact_dir / "reviews.index"))

        processed_reviews.to_csv(self.artifact_dir / "processed_reviews.csv", index=False)
        aspect_mentions.to_csv(self.artifact_dir / "aspect_mentions.csv", index=False)
        brand_overview.to_csv(self.artifact_dir / "brand_overview.csv", index=False)
        brand_aspect_summary.to_csv(self.artifact_dir / "brand_aspect_summary.csv", index=False)
        aspect_summary.to_csv(self.artifact_dir / "aspect_summary.csv", index=False)
        rag_metadata.to_csv(self.artifact_dir / "rag_metadata.csv", index=False)
        joblib.dump(vectorizer, self.artifact_dir / "sentiment_vectorizer.joblib")
        joblib.dump(model, self.artifact_dir / "sentiment_model.joblib")

        metrics = {
            "accuracy": accuracy,
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
            "train_size": int(len(x_train)),
            "test_size": int(len(x_test)),
            "embedding_model": self.settings.embedding_model_name,
            "retrieval_backend": "faiss" if faiss is not None else "cosine",
            "runtime_retrieval_backend": "tfidf",
            "dataset_rows": int(len(processed_reviews)),
        }
        (self.artifact_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

        powerbi_exports = {
            "brand_overview.csv": brand_overview,
            "brand_aspect_summary.csv": brand_aspect_summary,
            "aspect_summary.csv": aspect_summary,
            "processed_reviews.csv": processed_reviews,
        }
        for filename, frame in powerbi_exports.items():
            frame.to_csv(self.powerbi_dir / filename, index=False)
