from __future__ import annotations

import json
from typing import Any

from backend.app.config import Settings


class GeminiClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._enabled = bool(settings.google_api_key) and not settings.enable_fake_mode

        if self._enabled:
            try:
                import google.generativeai as genai

                genai.configure(api_key=settings.google_api_key)
                self._model = genai.GenerativeModel(settings.gemini_model)
            except Exception:
                self._enabled = False
                self._model = None

    @property
    def enabled(self) -> bool:
        return self._enabled and self._model is not None

    def generate_business_insight(
        self,
        *,
        query: str,
        filters: dict[str, Any],
        aggregate_stats: dict[str, Any],
        retrieved_reviews: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("Gemini is not enabled.")

        prompt = {
            "task": (
                "Act as a product intelligence analyst for laptop brands. "
                "Use the multilingual reviews to answer the business question."
            ),
            "user_query": query,
            "filters": filters,
            "aggregate_stats": aggregate_stats,
            "retrieved_reviews": retrieved_reviews[:8],
            "response_rules": {
                "answer": "One concise paragraph with evidence-backed insight.",
                "recommended_actions": "Three short actionable recommendations.",
                "dominant_sentiment": ["positive", "neutral", "negative"],
            },
            "output": "Return strict JSON only with keys: answer, recommended_actions, dominant_sentiment.",
        }

        response = self._model.generate_content(json.dumps(prompt))
        raw_text = getattr(response, "text", "").strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            raw_text = raw_text.replace("json", "", 1).strip()
        return json.loads(raw_text)
