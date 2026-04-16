from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


def get_api_base_url() -> str:
    if os.getenv("API_BASE_URL"):
        return os.getenv("API_BASE_URL", "http://localhost:8000")
    try:
        return str(st.secrets["API_BASE_URL"])
    except Exception:
        return "http://localhost:8000"


API_BASE_URL = get_api_base_url()

st.set_page_config(
    page_title="Laptop Review Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Multilingual Laptop Review Intelligence")
st.caption("Tanglish + English laptop review analytics, aspect comparison, and RAG-powered insights.")

PAGES = [
    "Dashboard",
    "RAG Insights",
    "Analyze Review",
    "Review Explorer",
    "Power BI Exports",
]

if "active_page" not in st.session_state:
    st.session_state.active_page = "Dashboard"
if "insight_result" not in st.session_state:
    st.session_state.insight_result = None
if "analyze_result" not in st.session_state:
    st.session_state.analyze_result = None


@st.cache_data(ttl=60)
def fetch_json(path: str, params: dict | None = None) -> dict:
    clean_params = {
        key: value
        for key, value in (params or {}).items()
        if value is not None and value != ""
    }
    response = requests.get(f"{API_BASE_URL}{path}", params=clean_params, timeout=120)
    response.raise_for_status()
    return response.json()


def post_json(path: str, payload: dict) -> dict:
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=180)
    response.raise_for_status()
    return response.json()


with st.sidebar:
    st.subheader("Backend")
    st.code(API_BASE_URL)
    if st.button("Check API health", use_container_width=True):
        try:
            st.success(fetch_json("/health"))
        except Exception as exc:
            st.error(f"Backend unavailable: {exc}")

try:
    overview = fetch_json("/api/v1/overview")
    filters = fetch_json("/api/v1/filters")
    aspect_summary = fetch_json("/api/v1/aspect-summary", params={"limit": 20})
    brand_comparison_default = fetch_json("/api/v1/brand-comparison", params={"limit": 15})
except Exception as exc:
    st.error(f"Could not load backend data from {API_BASE_URL}: {exc}")
    st.stop()

metric_cols = st.columns(5)
metric_cols[0].metric("Reviews", f"{overview['total_reviews']:,}")
metric_cols[1].metric("Brands", overview["total_brands"])
metric_cols[2].metric("Avg Rating", overview["avg_rating"])
metric_cols[3].metric("Positive %", overview["positive_rate"])
metric_cols[4].metric("Model Accuracy", round(float(overview["model_accuracy"] or 0), 4))

selected_page = st.radio(
    "Navigation",
    options=PAGES,
    key="active_page",
    horizontal=True,
    label_visibility="collapsed",
)

if selected_page == "Dashboard":
    left, right = st.columns(2)

    brand_df = pd.DataFrame(overview["top_brands"])
    if not brand_df.empty:
        fig = px.bar(
            brand_df,
            x="brand",
            y=["positive_rate", "negative_rate"],
            barmode="group",
            title="Top Brand Sentiment Rates",
        )
        left.plotly_chart(fig, use_container_width=True)

    aspect_df = pd.DataFrame(aspect_summary["items"])
    if not aspect_df.empty:
        fig = px.bar(
            aspect_df,
            x="aspect",
            y="mention_count",
            color="positive_rate",
            title="Most Discussed Laptop Aspects",
        )
        right.plotly_chart(fig, use_container_width=True)

    st.subheader("Brand vs Aspect Comparison")
    selected_aspect = st.selectbox(
        "Choose an aspect to compare brands",
        options=["All"] + filters["aspects"],
        index=0,
    )
    params = {"limit": 15}
    if selected_aspect != "All":
        params["aspect"] = selected_aspect
    comparison = fetch_json("/api/v1/brand-comparison", params=params)
    comparison_df = pd.DataFrame(comparison["items"])
    if not comparison_df.empty:
        chart = px.scatter(
            comparison_df,
            x="avg_rating",
            y="positive_rate",
            size="mention_count",
            color="brand",
            hover_data=["aspect"],
            title="Brand Comparison by Aspect",
        )
        st.plotly_chart(chart, use_container_width=True)
        st.dataframe(comparison_df, use_container_width=True)
    elif brand_comparison_default["items"]:
        st.info("No rows matched the selected aspect.")

elif selected_page == "RAG Insights":
    st.subheader("Ask a Business Question")
    with st.form("insight-form"):
        query = st.text_input(
            "Question",
            placeholder="Which brand has the best battery life for developers?",
        )
        insight_cols = st.columns(3)
        brand = insight_cols[0].selectbox("Brand", options=["All"] + filters["brands"])
        aspect = insight_cols[1].selectbox("Aspect", options=["All"] + filters["aspects"])
        top_k = insight_cols[2].slider("Retrieved reviews", min_value=3, max_value=15, value=5)
        use_gemini = st.checkbox("Use Gemini when configured", value=True)
        submitted = st.form_submit_button("Generate Insight", type="primary")

    if submitted:
        payload = {
            "query": query,
            "brand": None if brand == "All" else brand,
            "aspect": None if aspect == "All" else aspect,
            "top_k": top_k,
            "use_gemini": use_gemini,
        }
        st.session_state.insight_result = post_json("/api/v1/insights", payload)

    if st.session_state.insight_result:
        result = st.session_state.insight_result
        st.success(f"Insight mode: {result['mode']}")
        st.write(result["answer"])
        st.markdown("### Recommended Actions")
        for action in result["recommended_actions"]:
            st.write(f"- {action}")

        st.markdown("### Retrieved Reviews")
        hits = pd.DataFrame(result["retrieved_reviews"])
        if not hits.empty:
            st.dataframe(hits, use_container_width=True)
        else:
            st.info("No reviews retrieved for this query.")

elif selected_page == "Analyze Review":
    st.subheader("Analyze a New Review")
    with st.form("analyze-form"):
        product_name = st.text_input(
            "Product name",
            placeholder="ASUS Vivobook 15 Intel Core i5 12th Gen...",
        )
        review_text = st.text_area(
            "Review text",
            placeholder="Battery romba nalla iruku but speaker quality is average...",
            height=180,
        )
        top_k = st.slider("Similar reviews", min_value=3, max_value=15, value=5, key="analyze-top-k")
        analyze_submitted = st.form_submit_button("Analyze Review", type="primary")

    if analyze_submitted:
        payload = {
            "product_name": product_name or None,
            "review": review_text,
            "top_k": top_k,
        }
        st.session_state.analyze_result = post_json("/api/v1/analyze-review", payload)

    if st.session_state.analyze_result:
        result = st.session_state.analyze_result
        st.write(result["summary"])
        stat_cols = st.columns(3)
        stat_cols[0].metric("Predicted Sentiment", result["predicted_sentiment"])
        stat_cols[1].metric("Confidence", round(float(result["confidence"]), 4))
        stat_cols[2].metric("Brand", result["brand"] or "Not inferred")
        st.write("Aspects:", ", ".join(result["aspects"]))
        similar_df = pd.DataFrame(result["similar_reviews"])
        if not similar_df.empty:
            st.dataframe(similar_df, use_container_width=True)

elif selected_page == "Review Explorer":
    st.subheader("Review Explorer")
    search_cols = st.columns(4)
    search_query = search_cols[0].text_input("Search query")
    review_brand = search_cols[1].selectbox("Brand filter", options=["All"] + filters["brands"])
    review_aspect = search_cols[2].selectbox("Aspect filter", options=["All"] + filters["aspects"])
    review_sentiment = search_cols[3].selectbox(
        "Sentiment filter",
        options=["All"] + filters["sentiments"],
    )

    params = {
        "limit": 25,
        "query": search_query or None,
        "brand": None if review_brand == "All" else review_brand,
        "aspect": None if review_aspect == "All" else review_aspect,
        "sentiment": None if review_sentiment == "All" else review_sentiment,
    }
    reviews = fetch_json("/api/v1/reviews", params=params)
    reviews_df = pd.DataFrame(reviews["items"])
    if not reviews_df.empty:
        st.dataframe(reviews_df, use_container_width=True)
    else:
        st.info("No reviews matched the current filters.")

else:
    st.subheader("Power BI Export Files")
    exports = fetch_json("/api/v1/powerbi/exports")
    export_df = pd.DataFrame(exports["files"])
    if not export_df.empty:
        st.dataframe(export_df, use_container_width=True)
        st.caption("Import these CSV files into Power BI to build dashboards and scheduled reports.")
    else:
        st.info("No Power BI exports found yet. Run the artifact build first.")
