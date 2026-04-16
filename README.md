# Multilingual Laptop Review Intelligence System

This project turns the attached laptop review dataset into a deployable analytics product for multilingual aspect-based sentiment analysis, semantic retrieval, and RAG-style business insight generation.

## What It Does

- Preprocesses Tanglish + English laptop reviews
- Extracts laptop-focused aspects such as battery, performance, display, price, design, keyboard, and delivery
- Trains a deployable sentiment model from the attached dataset
- Stores text, voice, and video reviews in a dynamic SQL database
- Supports SQLite for local development and PostgreSQL for production
- Builds semantic retrieval embeddings with Sentence Transformers
- Builds Sentence Transformer embeddings offline and uses a lightweight TF-IDF retrieval runtime so free-tier deployments start reliably
- Exposes a FastAPI backend for dashboards, semantic search, review analysis, and insight generation
- Provides a Streamlit web app for real-time interaction
- Exports Power BI-ready CSV tables
- Supports optional Gemini-powered insight generation with a local template fallback


## Local Setup

1. Open PowerShell in the project folder:

```powershell
cd "C:\Users\kamat\OneDrive\Documents\New project"
```

2. Create `.env` safely for Windows:

```powershell
.\scripts\setup_local.ps1
```

This creates `.env` from [`.env.example`](C:/Users/kamat/OneDrive/Documents/New%20project/.env.example) and keeps SQLite as the default local database.

3. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

4. Build the model, retrieval index, and export files from the database:

```powershell
python scripts/build_artifacts.py --force
```

5. Start the FastAPI backend:

```powershell
uvicorn backend.app.main:app --reload
```

6. Start the Streamlit frontend in a second PowerShell window:

```powershell
streamlit run frontend/streamlit_app.py
```

### PowerShell Environment Variable Note

Do not run this in PowerShell:

```powershell
DATABASE_URL=sqlite:///./reviews.db
```

That is Bash syntax and causes the error you saw.

If you want to set it only for the current PowerShell session, use:

```powershell
$env:DATABASE_URL = "sqlite:///./reviews.db"
```

You usually do not need to do that for this project, because the app already defaults to SQLite and reads values from `.env`.

### Database Notes

- Local default:
  `DATABASE_URL=sqlite:///./reviews.db`
- PostgreSQL production example:
  `DATABASE_URL=postgresql://user:password@host:5432/dbname`
- The backend automatically creates the `reviews` table on startup.
- The first artifact build seeds the database from `data/raw/final_dataset.csv` if the database is empty.

## API Endpoints

- `GET /health`
- `GET /api/v1/overview`
- `GET /api/v1/filters`
- `GET /api/v1/brand-comparison`
- `GET /api/v1/aspect-summary`
- `GET /api/v1/reviews`
- `POST /api/v1/analyze-review`
- `POST /api/v1/voice-review`
- `POST /api/v1/video-review`
- `GET /api/v1/live-reviews`
- `GET /api/v1/analytics/summary`
- `POST /api/v1/insights`
- `GET /api/v1/powerbi/exports`

## Tests

```powershell
python -m pytest tests -q
```

## Deployment Notes

- `render.yaml` deploys the FastAPI backend and builds artifacts during deploy.
- `frontend/requirements.txt` lets Streamlit Community Cloud install only frontend dependencies when you deploy `frontend/streamlit_app.py`.
- The Streamlit frontend reads `API_BASE_URL` from either an environment variable or Streamlit secrets.
- `.python-version` pins Render to Python `3.12.13`, which matches the current NumPy compatibility range used by this project.
- The backend now lazy-loads the analytics service so `/health` binds quickly on Render, and runtime retrieval uses TF-IDF to stay within free-plan memory limits.
- `docker-compose.yml` starts both backend and frontend together.
- On Linux deployments, FAISS is used automatically.
- On Windows, the app falls back to cosine retrieval while keeping the same API behavior.
## live stream
- Backend Using render : https://laptop-review-intelligence-api.onrender.com/docs
- Frontend using Streamlit : https://aspect-based-sentiment-analysis-dpajyi7yvmmuztcwm8ax2z.streamlit.app/
---

