# House Price Estimator

An end-to-end machine-learning product: a messy real-world housing dataset goes in, a
trained regression pipeline comes out, and it's served through a FastAPI backend to a
React frontend where anyone can enter a property's details and get an instant price
estimate.

## Overview

| Stage | What happens |
|---|---|
| **Data** | ~187K raw property listings from India (Kaggle) |
| **Notebook** | Cleans messy text fields (`"1.2 Cr"`, `"1200 sqft"`, `"3 out of 10"`), engineers features, trains & compares 3 regression models, exports a single deployable `Pipeline` |
| **Backend** | FastAPI service that loads the pipeline once at startup and serves `/predict` |
| **Frontend** | React + TypeScript form that posts to the backend and shows the estimate |

## Architecture

```
┌─────────────────┐        ┌──────────────────┐        ┌─────────────────────┐
│  React Frontend  │  HTTP  │  FastAPI Backend  │  load  │   house_price.pkl    │
│  (Vite, :5173)    │ ─────▶ │      (:8000)      │ ─────▶ │  sklearn Pipeline    │
│                  │  JSON  │                    │  once  │  (preprocess + model)│
│  PredictionForm  │ ◀───── │  POST /predict     │  at    │                      │
│  ResultPage      │        │  GET  /health      │ startup│                      │
└─────────────────┘        └──────────────────┘        └─────────────────────┘
                                                                    ▲
                                                                    │ joblib.dump()
                                                          ┌───────────────────┐
                                                          │ Jupyter Notebook   │
                                                          │ (clean, train,     │
                                                          │  evaluate, export) │
                                                          └───────────────────┘
                                                                    ▲
                                                                    │
                                                          ┌───────────────────┐
                                                          │ house_prices.csv   │
                                                          │ (Kaggle dataset)   │
                                                          └───────────────────┘
```

## Tech stack

- **Data / ML:** Python, pandas, numpy, scikit-learn, matplotlib, seaborn, joblib
- **Backend:** FastAPI, Pydantic v2 / pydantic-settings, uvicorn, pytest
- **Frontend:** React 18, TypeScript, Vite, React Router
- **Model artifact:** a single `sklearn.pipeline.Pipeline` (preprocessing + regressor) pickled with `joblib`

## Project structure

```
house-price-project/
├── notebooks/
│   ├── house_price_model.ipynb   # data cleaning, EDA, training, evaluation, export
│   └── data/                     # house_prices.csv goes here (not committed — see below)
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI app, CORS, startup model loading
│   │   ├── api/routes/prediction.py
│   │   ├── core/config.py
│   │   ├── schemas/prediction.py
│   │   ├── services/preprocessing.py
│   │   ├── services/inference.py
│   │   └── locations.json
│   ├── models/house_price.pkl
│   ├── tests/test_prediction.py
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── api/predictionClient.ts
│   │   ├── components/PredictionForm.tsx
│   │   ├── pages/HomePage.tsx | ResultPage.tsx | NotFoundPage.tsx
│   │   ├── types/prediction.ts
│   │   └── App.tsx
│   ├── public/locations.json
│   └── .env.example
└── README.md
```

## Dataset

**[House Price](https://www.kaggle.com/datasets/juhibhojani/house-price)** by Juhi
Bhojani, on Kaggle — real property listings from India (~187,000 rows).

The raw CSV is **not committed** to this repo (it's large and Kaggle's terms prefer
redistribution via their own platform). Download it yourself:

**Option A — manual:** download from the link above, unzip, and place the CSV at
`notebooks/data/house_prices.csv`.

**Option B — Kaggle CLI:**
```bash
pip install kaggle
# Get an API token: Kaggle → Settings → API → "Create New Token"
# Place kaggle.json in ~/.kaggle/ (macOS/Linux) or C:\Users\<you>\.kaggle\ (Windows)
kaggle datasets download -d juhibhojani/house-price -p notebooks/data --unzip
```

## Setup & run

### 0. Prerequisites

| Tool | Minimum version |
|---|---|
| Python | 3.11 |
| Node.js + npm | 18 |

### 1. Notebook — clean, train, export

```bash
cd notebooks
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install jupyter pandas numpy scikit-learn matplotlib seaborn joblib
jupyter notebook house_price_model.ipynb
# Run all cells — this produces house_price.pkl and locations.json in notebooks/
```

Copy the two generated files into the backend and frontend before running them:

```bash
cp notebooks/house_price.pkl backend/models/house_price.pkl
cp notebooks/locations.json backend/app/locations.json
cp notebooks/locations.json frontend/public/locations.json
```

*(These are already committed in this repo from a reference training run — you only need
to re-copy them if you retrain on the full dataset yourself.)*

### 2. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
# open http://localhost:8000/docs to try /predict from the Swagger UI
```

Run the tests:
```bash
pytest
```

### 3. Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
# open http://localhost:5173
```

With both running, fill out the form on `http://localhost:5173` and submit to get a live
prediction from the model served on `http://localhost:8000`.

## Environment variables

**`backend/.env`**

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `House Price Prediction API` | Displayed in `/docs` |
| `MODEL_PATH` | `models/house_price.pkl` | Path to the pickled pipeline |
| `LOCATIONS_PATH` | `app/locations.json` | Allowed/known locations, for the "other" fallback |
| `CORS_ORIGINS` | `["http://localhost:5173"]` | Origins allowed to call the API |
| `USES_LOG_TARGET` | `true` | Whether the exported model was trained on `log1p(price)` — must match the notebook's winning model |

**`frontend/.env`**

| Variable | Default | Description |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000` | Base URL of the FastAPI backend |

## API reference

### `GET /health`

```bash
curl http://localhost:8000/health
```
```json
{ "status": "ok" }
```

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "location": "Location_3",
    "carpet_area_sqft": 1200,
    "floor_num": 4,
    "bathroom": 2,
    "balcony": 1,
    "car_parking": 1,
    "furnishing": "Semi-Furnished",
    "transaction": "Resale",
    "ownership": "Freehold",
    "facing": "East"
  }'
```
```json
{ "predicted_price": 10467642.06 }
```

Sending a value the API can't parse (e.g. `"carpet_area_sqft": -100`) returns `422` with
a Pydantic validation error describing exactly which field failed.

## Model metrics

> ⚠️ The numbers below are from a placeholder training run and **must be replaced** with
> the real metrics printed in section 2.5 of the notebook once it's run against the full
> Kaggle dataset (`notebooks/house_price_model.ipynb`, "Evaluate" section).

| Model | Target | MAE | RMSE | R² |
|---|---|---|---|---|
| Linear Regression | raw | *fill in* | *fill in* | *fill in* |
| Random Forest | log1p | *fill in* | *fill in* | *fill in* |
| **Gradient Boosting (chosen)** | **log1p** | *fill in* | *fill in* | *fill in* |

**Chosen model:** *(state which model won and why, in one paragraph, referencing the
comparison table your notebook produces).*

## Screenshots

*(Add screenshots of the running app here before submitting — the form on
`HomePage`, and the estimate on `ResultPage`.)*

## Common pitfalls

- **Don't commit** `.env`, `node_modules/`, `.venv/`, `__pycache__/`, or the raw dataset
  CSV — see `.gitignore`.
- The pickled model only loads reliably with the **same scikit-learn version** it was
  trained with — check `sklearn.__version__` in the notebook and pin it in
  `backend/requirements.txt`.
- Report evaluation metrics on the **test set**, never the training set.
- Don't hard-code `http://localhost:8000` in frontend components — always go through
  `VITE_API_BASE_URL`.
