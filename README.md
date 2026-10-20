# TransferIQ — Premier League Transfer Fee Predictor

TransferIQ is a portfolio-oriented football analytics application that estimates a player's transfer fee from season-level performance statistics. It combines a Python/scikit-learn machine-learning pipeline with a FastAPI REST backend and a responsive Next.js + TypeScript frontend.

> **Important:** Transfer fees are difficult to predict from performance statistics alone. TransferIQ is an educational research project, not an official valuation service. Results are estimates, not guarantees.

## Product

- **Valuation Studio:** enter player statistics and request an estimate from the live API.
- **Separate application layers:** Next.js frontend, FastAPI backend, and a reusable Python model pipeline.
- **Input validation:** typed request schema and bounds checks at the API boundary.
- **Model transparency:** explicit limitations around historical data, contract context, injuries, negotiations, and club finances.
- **Responsive UI:** dashboard layout for desktop and mobile.

## Architecture

```text
Browser (Next.js + TypeScript)
        |
        | JSON / REST
        v
FastAPI (src/api.py)
        |
        v
Prediction utilities (src/models/predict.py)
        |
        v
Saved scikit-learn pipeline (models/transfer_value_model.pkl)

Training pipeline:
data/processed/transfer_features.csv
        -> src/models/train.py
        -> models/transfer_value_model.pkl
```

## Project structure

```text
transfer-iq/
├── frontend/
│   ├── package.json
│   ├── .env.example
│   └── src/app/
│       ├── layout.tsx
│       ├── page.tsx
│       └── globals.css
├── app/
│   └── streamlit_app.py        # legacy interface; not the primary app
├── src/
│   ├── api.py                  # FastAPI REST API
│   ├── config.py
│   ├── data/
│   │   ├── collectors.py
│   │   ├── cleaners.py
│   │   └── merger.py
│   ├── features/
│   │   └── engineering.py
│   └── models/
│       ├── train.py
│       ├── evaluate.py
│       └── predict.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/
│   └── transfer_value_model.pkl
├── notebooks/
└── tests/
```

## Run locally

### 1. Python backend

Use Python 3.11 or newer if available. From the repository root, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Ensure the trained model exists at `models/transfer_value_model.pkl`. If it does not, first prepare the processed dataset and train the model (see [Training the model](#training-the-model)).

Start the API:

```powershell
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

- API health: http://127.0.0.1:8000/health
- Model metadata: http://127.0.0.1:8000/api/v1/model
- Interactive API docs: http://127.0.0.1:8000/docs

### 2. Next.js frontend

Open a second PowerShell terminal at the repository root:

```powershell
cd frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

Open http://localhost:3000. The default API URL is `http://localhost:8000`. To use another backend, set `NEXT_PUBLIC_API_URL` in `frontend/.env.local` and restart the frontend server.

### 3. Make a prediction

Enter the player's age, position and season-level statistics in Valuation Studio, then choose **Generate valuation**. The browser sends a JSON request to `POST /api/v1/predict`; the API validates the input and returns a fee estimate in euros.

Example API request body:

```json
{
  "age": 23,
  "position": "Midfielder",
  "element_type": "3",
  "minutes": 2450,
  "goals_scored": 9,
  "assists": 7,
  "total_points": 164,
  "goals_conceded": 25,
  "clean_sheets": 8,
  "bonus": 14,
  "bps": 510,
  "influence": 690,
  "creativity": 720,
  "threat": 640,
  "ict_index": 205,
  "red_cards": 0,
  "yellow_cards": 4,
  "selected_by_percent": 18.5,
  "now_cost": 75
}
```

## Training the model

The model training pipeline expects `data/processed/transfer_features.csv` to exist. If the source datasets are already present, run the pipeline stages from the repository root:

```powershell
python -m src.data.collectors
python -m src.data.cleaners
python -m src.data.merger
python -m src.features.engineering
python -m src.models.train
python -m src.models.evaluate
```

The training module compares regression approaches using a chronological season split and saves the selected pipeline to `models/transfer_value_model.pkl`. Numeric and categorical imputers are fitted as part of preprocessing so missing feature values can be handled consistently at inference time.

The repository's raw and processed datasets may not be included because they are large and/or subject to their source licenses. Follow the data-source instructions and confirm that the expected files exist before training.

## API reference

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check API and model-file availability |
| `GET` | `/api/v1/model` | Return model metadata and limitations |
| `POST` | `/api/v1/predict` | Validate statistics and estimate a fee |
| `GET` | `/docs` | Interactive OpenAPI documentation |

The backend allows the local frontend origin by default. Set `TRANSFERIQ_CORS_ORIGINS` to a comma-separated list of trusted frontend origins when hosting the API elsewhere.

## Testing

From the repository root:

```powershell
python -m pytest -v
```

Frontend production build:

```powershell
cd frontend
npm run build
```

## Data and model limitations

- The current matched transfer dataset is limited compared with the full transfer market.
- Performance statistics do not include all major valuation drivers such as contract length, injuries, club finances, player demand, and negotiation structure.
- Historical transfer data may contain missing, inconsistent, or unmatched records.
- Estimates can be substantially wrong for unusually expensive transfers and should not be treated as a market quote.
- The app uses manually entered statistics; it does not claim to fetch a player's latest season automatically.

## Data sources

- Fantasy Premier League API and historical FPL season datasets for player statistics.
- Transfermarkt transfer datasets for historical transfer fees.
- Football-Data.co.uk for supplementary historical football data.

Check the individual source terms before redistributing any data.

## License

See [LICENSE](LICENSE) for the project's licensing terms.
