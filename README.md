# TransferIQ — Premier League Transfer Fee Predictor

TransferIQ is a machine-learning project that estimates football transfer fees using historical Premier League player statistics, age, playing time, position, and performance metrics.

The project includes a data collection pipeline, data cleaning and merging, feature engineering, model training and evaluation, automated tests, and a Streamlit dashboard.

## Features

* Collect current and historical Fantasy Premier League (FPL) statistics.
* Collect historical transfer records and football match data.
* Clean and merge player and transfer datasets.
* Engineer performance, playing-time, age, and disciplinary features.
* Compare regression models and evaluate predictions on held-out seasons.
* Predict an estimated transfer fee in euros.
* Run a web dashboard using Streamlit.
* Validate key functionality with automated tests.

## Project Structure

```text
transfer-iq/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   └── 04_model_training.ipynb
├── src/
│   ├── data/
│   │   ├── collectors.py
│   │   ├── cleaners.py
│   │   └── merger.py
│   ├── features/
│   │   └── engineering.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   └── config.py
├── models/
│   └── transfer_value_model.pkl
├── app/
│   └── streamlit_app.py
└── tests/
    ├── test_data.py
    ├── test_features.py
    └── test_prediction.py
```

## Requirements

* Python
* pandas
* NumPy
* scikit-learn
* requests
* matplotlib
* Streamlit
* pytest

## Installation

Clone the repository and enter the project directory:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd transfer-iq
```

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Data Sources

TransferIQ collects data from these external sources:

* **Fantasy Premier League API:** current FPL player statistics.
* **Fantasy Premier League historical repository:** historical season-level player statistics.
* **Transfermarkt transfer dataset:** historical transfer records and reported fees.
* **Football-Data.co.uk:** historical football match results and related match data.

Collected data is stored under `data/raw/` and processed outputs are stored under `data/processed/`.

Data source availability and formats can change. Some transfer records may have missing, inconsistent, or unmatched player information.

## Running the Pipeline

Run these modules from the repository root, in order:

```powershell
python -m src.data.collectors
python -m src.data.cleaners
python -m src.data.merger
python -m src.features.engineering
python -m src.models.train
python -m src.models.evaluate
```

The collector downloads the available source data. The cleaning and merging stages prepare the training dataset, feature engineering creates model inputs, and training saves the selected model to:

`models/transfer_value_model.pkl`

The evaluation module reports model performance using the evaluation logic implemented in the project.

If a source dataset is unavailable, inspect the collector output and confirm that the required input files exist before rerunning later stages.

## Running the Dashboard

From the repository root:

```powershell
python -m streamlit run app/streamlit_app.py
```

Open the local URL shown in the terminal, usually:

`http://localhost:8501`

Enter the player's statistics and click **Estimate transfer fee** to obtain a predicted fee in euros.

The dashboard uses manually entered statistics; it does not automatically retrieve a player's latest statistics by name.

## Model and Evaluation

The training pipeline compares multiple regression approaches using a season-based validation and test split. The selected model is saved as a scikit-learn pipeline.

Recorded evaluation results:

| Metric                         | Validation (2024) |    Test (2025) |
| ------------------------------ | ----------------: | -------------: |
| Mean Absolute Error (MAE)      |     €9.31 million | €15.13 million |
| Root Mean Squared Error (RMSE) |    €12.70 million | €20.50 million |

These results apply to the dataset and evaluation procedure used during training. They do not guarantee equivalent performance on future transfer markets.

## Automated Tests

Run the test suite:

```powershell
python -m pytest -v
```

The current test suite checks data configuration, feature calculations, invalid inputs, and prediction behavior.

## Limitations

* Historical transfer fees and player statistics may be incomplete or inconsistent.
* The merged training dataset contains a limited number of matched player-transfer records.
* Transfer fees depend on factors not fully represented by FPL statistics, including contract duration, injuries, club finances, negotiations, and market conditions.
* The available transfer data does not constitute complete coverage of the 2026 transfer market.
* Predictions are estimates for research and demonstration, not official valuations or guaranteed transfer fees.

## Future Improvements

* Improve player identity matching across datasets.
* Expand the training dataset with additional verified transfer records.
* Add stronger validation and more comprehensive integration tests.
* Compare model performance across positions, seasons, and fee ranges.
* Investigate additional features such as contract duration and injury history.

## License

See `LICENSE` for the project's licensing terms.
