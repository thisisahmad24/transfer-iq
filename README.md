# TransferIQ

Premier League Transfer Value Predictor.

## Status

Phase 0 — Repository foundation.

The repository structure is intentionally frozen at Phase 0. New folders or duplicate files will not be introduced unless a concrete technical need is explained and approved first.

## Planned stack

- Python
- requests
- pandas
- NumPy
- scikit-learn
- matplotlib
- Streamlit
- pytest

## Frozen project structure

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
│   └── models/
│       ├── train.py
│       ├── evaluate.py
│       └── predict.py
├── models/
│   └── transfer_value_model.pkl
├── app/
│   └── streamlit_app.py
└── tests/
    ├── test_data.py
    ├── test_features.py
    └── test_prediction.py
```

## Development plan

1. Repository foundation
2. Data acquisition
3. Data processing
4. Exploratory data analysis
5. Feature engineering
6. Baseline machine learning
7. Model evaluation
8. Model comparison
9. Prediction engine
10. Streamlit application
11. Dynamic data refresh
12. Testing
13. Documentation and deployment

Data sources will be verified for availability, coverage, usage terms, and update frequency before the ingestion pipeline is implemented.
