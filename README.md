# ML Failure Analyzer + Explainable AI Dashboard

## Problem Statement
Most ML demos stop at accuracy. This project goes further by identifying where a classifier fails, why it fails, and which features drive its predictions.

## Features
- Built-in Adult Income and Loan Default datasets with local sample CSVs
- CSV upload support for general classification tasks
- Automatic target selection and classification sanity checks
- Modular preprocessing with missing value handling, one-hot encoding, and scaling where needed
- Model selection: Logistic Regression, Random Forest, and optional XGBoost
- Evaluation metrics: accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and classification report
- Misclassification analysis with confidence scores
- Failure pattern detection across categorical and numeric features
- Human-readable insights for where the model fails
- SHAP-based explainability with graceful fallback to model feature importance
- Export of misclassified rows and trained model artifacts

## Tech Stack
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- SHAP
- Joblib

## How It Works
1. Load the built-in Adult Income or Loan Default dataset, or upload your own CSV.
2. Select a target column.
3. Choose a classification model and train/test split.
4. Train the pipeline and inspect evaluation metrics.
5. Review misclassified samples, failure patterns, and confidence differences.
6. Use SHAP or feature importance to understand model behavior.
7. Download failure cases and the trained model.

## Project Structure
```text
ml-failure-analyzer/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── sample_data/
│   ├── adult_income_sample.csv
│   └── loan_default_sample.csv
├── models/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
└── utils/
    ├── __init__.py
    ├── preprocessing.py
    ├── evaluation.py
    ├── error_analysis.py
    └── explainability.py
```

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this project to GitHub.
2. Create a new app on Streamlit Community Cloud.
3. Choose the repository and set `app.py` as the entry point.
4. Make sure `requirements.txt` is included in the repo.
5. Deploy and share the app link.

## Screenshots
Add screenshots here after running the dashboard locally.

## Resume Bullet Points
- Built an ML Failure Analysis Dashboard to identify model weaknesses beyond accuracy using error pattern detection and explainable AI.
- Implemented end-to-end ML pipeline including preprocessing, model training, evaluation, misclassification analysis, and SHAP-based interpretability.
- Designed interactive Streamlit dashboard to visualize accuracy metrics, confusion matrix, failure clusters, and feature-level prediction explanations.

## LinkedIn Post Caption
“I built an ML system that doesn’t stop at accuracy — it explains where the model fails and why.”

The project highlights model evaluation, error analysis, explainable AI, a Streamlit dashboard, and SHAP-driven insights.

