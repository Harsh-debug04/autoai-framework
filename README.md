# AutoAI: Core ML Pipeline

AutoAI is an open-source framework that automates key stages of the machine learning lifecycle for tabular data. This version contains the core engine for data loading, cleaning, EDA, training, tuning, and explainability, all accessible via a powerful command-line interface (CLI) and a servable REST API.

This project is ready to be picked up in a new environment for final deployment and UI development.

---

## Core Features Implemented

*   **Automated EDA**: Generate a comprehensive HTML report from a dataset.
*   **Data Cleaning**: Automatic imputation of missing values.
*   **Model Training**: Train XGBoost, LightGBM, and CatBoost models.
*   **Hyperparameter Tuning**: Use Optuna to find the best model hyperparameters.
*   **Model Explainability**: Generate detailed evaluation metrics and SHAP feature importance plots.
*   **Model Serving**: A FastAPI backend to serve trained models via a `/predict` endpoint.
*   **Save/Load Models**: Ability to save trained models to a file for later use.

---

## Quickstart

### 1. Setup Environment

It is highly recommended to use a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate  # On Windows

# Install all dependencies
pip install -r requirements.txt
```

### 2. CLI Usage Examples

The main entry point is `main.py`. Use `python main.py --help` to see all available commands.

#### Generate an EDA Report
```bash
python main.py --data data/sample.csv --report eda_report.html
```
This will create a detailed `eda_report.html` file in your directory.

#### Train a Model
This command will load the data, clean it (impute missing values), train an XGBoost model, and save the final model to a file.
```bash
python main.py \
  --data data/ml_sample.csv \
  --target purchased \
  --clean \
  --output-model trained_model.joblib
```

#### Train with Tuning and Explainability
This is the full-power command. It will perform hyperparameter tuning with Optuna and generate SHAP explanation plots.
```bash
python main.py \
  --data data/ml_sample.csv \
  --target purchased \
  --clean \
  --tune \
  --explain \
  --output-model tuned_model.joblib
```
This will create `tuned_model.joblib` and `shap_summary.png`.

### 3. Running the API Server

The API allows you to serve a trained model.

**Step 1: Train and save a model** (if you haven't already)
```bash
python main.py --data data/ml_sample.csv --target purchased --output-model trained_model.joblib
```

**Step 2: Run the API server**
```bash
uvicorn autoai.api.main:app --host 0.0.0.0 --port 8000
```
The server will start, and you should see a message that the model was loaded.

**Step 3: Send a prediction request** (from another terminal)
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"data": {"age": 40, "salary": 90000, "country_UK": 1, "country_USA": 0}}'
```
You should receive a JSON response with the prediction, e.g., `{"prediction":1}`.

---

## Next Steps for Development


1.  **Dockerization**: The included `Dockerfile` is designed to containerize the application. It failed to build in the previous environment due to disk space limitations. The next step is to run `docker build -t autoai-app .`.
2.  **UI Development**: A Streamlit or React UI can be built to interact with the FastAPI backend, providing a user-friendly interface for the entire pipeline.
3.  **Expanding Features**: In a larger environment, the full feature set (AutoGluon, NLP, CV models) can be implemented.
