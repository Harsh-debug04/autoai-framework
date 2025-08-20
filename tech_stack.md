# AutoAI Tech Stack

This document outlines the core technologies, libraries, and frameworks used in the AutoAI project.

## Core ML & Data Science

- **pandas**: For data manipulation and analysis, primarily using its powerful DataFrame structures.
- **numpy**: For numerical operations and handling multi-dimensional arrays.
- **scikit-learn**: Used for essential preprocessing tasks, such as `SimpleImputer` for data cleaning.
- **joblib**: For efficiently saving and loading trained machine learning models.

## Exploratory Data Analysis (EDA)

- **ydata-profiling**: For generating detailed, comprehensive EDA reports in HTML format.
- **sweetviz**: An alternative library for creating beautiful, high-density EDA reports.

## Machine Learning Models

- **XGBoost**: A powerful and popular gradient boosting library for structured data.
- **LightGBM**: A high-performance gradient boosting framework known for its speed and efficiency.
- **CatBoost**: A gradient boosting library that handles categorical features automatically and effectively.

## AutoML & Hyperparameter Tuning

- **Optuna**: A state-of-the-art hyperparameter optimization framework used to find the best model configurations.

## Model Explainability

- **SHAP (SHapley Additive exPlanations)**: For explaining the output of machine learning models, providing insights into feature importance and model behavior.
- **matplotlib**: Used by SHAP to generate and save plots, such as the summary feature importance plot.

## API & Deployment

- **FastAPI**: A modern, high-performance web framework for building the model-serving API.
- **uvicorn**: A lightning-fast ASGI server, used to run the FastAPI application.

## Development & Tooling

- **Python**: The core programming language for the entire project.
- **argparse**: For building the user-friendly command-line interface (CLI).
- **Docker**: For containerizing the application, ensuring a consistent and reproducible environment (as indicated in `Dockerfile` and `README.md`).
