# autoai-framework

autoai/
├── core/                    # Data preprocessing
│   ├── cleaner.py
│   ├── feature_engineer.py
│   └── task_identifier.py
├── eda/                     # Exploratory Data Analysis
│   ├── visualizer.py
│   └── report_generator.py
├── models/                  # Model training, tuning
│   ├── tabular/
│   ├── image/
│   └── text/
├── explainability/          # SHAP, LIME, Counterfactuals
│   ├── explainer.py
│   └── fairness.py
├── deployment/              # FastAPI app
│   ├── api.py
│   ├── routes.py
│   └── model_loader.py
├── ui/                      # React + Tailwind frontend
│   ├── public/
│   └── src/
├── utils/                   # Logging, metrics, config
├── tests/                   # Unit & integration tests
├── configs/                 # YAML/JSON configurations
├── notebooks/               # Experiments
├── scripts/                 # CLI or batch pipelines
├── Dockerfile
├── docker-compose.yml
├── main.py                  # CLI entry point
├── requirements.txt
├── README.md
└── .github/workflows/       # CI/CD
