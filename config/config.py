"""
Configuração central do projeto de ML
"""
import os
from pathlib import Path

# Diretórios
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
CONFIG_DIR = PROJECT_ROOT / "config"
SRC_DIR = PROJECT_ROOT / "src"

# Arquivo de dados
RAW_DATA_FILE = RAW_DATA_DIR / "default_credit_card_clients.xls"

# MLflow
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "credit_default_prediction"

# Parâmetros do projeto
RANDOM_STATE = 42
TEST_SIZE = 0.3
TRAIN_SIZE = 0.7
VALIDATION_SIZE = 0.15

# Colunas do dataset
TARGET_COLUMN = "default payment next month"
ID_COLUMN = "ID"

# Modelos candidatos
MODELS_CANDIDATES = {
    "perceptron": {"type": "Perceptron", "params": {"random_state": RANDOM_STATE, "max_iter": 1000}},
    "decision_tree": {"type": "DecisionTreeClassifier", "params": {"random_state": RANDOM_STATE, "max_depth": 10}},
    "random_forest": {"type": "RandomForestClassifier", "params": {"n_estimators": 100, "random_state": RANDOM_STATE, "n_jobs": -1}},
    "logistic_regression": {"type": "LogisticRegression", "params": {"random_state": RANDOM_STATE, "max_iter": 1000}},
}

# Técnicas de redução de dimensionalidade
DIMENSIONALITY_REDUCTION = {
    "pca": {"n_components": [5, 10, 15, 20]},
    "lda": {"n_components": [1, 2, 3, 5]},
    "tsne": {"n_components": 2, "perplexity": 30},
}

# Métricas de avaliação
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"]

# Criar diretórios se não existirem
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
