"""
Utilitários do Dashboard Streamlit
"""
import sys
from pathlib import Path
import streamlit as st
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import MODELS_DIR


@st.cache_resource
def load_model():
    """Carrega o modelo com cache"""
    try:
        project_root = Path(__file__).parent.parent
        model_path = project_root / "models" / "random_forest.pkl"
        scaler_path = project_root / "models" / "scaler.pkl"
        
        if not model_path.exists():
            st.error(f"❌ Modelo não encontrado")
            st.stop()
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        
        return model, scaler
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        st.stop()


def scale_features(X, scaler):
    """Normaliza as features"""
    if scaler:
        return scaler.transform(X)
    return X


def make_prediction(model, X, scaler):
    """Realiza predição"""
    X_scaled = scale_features(X, scaler)
    prediction = model.predict(X_scaled)
    probability = model.predict_proba(X_scaled)
    return prediction, probability
