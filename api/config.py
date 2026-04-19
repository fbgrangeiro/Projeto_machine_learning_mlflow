"""
Configuração da API Flask
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.production_service import ModelPredictor
from config.config import MODELS_DIR


# Cache do modelo
_model = None
_scaler = None
_model_metadata = None


def setup_config(app, debug=False):
    """Setup configuração da aplicação"""
    app.config['DEBUG'] = debug
    app.config['JSON_SORT_KEYS'] = False


def load_model_once():
    """Carrega modelo apenas uma vez (singleton)"""
    global _model, _scaler, _model_metadata
    
    if _model is not None:
        return _model, _scaler, _model_metadata
    
    model_path = MODELS_DIR / "random_forest.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}")
    
    _model = ModelPredictor(model_path, scaler_path if scaler_path.exists() else None)
    
    try:
        _scaler = _model.scaler
    except:
        _scaler = None
    
    _model_metadata = {
        "model_name": "random_forest",
        "model_path": str(model_path),
        "scaler_loaded": scaler_path.exists(),
        "version": "1.0"
    }
    
    print("✓ Modelo carregado com sucesso!")
    print(f"  Modelo: {_model_metadata['model_name']}")
    print(f"  Path: {_model_metadata['model_path']}")
    
    return _model, _scaler, _model_metadata
