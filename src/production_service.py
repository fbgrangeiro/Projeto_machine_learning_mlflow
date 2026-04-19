"""
Módulo de deploy e monitoramento do modelo em produção
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import MODELS_DIR, RESULTS_DIR


class ModelPredictor:
    """Carrega e realiza predições com modelo persistido"""
    
    def __init__(self, model_path, scaler_path=None):
        """
        Args:
            model_path: Path ao arquivo .pkl do modelo
            scaler_path: Path ao arquivo .pkl do scaler
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path else None
        self.model_path = model_path
        self.predictions_history = []
    
    def predict(self, X):
        """Realiza predições em batch"""
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def predict_single(self, x):
        """Realiza predição para uma única amostra"""
        x_reshaped = x.reshape(1, -1)
        pred, prob = self.predict(x_reshaped)
        return pred[0], prob[0]
    
    def log_predictions(self, X, y_true, timestamp=None):
        """Registra predições para monitoramento"""
        if timestamp is None:
            timestamp = datetime.now()
        
        y_pred, y_proba = self.predict(X)
        
        for i in range(len(X)):
            self.predictions_history.append({
                'timestamp': timestamp,
                'actual': y_true.iloc[i] if hasattr(y_true, 'iloc') else y_true[i],
                'predicted': y_pred[i],
                'probability': y_proba[i][1]
            })


class DataDriftDetector:
    """Detecta drift de dados em produção"""
    
    def __init__(self, reference_data):
        """
        Args:
            reference_data: DataFrame com dados de referência (dados de treino/validação)
        """
        self.reference_mean = reference_data.mean()
        self.reference_std = reference_data.std()
        self.reference_min = reference_data.min()
        self.reference_max = reference_data.max()
    
    def detect_drift_kolmogorov_smirnov(self, new_data, feature, threshold=0.05):
        """Usa teste Kolmogorov-Smirnov para detectar drift"""
        statistic, pvalue = stats.ks_2samp(
            self.reference_data[feature],
            new_data[feature]
        )
        return pvalue < threshold, pvalue
    
    def detect_drift_statistical(self, new_data, threshold_std=3):
        """Detecta drift usando desvios estatísticos"""
        drift_report = {}
        
        for col in new_data.columns:
            new_mean = new_data[col].mean()
            ref_mean = self.reference_mean[col]
            ref_std = self.reference_std[col]
            
            z_score = abs((new_mean - ref_mean) / ref_std) if ref_std > 0 else 0
            is_drift = z_score > threshold_std
            
            drift_report[col] = {
                'reference_mean': ref_mean,
                'new_mean': new_mean,
                'z_score': z_score,
                'is_drift': is_drift
            }
        
        return drift_report


class ModelDriftDetector:
    """Detecta degradação de performance do modelo em produção"""
    
    def __init__(self, baseline_metrics):
        """
        Args:
            baseline_metrics: Dict com métricas de baseline
        """
        self.baseline_metrics = baseline_metrics
        self.monitoring_history = []
    
    def evaluate_performance(self, y_true, y_pred, y_proba, timestamp=None):
        """Avalia performance atual do modelo"""
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
        
        if timestamp is None:
            timestamp = datetime.now()
        
        metrics = {
            'timestamp': timestamp,
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba)
        }
        
        self.monitoring_history.append(metrics)
        return metrics
    
    def detect_performance_degradation(self, current_metrics, threshold=0.05):
        """Detecta degradação de performance"""
        degradation_report = {}
        
        for metric, baseline_value in self.baseline_metrics.items():
            current_value = current_metrics.get(metric)
            if current_value is None:
                continue
            
            degradation = baseline_value - current_value
            degradation_pct = degradation / baseline_value if baseline_value > 0 else 0
            is_degraded = degradation_pct > threshold
            
            degradation_report[metric] = {
                'baseline': baseline_value,
                'current': current_value,
                'degradation': degradation,
                'degradation_pct': degradation_pct,
                'is_degraded': is_degraded
            }
        
        return degradation_report


class ProductionService:
    """Serviço de produção integrado"""
    
    def __init__(self, model_name='random_forest'):
        """Inicializa serviço de produção"""
        self.model_name = model_name
        self.model_path = MODELS_DIR / f"{model_name}.pkl"
        
        # Carregar modelo
        self.predictor = ModelPredictor(self.model_path)
        
        print(f"\n✓ Serviço de produção inicializado")
        print(f"  Modelo: {model_name}")
        print(f"  Path: {self.model_path}")
    
    def serve_prediction(self, X, return_confidence=True):
        """Realiza predição com informações adicionais"""
        predictions, probabilities = self.predictor.predict(X)
        
        result = {
            'predictions': predictions,
            'probabilities': probabilities,
        }
        
        if return_confidence:
            result['confidence'] = np.max(probabilities, axis=1)
        
        return result
    
    def batch_inference(self, X_data, output_path=None):
        """Realiza inferência em batch"""
        print(f"\n{'='*60}")
        print("INFERÊNCIA EM BATCH")
        print(f"{'='*60}")
        
        results = self.serve_prediction(X_data, return_confidence=True)
        
        df_results = pd.DataFrame({
            'prediction': results['predictions'],
            'prob_class_0': results['probabilities'][:, 0],
            'prob_class_1': results['probabilities'][:, 1],
            'confidence': results['confidence']
        })
        
        print(f"✓ Predições realizadas: {len(df_results)} amostras")
        print(f"  Classe 0 (Bom pagador): {(df_results['prediction'] == 0).sum()}")
        print(f"  Classe 1 (Inadimplente): {(df_results['prediction'] == 1).sum()}")
        print(f"  Confiança média: {df_results['confidence'].mean():.4f}")
        
        if output_path:
            df_results.to_csv(output_path, index=False)
            print(f"✓ Resultados salvos em {output_path}")
        
        return df_results


def deploy_model(model_name='random_forest'):
    """Inicia serviço de deploy do modelo"""
    service = ProductionService(model_name)
    return service


if __name__ == "__main__":
    from src.data_pipeline import prepare_data_pipeline
    
    # Carregar dados
    data = prepare_data_pipeline()
    
    # Deploy
    service = deploy_model('random_forest')
    
    # Realizar inferência em batch
    results = service.batch_inference(
        data['X_test_scaled'],
        output_path=RESULTS_DIR / "batch_predictions.csv"
    )
    
    print("\n✓ Deploy concluído com sucesso!")
