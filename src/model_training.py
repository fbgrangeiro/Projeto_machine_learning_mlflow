"""
Módulo de treinamento e avaliação de modelos
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, auc)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MODELS_DIR,
    RANDOM_STATE, MODELS_CANDIDATES
)


class ModelTrainer:
    """Treina e avalia modelos de classificação"""
    
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        self.models = {}
        self.results = {}
    
    def train_model(self, X_train, y_train, model_name, model_class, params, use_grid_search=False):
        """Treina um modelo individual"""
        print(f"\n{'='*60}")
        print(f"Treinando: {model_name}")
        print(f"{'='*60}")
        
        # Não fechar a run aqui - manter aberta para avaliação
        mlflow.start_run(run_name=model_name)
        
        try:
            if use_grid_search:
                # Grid Search para otimização de hiperparâmetros
                param_grid = self._get_grid_params(model_name)
                model = GridSearchCV(
                    model_class(**params),
                    param_grid,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                    scoring='f1',
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                print(f"Melhores parametros: {model.best_params_}")
                print(f"Melhor score CV: {model.best_score_:.4f}")
                # Log apenas dos melhores parâmetros
                mlflow.log_params(model.best_params_)
                model = model.best_estimator_
            else:
                # Treinamento simples
                # Log de parâmetros
                mlflow.log_params(params)
                model = model_class(**params)
                model.fit(X_train, y_train)
            
            self.models[model_name] = model
            
            # Salvar modelo
            model_path = MODELS_DIR / f"{model_name}.pkl"
            joblib.dump(model, model_path)
            mlflow.log_artifact(str(model_path))
            print(f"Modelo salvo em {model_path}")
            
            return model
        finally:
            # Não fechar aqui - evaluate_model vai fazê-lo
            pass
    
    def evaluate_model(self, X_train, X_test, y_train, y_test, model, model_name):
        """Avalia um modelo treinado"""
        print(f"\n{'─'*60}")
        print(f"Avaliação: {model_name}")
        print(f"{'─'*60}")
        
        # Predições
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Verificar se o modelo tem predict_proba
        has_proba = hasattr(model, 'predict_proba')
        if has_proba:
            y_proba_test = model.predict_proba(X_test)[:, 1]
            y_proba_train = model.predict_proba(X_train)[:, 1]
        else:
            y_proba_test = y_pred_test.astype(float)
            y_proba_train = y_pred_train.astype(float)
        
        # Métricas de treino
        metrics_train = {
            'accuracy': accuracy_score(y_train, y_pred_train),
            'precision': precision_score(y_train, y_pred_train),
            'recall': recall_score(y_train, y_pred_train),
            'f1': f1_score(y_train, y_pred_train),
            'roc_auc': roc_auc_score(y_train, y_proba_train)
        }
        
        # Métricas de teste
        metrics_test = {
            'accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test),
            'recall': recall_score(y_test, y_pred_test),
            'f1': f1_score(y_test, y_pred_test),
            'roc_auc': roc_auc_score(y_test, y_proba_test)
        }
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred_test)
        
        print(f"\n[INFO] Métricas de TREINO:")
        for metric, value in metrics_train.items():
            print(f"  {metric:12s}: {value:.4f}")
        
        print(f"\n[INFO] Métricas de TESTE:")
        for metric, value in metrics_test.items():
            print(f"  {metric:12s}: {value:.4f}")
        
        print(f"\n[INFO] Matriz de Confusao:")
        print(f"  TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
        print(f"  FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
        
        # Log no MLflow (na run já aberta, não nested!)
        for metric, value in metrics_train.items():
            mlflow.log_metric(f"train_{metric}", value)
        for metric, value in metrics_test.items():
            mlflow.log_metric(f"test_{metric}", value)
        mlflow.log_metric("tn", cm[0,0])
        mlflow.log_metric("fp", cm[0,1])
        mlflow.log_metric("fn", cm[1,0])
        mlflow.log_metric("tp", cm[1,1])
        
        # Fechar a run aqui (que foi iniciada em train_model)
        mlflow.end_run()
        
        self.results[model_name] = {
            'metrics_train': metrics_train,
            'metrics_test': metrics_test,
            'confusion_matrix': cm,
            'y_pred': y_pred_test,
            'y_proba': y_proba_test
        }
        
        return metrics_test
    
    @staticmethod
    def _get_grid_params(model_name):
        """Retorna grid de hiperparâmetros para cada modelo"""
        grids = {
            'decision_tree': {
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [2, 4, 8]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [5, 10]
            },
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'solver': ['lbfgs', 'liblinear']
            }
        }
        return grids.get(model_name, {})
    
    def train_all_candidates(self, X_train, X_test, y_train, y_test):
        """Treina todos os modelos candidatos"""
        results_summary = []
        
        for model_name, config in MODELS_CANDIDATES.items():
            model_class_name = config['type']
            params = config['params']
            
            # Mapear nome para classe
            model_class_map = {
                'Perceptron': Perceptron,
                'DecisionTreeClassifier': DecisionTreeClassifier,
                'RandomForestClassifier': RandomForestClassifier,
                'LogisticRegression': LogisticRegression
            }
            
            model_class = model_class_map[model_class_name]
            
            # Treinar
            model = self.train_model(X_train, y_train, model_name, model_class, params,
                                    use_grid_search=(model_name != 'perceptron'))
            
            # Avaliar
            metrics = self.evaluate_model(X_train, X_test, y_train, y_test, model, model_name)
            
            results_summary.append({
                'model': model_name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'roc_auc': metrics['roc_auc']
            })
        
        # Resumo comparativo
        df_results = pd.DataFrame(results_summary)
        print(f"\n{'='*60}")
        print("RESUMO COMPARATIVO DE MODELOS")
        print(f"{'='*60}")
        print(df_results.to_string(index=False))
        
        return df_results


def train_all_models(X_train, X_test, y_train, y_test):
    """Pipeline de treinamento de todos os modelos"""
    trainer = ModelTrainer()
    results_df = trainer.train_all_candidates(X_train, X_test, y_train, y_test)
    
    # Salvar resultados
    results_df.to_csv(MODELS_DIR / "training_results.csv", index=False)
    
    return trainer


if __name__ == "__main__":
    from src.data_pipeline import prepare_data_pipeline
    
    # Carregar dados
    data = prepare_data_pipeline()
    
    # Treinar modelos
    trainer = train_all_models(
        data['X_train_scaled'],
        data['X_test_scaled'],
        data['y_train'],
        data['y_test']
    )
