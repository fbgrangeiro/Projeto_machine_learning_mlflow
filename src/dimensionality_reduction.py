"""
Módulo de redução de dimensionalidade
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import MODELS_DIR, RESULTS_DIR, RANDOM_STATE


class DimensionalityReducer:
    """Aplica técnicas de redução de dimensionalidade"""
    
    def __init__(self):
        self.reducers = {}
        self.results = {}
    
    def apply_pca(self, X_train, X_test, n_components_list=[5, 10, 15, 20]):
        """Aplica PCA com diferentes números de componentes"""
        print(f"\n{'='*60}")
        print("REDUÇÃO DE DIMENSIONALIDADE - PCA")
        print(f"{'='*60}")
        
        pca_results = []
        
        for n_comp in n_components_list:
            print(f"\n[INFO] Testando PCA com {n_comp} componentes...")
            
            pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            
            explained_var = pca.explained_variance_ratio_.sum()
            print(f"  [OK] Variancia explicada: {explained_var:.4f} ({explained_var*100:.2f}%)")
            print(f"  [OK] Dimensionalidade reduzida de {X_train.shape[1]} para {n_comp}")
            
            self.reducers[f'pca_{n_comp}'] = pca
            pca_results.append({
                'n_components': n_comp,
                'X_train': X_train_pca,
                'X_test': X_test_pca,
                'explained_variance': explained_var,
                'variance_ratio': pca.explained_variance_ratio_
            })
        
        return pca_results
    
    def apply_lda(self, X_train, X_test, y_train, n_components_list=[1, 2, 3, 5]):
        """Aplica LDA com diferentes números de componentes"""
        print(f"\n{'='*60}")
        print("REDUÇÃO DE DIMENSIONALIDADE - LDA")
        print(f"{'='*60}")
        
        # LDA: max componentes = min(n_features, n_classes - 1)
        n_classes = len(np.unique(y_train))
        max_components = min(X_train.shape[1], n_classes - 1)
        
        # Filtrar componentes válidos
        valid_n_comp = [n for n in n_components_list if n <= max_components]
        
        if not valid_n_comp:
            print(f"\n[AVISO] LDA nao pode ser aplicado: max componentes = {max_components}")
            print(f"        (formula: min(n_features={X_train.shape[1]}, n_classes-1={n_classes-1}))")
            return []
        
        lda_results = []
        
        for n_comp in valid_n_comp:
            print(f"\n[INFO] Testando LDA com {n_comp} componentes...")
            
            lda = LinearDiscriminantAnalysis(n_components=n_comp)
            X_train_lda = lda.fit_transform(X_train, y_train)
            X_test_lda = lda.transform(X_test)
            
            explained_var = lda.explained_variance_ratio_.sum()
            print(f"  [OK] Variancia explicada: {explained_var:.4f} ({explained_var*100:.2f}%)")
            print(f"  [OK] Dimensionalidade reduzida de {X_train.shape[1]} para {n_comp}")
            
            self.reducers[f'lda_{n_comp}'] = lda
            lda_results.append({
                'n_components': n_comp,
                'X_train': X_train_lda,
                'X_test': X_test_lda,
                'explained_variance': explained_var,
                'variance_ratio': lda.explained_variance_ratio_
            })
        
        return lda_results
    
    def apply_tsne(self, X_train, X_test, perplexity=30):
        """Aplica t-SNE para visualização"""
        print(f"\n{'='*60}")
        print("REDUÇÃO DE DIMENSIONALIDADE - t-SNE")
        print(f"{'='*60}")
        print(f"\n[INFO] Aplicando t-SNE com perplexity={perplexity}...")
        
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_STATE, max_iter=1000)
        X_train_tsne = tsne.fit_transform(X_train)
        
        # Para teste, usar fit_transform tambem (limitacao do t-SNE)
        X_test_tsne = tsne.fit_transform(X_test)
        
        print(f"  [OK] t-SNE aplicado com sucesso")
        print(f"  [OK] Dimensionalidade reduzida para 2D (visualizacao)")
        
        self.reducers['tsne'] = tsne
        
        return {
            'X_train': X_train_tsne,
            'X_test': X_test_tsne,
            'perplexity': perplexity
        }
    
    def compare_with_models(self, X_train_original, X_test_original, y_train, y_test,
                           reduced_datasets, model_func):
        """Compara desempenho de modelos com dados reduzidos vs originais"""
        print(f"\n{'='*60}")
        print("COMPARAÇÃO: MODELOS COM DADOS REDUZIDOS")
        print(f"{'='*60}")
        
        comparison_results = []
        
        # Avaliar com dados originais
        print(f"\n→ Treinando com DADOS ORIGINAIS ({X_train_original.shape[1]} features)...")
        model = model_func()
        model.fit(X_train_original, y_train)
        y_pred = model.predict(X_test_original)
        
        results_original = {
            'dataset': 'Original',
            'n_features': X_train_original.shape[1],
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test_original)[:, 1])
        }
        comparison_results.append(results_original)
        print(f"  Acurácia: {results_original['accuracy']:.4f}, F1: {results_original['f1']:.4f}, ROC-AUC: {results_original['roc_auc']:.4f}")
        
        # Avaliar com dados reduzidos
        for dataset_name, data_dict in reduced_datasets.items():
            print(f"\n→ Treinando com {dataset_name}...")
            
            X_train_red = data_dict['X_train']
            X_test_red = data_dict['X_test']
            
            print(f"  Dimensionalidade: {X_train_red.shape[1]} features")
            
            model = model_func()
            model.fit(X_train_red, y_train)
            y_pred = model.predict(X_test_red)
            
            results = {
                'dataset': dataset_name,
                'n_features': X_train_red.shape[1],
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test_red)[:, 1])
            }
            comparison_results.append(results)
            print(f"  Acurácia: {results['accuracy']:.4f}, F1: {results['f1']:.4f}, ROC-AUC: {results['roc_auc']:.4f}")
        
        df_comparison = pd.DataFrame(comparison_results)
        print(f"\n{'─'*60}")
        print("RESUMO DE COMPARACAO:")
        print(f"{'─'*60}")
        print(df_comparison.to_string(index=False))
        
        return df_comparison


def analyze_dimensionality(X_train, X_test, y_train, y_test, model_func):
    """Pipeline completo de análise de dimensionalidade"""
    
    reducer = DimensionalityReducer()
    
    # Aplicar técnicas
    pca_results = reducer.apply_pca(X_train, X_test)
    lda_results = reducer.apply_lda(X_train, X_test, y_train)
    tsne_result = reducer.apply_tsne(X_train, X_test)
    
    # Preparar datasets reduzidos para comparação (com validação)
    reduced_datasets = {
        'PCA_5': {'X_train': pca_results[0]['X_train'], 'X_test': pca_results[0]['X_test']},
        'PCA_10': {'X_train': pca_results[1]['X_train'], 'X_test': pca_results[1]['X_test']},
    }
    
    # Adicionar LDA apenas se disponível
    if lda_results:
        reduced_datasets['LDA_1'] = {'X_train': lda_results[0]['X_train'], 'X_test': lda_results[0]['X_test']}
    
    # Comparar com modelo
    comparison_df = reducer.compare_with_models(
        X_train, X_test, y_train, y_test,
        reduced_datasets, model_func
    )
    
    # Salvar resultados
    comparison_df.to_csv(RESULTS_DIR / "dimensionality_comparison.csv", index=False)
    
    # Salvar reducers
    joblib.dump(reducer.reducers, MODELS_DIR / "dimension_reducers.pkl")
    
    return reducer, comparison_df


if __name__ == "__main__":
    from src.data_pipeline import prepare_data_pipeline
    from sklearn.ensemble import RandomForestClassifier
    
    data = prepare_data_pipeline()
    
    def get_model():
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    reducer, comparison = analyze_dimensionality(
        data['X_train_scaled'],
        data['X_test_scaled'],
        data['y_train'],
        data['y_test'],
        get_model
    )
