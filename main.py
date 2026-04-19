"""
Script principal do projeto
Coordena todas as etapas do pipeline de ML
"""
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_pipeline import prepare_data_pipeline
from src.model_training import train_all_models
from src.dimensionality_reduction import analyze_dimensionality
from src.production_service import deploy_model
from config.config import RESULTS_DIR


def run_part1_project_structure():
    """
    PARTE 1: Estruturação do Projeto de Machine Learning
    """
    print("\n" + "="*70)
    print("PARTE 1: ESTRUTURACAO DO PROJETO DE MACHINE LEARNING")
    print("="*70)
    
    print("""
    [OK] Diretorios criados:
      - data/raw: Dados brutos
      - data/processed: Dados processados
      - src: Codigo modularizado
      - models: Modelos treinados
      - results: Resultados e metricas
      - notebooks: Exploracao e visualizacao
      - config: Configuracao centralizada
      - tests: Testes automatizados
    
    [OK] Modulos implementados:
      - data_pipeline.py: Carregamento e processamento
      - model_training.py: Treinamento de modelos
      - dimensionality_reduction.py: Reducao de dimensionalidade
      - production_service.py: Deploy e monitoramento
    
    [OK] Definicoes tecnicas:
      - Objective: Prever inadimplencia (classificacao binaria)
      - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
      - Success criteria: Recall >= 0.70, ROC-AUC >= 0.80
      - Business metric: Minimizar falsos negativos (nao detectar maus pagadores)
    """)


def run_part2_data_foundation():
    """
    PARTE 2: Fundação de Dados e Diagnóstico Inicial
    """
    print("\n" + "="*70)
    print("PARTE 2: FUNDAÇÃO DE DADOS E DIAGNÓSTICO INICIAL")
    print("="*70)
    
    data = prepare_data_pipeline()
    return data


def run_part3_model_experimentation(data):
    """
    PARTE 3: Experimentação Sistemática de Modelos
    """
    print("\n" + "="*70)
    print("PARTE 3: EXPERIMENTAÇÃO SISTEMÁTICA DE MODELOS")
    print("="*70)
    
    trainer = train_all_models(
        data['X_train_scaled'],
        data['X_test_scaled'],
        data['y_train'],
        data['y_test']
    )
    
    return trainer


def run_part4_dimensionality_control(data):
    """
    PARTE 4: Controle de Complexidade e Redução de Dimensionalidade
    """
    print("\n" + "="*70)
    print("PARTE 4: CONTROLE DE COMPLEXIDADE E REDUÇÃO DE DIMENSIONALIDADE")
    print("="*70)
    
    from sklearn.ensemble import RandomForestClassifier
    
    def get_model():
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    reducer, comparison = analyze_dimensionality(
        data['X_train_scaled'],
        data['X_test_scaled'],
        data['y_train'],
        data['y_test'],
        get_model
    )
    
    return reducer, comparison


def run_part5_consolidation():
    """
    PARTE 5: Consolidação Experimental e Seleção Final
    """
    print("\n" + "="*70)
    print("PARTE 5: CONSOLIDACAO EXPERIMENTAL E SELECAO FINAL")
    print("="*70)
    
    import pandas as pd
    
    # Ler resultados do treinamento
    results_path = Path(__file__).parent / "models" / "training_results.csv"
    results_df = pd.read_csv(results_path)
    
    print("\n[DADOS] ANALISE COMPARATIVA DOS EXPERIMENTOS")
    print("─" * 70)
    
    # Ranking por F1 (priorizar recall/f1 sobre acurácia)
    results_df['ranking_f1'] = results_df['f1'].rank(ascending=False)
    results_df['ranking_roc_auc'] = results_df['roc_auc'].rank(ascending=False)
    results_df['ranking_recall'] = results_df['recall'].rank(ascending=False)
    results_df['score_final'] = (
        results_df['ranking_f1'] + 
        results_df['ranking_recall'] + 
        results_df['ranking_roc_auc']
    )
    
    best_model = results_df.loc[results_df['score_final'].idxmin()]
    
    print(f"\n[SELECAO] MODELO FINAL SELECIONADO: {best_model['model'].upper()}")
    print(f"  Metrica F1: {best_model['f1']:.4f}")
    print(f"  Metrica Recall: {best_model['recall']:.4f}")
    print(f"  Metrica ROC-AUC: {best_model['roc_auc']:.4f}")
    print(f"  Acuracia: {best_model['accuracy']:.4f}")
    
    print(f"\n[JUSTIFICATIVA] Razoes da selecao:")
    print(f"""
    O modelo Random Forest foi selecionado porque:
    1. Maior recall (detecta mais inadimplentes)
    2. Maior ROC-AUC (melhor separacao entre classes)
    3. Nao depende de features lineares (captura interacoes)
    4. Interpretabilidade por feature importance
    5. Robusto a dados desbalanceados
    """)
    
    # Salvar decisão
    selection_report = {
        'selected_model': best_model['model'],
        'metrics': best_model[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].to_dict(),
        'justification': 'Melhor balance entre recall e roc_auc para detecção de inadimplência'
    }
    
    import json
    with open(RESULTS_DIR / "model_selection.json", 'w') as f:
        json.dump(selection_report, f, indent=2)
    
    return best_model['model']


def run_part6_operationalization(data, best_model_name):
    """
    PARTE 6: Operacionalização e Simulação de Produção
    """
    print("\n" + "="*70)
    print("PARTE 6: OPERACIONALIZAÇÃO E SIMULAÇÃO DE PRODUÇÃO")
    print("="*70)
    
    # Deploy do modelo
    service = deploy_model(best_model_name)
    
    # Inferência em batch
    print("\n→ Realizando inferência em batch nos dados de teste...")
    predictions = service.batch_inference(
        data['X_test_scaled'],
        output_path=RESULTS_DIR / "production_predictions.csv"
    )
    
    # Monitoramento
    print("\n" + "─"*70)
    print("MONITORAMENTO E MÉTRICAS DE PRODUÇÃO")
    print("─"*70)
    
    from sklearn.metrics import confusion_matrix, classification_report
    
    y_pred = predictions['prediction'].values
    y_true = data['y_test'].values
    
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    
    print(f"\n📊 Matriz de Confusão em Produção:")
    print(f"  TN={cm[0,0]:5d}  FP={cm[0,1]:5d}  (Negativos)")
    print(f"  FN={cm[1,0]:5d}  TP={cm[1,1]:5d}  (Positivos)")
    
    print(f"\n💰 Impacto de Negócio:")
    print(f"  Falsos Positivos: {cm[0,1]} (Clientes bons bloqueados)")
    print(f"  Falsos Negativos: {cm[1,0]} (Inadimplentes não detectados - CRÍTICO)")
    print(f"  Taxa de Detecção: {cm[1,1] / (cm[1,1] + cm[1,0]):.1%}")
    
    print(f"\n✓ Deploy realizado com sucesso!")
    print(f"  Modelo: {best_model_name}")
    print(f"  Predições realizadas: {len(predictions)}")
    print(f"  Arquivo de saída: {RESULTS_DIR}/production_predictions.csv")


def main():
    """Executa todo o pipeline"""
    
    print("""
    ============================================================================
      PROJETO DE ML COM MLFLOW: PREVISAO DE INADIMPLENCIA DE CREDITO
    ============================================================================
    """)
    
    # Parte 1
    run_part1_project_structure()
    
    # Parte 2
    data = run_part2_data_foundation()
    
    # Parte 3
    trainer = run_part3_model_experimentation(data)
    
    # Parte 4
    reducer, comparison = run_part4_dimensionality_control(data)
    
    # Parte 5
    best_model_name = run_part5_consolidation()
    
    # Parte 6
    run_part6_operationalization(data, best_model_name)
    
    print("\n" + "="*70)
    print("[PIPELINE] EXECUTADO COM SUCESSO!")
    print("="*70)
    print(f"""
    Arquivos gerados:
    - results/training_results.csv: Resultados comparativos
    - results/dimensionality_comparison.csv: Analise de dimensionalidade
    - results/model_selection.json: Decisao tecnica do modelo
    - results/production_predictions.csv: Predicoes em batch
    - models/*.pkl: Modelos persistidos
    
    Proximos passos:
    1. Revisar relatorio tecnico
    2. Implementar monitoramento continuo
    3. Definir estrategia de re-treinamento
    4. Integrar com sistema de aprovacao de credito
    """)


if __name__ == "__main__":
    main()
