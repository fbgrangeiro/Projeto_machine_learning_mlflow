"""
Módulo de carregamento e processamento de dados
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Adicionar o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import (
    RAW_DATA_FILE, PROCESSED_DATA_DIR, TARGET_COLUMN, ID_COLUMN,
    RANDOM_STATE, TEST_SIZE, TRAIN_SIZE
)


class DataLoader:
    """Carrega e realiza diagnostico inicial dos dados"""
    
    @staticmethod
    def load_raw_data():
        """Carrega dados brutos do arquivo Excel"""
        print(f"Carregando dados de {RAW_DATA_FILE}...")
        df = pd.read_excel(RAW_DATA_FILE, header=1)
        print(f"Dados carregados: {df.shape}")
        return df
    
    @staticmethod
    def diagnose_data(df):
        """Diagnostica problemas de qualidade nos dados"""
        print("\n" + "="*60)
        print("DIAGNÓSTICO DE QUALIDADE DOS DADOS")
        print("="*60)
        
        # Informações gerais
        print(f"\n[OK] Dimensoes: {df.shape[0]} linhas, {df.shape[1]} colunas")
        print(f"[OK] Tipo de dados:\n{df.dtypes.value_counts()}")
        
        # Valores ausentes
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n[AVISO] Valores ausentes detectados:")
            print(df.isnull().sum()[df.isnull().sum() > 0])
        else:
            print("\n[OK] Nenhum valor ausente detectado")
        
        # Desbalanceamento de classes
        target_dist = df[TARGET_COLUMN].value_counts()
        print(f"\n[DADOS] Distribuicao da variavel-alvo:")
        print(f"   Classe 0 (Bom pagador): {target_dist[0]} ({target_dist[0]/len(df)*100:.1f}%)")
        print(f"   Classe 1 (Inadimplente): {target_dist[1]} ({target_dist[1]/len(df)*100:.1f}%)")
        print(f"   Razao de desbalanceamento: {target_dist[0]/target_dist[1]:.2f}:1")
        
        # Estatísticas descritivas
        print(f"\n[STATS] Estatisticas descritivas das variaveis numericas:")
        print(df.describe().T[['count', 'mean', 'std', 'min', 'max']].round(2))
        
        # Duplicatas
        duplicatas = df.duplicated().sum()
        print(f"\n[DADOS] Linhas duplicadas: {duplicatas}")
        
        return {
            'shape': df.shape,
            'missing_values': missing.sum(),
            'class_distribution': target_dist.to_dict(),
            'imbalance_ratio': target_dist[0] / target_dist[1],
            'duplicates': duplicatas
        }


class DataPreprocessor:
    """Preprocessa os dados para modelagem"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def preprocess(self, df):
        """Pipeline completo de preprocessamento"""
        print("\n" + "="*60)
        print("PREPROCESSAMENTO DE DADOS")
        print("="*60)
        
        # Remover coluna ID
        df_clean = df.drop(columns=[ID_COLUMN])
        
        # Separar features e target
        X = df_clean.drop(columns=[TARGET_COLUMN])
        y = df_clean[TARGET_COLUMN]
        
        print(f"\n[OK] Features: {X.shape[1]} variaveis")
        print(f"[OK] Target: {y.name}")
        print(f"[OK] Nomes das features:\n  {list(X.columns)}")
        
        return X, y
    
    def split_data(self, X, y):
        """Dividir dados em treino e teste com estratificação"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE,
            stratify=y
        )
        
        print(f"\n[OK] Divisao Treino/Teste:")
        print(f"  Treino: {X_train.shape[0]} amostras ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"  Teste: {X_test.shape[0]} amostras ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        # Verificar estratificação
        prop_train = y_train.value_counts(normalize=True)
        prop_test = y_test.value_counts(normalize=True)
        print(f"\n  Proporcao no treino: {prop_train[0]:.1%} / {prop_train[1]:.1%}")
        print(f"  Proporcao no teste: {prop_test[0]:.1%} / {prop_test[1]:.1%}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_data(self, X_train, X_test):
        """Padronizar features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\n[OK] Dados padronizados com StandardScaler")
        print(f"  Media das features (treino): {X_train_scaled.mean(axis=0).mean():.4f}")
        print(f"  Desvio padrao (treino): {X_train_scaled.std(axis=0).mean():.4f}")
        
        return X_train_scaled, X_test_scaled


def prepare_data_pipeline():
    """Pipeline completo de preparação de dados"""
    
    # Carregar
    loader = DataLoader()
    df = loader.load_raw_data()
    
    # Diagnosticar
    diagnostics = loader.diagnose_data(df)
    
    # Preprocessar
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess(df)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    X_train_scaled, X_test_scaled = preprocessor.scale_data(X_train, X_test)
    
    # Salvar dados processados
    pd.DataFrame(X_train_scaled).to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    pd.DataFrame(X_test_scaled).to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)
    X.to_csv(PROCESSED_DATA_DIR / "X_features_names.csv", index=False)
    
    print(f"\n[OK] Dados processados salvos em {PROCESSED_DATA_DIR}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'diagnostics': diagnostics,
        'feature_names': X.columns,
        'scaler': preprocessor.scaler
    }


if __name__ == "__main__":
    data = prepare_data_pipeline()
    print("\n[OK] Pipeline de dados concluido com sucesso!")
