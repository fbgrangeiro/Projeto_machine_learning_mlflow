"""
Documentação de Arquitetura do Projeto
"""

ARCHITECTURE_DOCS = """
╔════════════════════════════════════════════════════════════════════════╗
║          📁 ARQUITETURA DO PROJETO DE MACHINE LEARNING                ║
╚════════════════════════════════════════════════════════════════════════╝

## 📂 Estrutura de Diretórios

```
Projeto_machine_learning_mlflow/
│
├── 📁 api/                           ⭐ NOVA - Servidor Flask
│   ├── __init__.py
│   ├── app.py                        # Factory e entry point
│   ├── config.py                     # Configuração e carregamento do modelo
│   └── routes.py                     # Definição de endpoints
│
├── 📁 dashboard/                     ⭐ NOVA - Interface Streamlit
│   ├── __init__.py
│   ├── app.py                        # Entry point Streamlit
│   ├── utils.py                      # Funções utilitárias
│   └── 📁 pages/                     # Sub-páginas (futuro)
│       ├── __init__.py
│       ├── home.py
│       ├── prediction.py
│       ├── batch.py
│       └── monitoring.py
│
├── 📁 scripts/                       ⭐ NOVA - Scripts utilitários
│   ├── dev.py                        # Menu de desenvolvimento
│   ├── tree.py                       # Gera árvore de diretórios
│   └── start_dev.bat                 # Windows batch script
│
├── 📁 src/
│   ├── __init__.py
│   ├── data_pipeline.py              # ETL e processamento
│   ├── model_training.py             # Treinamento e MLflow
│   ├── dimensionality_reduction.py   # PCA, LDA, t-SNE
│   └── production_service.py         # Predição e monitoramento
│
├── 📁 config/
│   ├── __init__.py
│   └── config.py                     # Configuração centralizada
│
├── 📁 data/
│   ├── raw/                          # Dados brutos (ignorar)
│   └── processed/                    # Dados processados (ignorar)
│
├── 📁 models/                        # Modelos treinados (ignorar)
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── *.pkl
│
├── 📁 results/                       # Resultados de experimentos (ignorar)
│   ├── training_results.csv
│   ├── dimensionality_comparison.csv
│   └── production_predictions.csv
│
├── 📁 notebooks/                     # Jupyter notebooks para exploração
│   ├── 05_Results_Summary.ipynb
│   ├── 06_TSNE_Parameter_Debug.ipynb
│   └── 07_Producao_Demo.ipynb
│
├── 📁 docs/                          ⭐ NOVA - Documentação
│   ├── ARCHITECTURE.md               # Este arquivo
│   ├── QUICK_START.md                # Guia rápido
│   ├── API_REFERENCE.md
│   └── DEPLOYMENT.md
│
├── 📄 main.py                        # Script principal (pipeline completo)
├── 📄 requirements.txt                # Dependências
├── 📄 Dockerfile                     # Containerização
├── 📄 README.md                      # Informações gerais
├── 📄 STREAMLIT_APP.md               # Documentação Streamlit
├── 📄 MODEL_SERVER.md                # Documentação Flask
├── 📄 QUICK_START.md                 # Guia rápido (três opções)
└── 📄 LICENSE                        # Licença
```

---

## 🎯 Organização por Contexto

### 1️⃣ **Data Science & Jupyter**
- 📁 `notebooks/` - Exploração interativa
- 📁 `src/data_pipeline.py` - Processamento
- 📁 `src/dimensionality_reduction.py` - Técnicas de redução

### 2️⃣ **Treinamento de Modelos**
- 📁 `src/model_training.py` - Treinamento com MLflow
- 📁 `src/production_service.py` - Predição e monitoramento
- 📁 `models/` - Artefatos persistidos

### 3️⃣ **API REST (Backend)**
- 📁 `api/` - Servidor Flask
  - Endpoints: `/predict`, `/predict/batch`, `/predict/csv`
  - Health check: `/health`
  - Documentação: `/example`

### 4️⃣ **Website/Dashboard (Frontend)**
- 📁 `dashboard/` - Interface Streamlit
  - Páginas: Home, Predição, Batch, Monitoramento
  - Componentes: Upload, Visualizações, Métricas

### 5️⃣ **Orquestração & Scripts**
- 📁 `scripts/` - Ferramentas de desenvolvimento
- 📄 `main.py` - Pipeline completo

### 6️⃣ **Documentação & Configuração**
- 📁 `docs/` - Documentação técnica
- 📄 `requirements.txt` - Dependências
- 📄 `Dockerfile` - Containerização

---

## 🔄 Fluxo de Dados

```
┌─────────────────────────────────────────────────────────────┐
│ 1. CARREGAR DADOS (src/data_pipeline.py)                   │
│    └─ CSV → Pandas → Análise de Qualidade                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ 2. PREPROCESSAR (src/data_pipeline.py)                     │
│    └─ Split Train/Test → StandardScaler → Salvar          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ 3. TREINAR MODELOS (src/model_training.py)                │
│    └─ GridSearchCV → 4 Modelos → MLflow Tracking          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ 4. REDUZIR DIMENSIONALIDADE (src/dimensionality_reduction.py)
│    └─ PCA → LDA → t-SNE → Comparação                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ 5. SELECIONAR MODELO (main.py - Parte 5)                  │
│    └─ Best ROC-AUC → Serializar com Joblib                │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
    ┌─────────────┐           ┌─────────────────┐
    │ 6A. DEPLOY  │           │ 6B. MONITORING  │
    │ (Produção)  │           │ (Drift Detection)
    └─────────────┘           └─────────────────┘
        │
        ├─► API Flask (api/)
        │   ├─ POST /predict
        │   ├─ POST /predict/batch
        │   └─ POST /predict/csv
        │
        └─► Dashboard Streamlit (dashboard/)
            ├─ Home
            ├─ Predição Manual
            ├─ Análise Batch
            └─ Monitoramento
```

---

## 📊 Dependências & Imports

### Core ML
```python
import sklearn        # Modelos, preprocessing, métricas
import numpy         # Cálculos numéricos
import pandas        # Manipulação de dados
```

### Tracking & Deployment
```python
import mlflow        # Rastreamento de experimentos
import joblib        # Serialização de modelos
```

### Web Frameworks
```python
from flask import Flask                # API REST
import streamlit as st                 # Dashboard web
```

### Visualização
```python
import matplotlib, seaborn             # Gráficos estáticos
import plotly                          # Gráficos interativos
```

---

## 🚀 Como Usar a Arquitetura

### 1️⃣ Desenvolvimento
```bash
# Opção A: Menu interativo
python scripts/dev.py

# Opção B: Comandos diretos
streamlit run dashboard/app.py         # Dashboard
python api/app.py                      # API
mlflow ui                              # Tracking
python main.py                         # Treinar
```

### 2️⃣ Produção
```bash
# API
gunicorn -w 4 -b 0.0.0.0:5001 api.app:create_app()

# Dashboard (Streamlit Cloud)
git push & deploy

# Docker
docker build -t ml-inadimplencia .
docker run -p 5001:5001 ml-inadimplencia
```

---

## 📝 Convenções de Código

### Estrutura de Módulos
```python
# 1. Imports
import sys
from pathlib import Path

# 2. Adicionar ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 3. Imports locais
from config.config import MODELS_DIR

# 4. Funções/Classes
def main():
    pass

# 5. Entry point
if __name__ == "__main__":
    main()
```

### Documentação
- **Modules**: Docstring no topo do arquivo
- **Functions**: Docstring com description, args, returns
- **Classes**: Docstring descrevendo propósito

---

## 🧪 Testes

```bash
# Estrutura proposta
📁 tests/
├── test_models.py          # Testes de modelos
├── test_pipeline.py        # Testes do pipeline
├── test_server.py          # Testes da API
└── test_dashboard.py       # Testes Streamlit (futuro)

# Executar
pytest tests/
```

---

## 📦 Deployment

### Opção 1: Streamlit Cloud
1. Push do código para GitHub
2. Conectar repositório em https://streamlit.io/cloud
3. Deploy automático

### Opção 2: Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5001
CMD ["python", "api/app.py"]
```

### Opção 3: Heroku
```bash
heroku create seu-app
git push heroku main
```

---

## 🔄 Melhorias Futuras

- [ ] Testes automatizados mais robustos
- [ ] CI/CD pipeline com GitHub Actions
- [ ] Autenticação e autorização
- [ ] Database para histórico de predições
- [ ] Alertas de drift em tempo real
- [ ] API GraphQL
- [ ] Multi-modelo (A/B testing)
- [ ] SDK Python

---

## 📚 Documentação Relacionada

- `docs/QUICK_START.md` - Guia rápido
- `docs/API_REFERENCE.md` - Endpoints da API
- `MODEL_SERVER.md` - Documentação Flask
- `STREAMLIT_APP.md` - Documentação Streamlit

---

**Última atualização:** Abril 2026
**Autor:** Francisco Bruno Lopes Grangeiro
"""

if __name__ == "__main__":
    print(ARCHITECTURE_DOCS)
