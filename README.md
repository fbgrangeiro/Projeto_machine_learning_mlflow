# 🤖 Projeto de Machine Learning com MLflow - Previsão de Inadimplência

## 📌 Contexto

Este projeto reutiliza o dataset do trabalho anterior de **Previsão de Inadimplência de Cartão de Crédito** e o reformula sob a perspectiva de **engenharia de ML** e **operacionalização em produção**.

O objetivo é demonstrar uma abordagem profissional, estruturada e escalável, com ênfase em:
- ✅ Estruturação técnica (não apenas explorações em notebooks)
- ✅ Experimentação sistemática (rastreamento com MLflow)
- ✅ Redução de dimensionalidade (PCA, LDA, t-SNE)
- ✅ Deploy e monitoramento em produção

---

## 🏗️ Arquitetura do Projeto

```
Projeto_machine_learning_mlflow/
│
├── 📋 RAIZ (Entry Points)
│   ├── main.py                       # Pipeline ML (6 partes)
│   ├── README.md                     # Este arquivo
│   ├── requirements.txt              # Dependências
│   ├── VIDEO_PRODUCAO_GUIA.md        # Guia para demo em vídeo
│   └── ROTEIRO_VIDEO.md              # Script do vídeo
│
├── 🔧 CÓDIGO-FONTE
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py                 # Configuração centralizada
│   └── src/
│       ├── __init__.py
│       ├── data_pipeline.py          # Carregamento e diagnóstico
│       ├── model_training.py         # Treinamento com MLflow
│       ├── dimensionality_reduction.py   # PCA, LDA, t-SNE
│       └── production_service.py     # Monitoramento de drift
│
├── 🚀 PRODUÇÃO
│   ├── api/                          # Flask REST API
│   │   ├── app.py                    # Factory pattern
│   │   ├── config.py                 # Model loading
│   │   └── routes.py                 # 6 endpoints
│   │
│   ├── dashboard/                    # Streamlit Multi-página
│   │   ├── 📄_Home.py                # Página inicial
│   │   ├── pages/
│   │   │   ├── 📊_Predicao.py        # Predição individual
│   │   │   ├── 📈_Batch.py           # Análise em batch
│   │   │   └── 📉_Monitoramento.py   # Métricas e alertas
│   │   ├── utils.py                  # Funções compartilhadas
│   │   └── .streamlit/
│   │       └── config.toml           # Tema visual
│   │
│   └── scripts/                      # Dev helpers
│       ├── dev.py                    # Menu interativo
│       └── tree.py                   # Visualizador de estrutura
│
├── 📚 DOCUMENTAÇÃO
│   └── docs/
│       ├── ARCHITECTURE.md           # Design técnico detalhado
│       ├── QUICK_START.md            # Guia de início rápido
│       └── NAVIGATION.md             # Mapa de navegação
│
├── 📊 DADOS
│   ├── raw/
│   │   └── default_credit_card_clients.xls
│   └── processed/
│       ├── X_train.csv, y_train.csv
│       ├── X_test.csv, y_test.csv
│       └── X_features_names.csv
│
├── 🎯 MODELOS & RESULTADOS
│   ├── models/                       # Modelos treinados
│   │   ├── *.pkl (4 modelos)
│   │   └── training_results.csv
│   └── results/                      # Relatórios
│       ├── model_selection.json
│       ├── dimensionality_comparison.csv
│       └── production_predictions.csv
│
├── 📓 NOTEBOOKS
│   ├── 05_Results_Summary.ipynb
│   ├── 06_TSNE_Parameter_Debug.ipynb
│   └── 07_Producao_Demo.ipynb
│
├── 📊 TRACKING
│   ├── mlruns/                       # Histórico de experimentos
│   └── mlflow.db                     # Banco de dados MLflow
│
└── 🐍 AMBIENTE
    └── venv/                         # Ambiente virtual Python
```

---

## 🚀 Como Começar

### 1. Preparar Ambiente

```bash
# Clone ou entre no repositório
cd Projeto_machine_learning_mlflow

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instale as dependências
pip install -r requirements.txt
```

### 2. Opções de Execução

#### **Opção A: Menu Interativo (Recomendado)**
```bash
python scripts/dev.py
```
Abre menu interativo com 5 opções:
1. 🚀 Treinar novo modelo (completo)
2. 📊 Servir API Flask (http://localhost:5001)
3. 🎨 Servir Dashboard Streamlit (http://localhost:8501)
4. 📈 Visualizar Experimentos MLflow (http://localhost:5000)
5. 🌳 Ver estrutura do projeto

#### **Opção B: Pipeline Completo**
```bash
python main.py
```
Executa 6 partes sequencialmente:
- Parte 1: Verificação de estrutura
- Parte 2: Fundação de dados
- Parte 3: Experimentação de modelos
- Parte 4: Redução de dimensionalidade
- Parte 5: Consolidação
- Parte 6: Operacionalização

#### **Opção C: Flask REST API**
```bash
python api/app.py
```
Inicia servidor na porta 5001 com 6 endpoints:
- `GET /health` - Status do servidor
- `GET /info` - Informações do modelo
- `POST /predict` - Predição única
- `POST /predict/batch` - Múltiplas predições
- `POST /predict/csv` - Upload CSV
- `GET /example` - Dados de exemplo

#### **Opção D: Streamlit Dashboard**
```bash
streamlit run dashboard/📄_Home.py
```
Inicia interface web na porta 8501 com 4 páginas:
- 🏠 **Home**: Visão geral do projeto
- 📊 **Predição**: Interface manual ou presets
- 📈 **Batch**: Upload CSV e análise
- 📉 **Monitoramento**: Métricas e alertas

#### **Opção E: MLflow Tracking**
```bash
mlflow ui
```
Visualiza todos os experimentos em http://localhost:5000

---

## 📋 Partes do Projeto

### ✅ PARTE 1: Estruturação do Projeto
- Definição técnica do objetivo
- Critérios de sucesso (métricas)
- Organização modular do código
- Separação: Notebooks são só exploração, lógica em módulos

### ✅ PARTE 2: Fundação de Dados
- **Diagnóstico de qualidade**: Valores ausentes, ruído, viés
- **Análise de desbalanceamento**: 78% bons pagadores vs 22% inadimplentes
- **Problemas identificados**: Dados finitos, sem erros críticos
- **Output**: Dados preparados e compreendidos

### ✅ PARTE 3: Experimentação de Modelos
- 4 modelos candidatos: Perceptron, Decision Tree, Random Forest, Logistic Regression
- Treinamento com GridSearchCV (otimização de hiperparâmetros)
- Validação cruzada estratificada
- **Rastreamento no MLflow**: Cada rodada registra parâmetros, métricas e modelo

### ✅ PARTE 4: Redução de Dimensionalidade
- **PCA**: Redução linear preservando variância
- **LDA**: Redução supervisionada (separabilidade entre classes)
- **t-SNE**: Visualização 2D
- **Comparação**: Desempenho vs redução de features

### ✅ PARTE 5: Consolidação
- Análise comparativa via MLflow
- Seleção técnica do modelo final
- Justificativa baseada em métricas

### ✅ PARTE 6: Operacionalização
- Persistência do modelo (joblib)
- Inferência em batch
- Monitoramento de drift (dados e modelo)
- Simulation de produção com métricas de negócio

---

## 📊 Dataset

**Fonte:** Clientes de cartão de crédito em Taiwan

| Propriedade | Valor |
|-------------|-------|
| Linhas | ~30.000 |
| Features | 23 |
| Target | `default.payment.next.month` (binária) |
| Desbalanceamento | 78% / 22% |

### Features principais:
- **Demográficas**: Sexo, Educação, Estado civil, Idade
- **Financeiras**: Limite de crédito, Valores de fatura, Valores pagos
- **Histórico**: 6 meses de status de pagamento (PAY_0 até PAY_6)

---

## 🧪 Modelos Candidatos

| Modelo | Tipo | Vantagens | Limitações |
|--------|------|-----------|-----------|
| **Perceptron** | Linear | Simples, rápido | Underfitting |
| **Decision Tree** | Árvore | Interpretável | Overfitting na profundidade |
| **Random Forest** | Ensemble | Robusto, non-linear | Menos interpretável |
| **Logistic Regression** | Linear | Rápido, bem calibrado | Linear |

---

## 📈 Métricas de Avaliação

- **Acurácia**: Proporção de predições corretas (⚠️ não é suficiente devido ao desbalanceamento)
- **Precision**: De todos os que previmos como inadimplentes, quantos realmente são? (FP importante)
- **Recall**: De todos os inadimplentes, quantos detectamos? (FN crítico no negócio)
- **F1-Score**: Média harmônica de Precision e Recall
- **ROC-AUC**: Área sob a curva ROC (melhor métrica para desbalanceamento)

### Critério de Sucesso
- ✅ Recall ≥ 0.70 (detectar ao menos 70% dos inadimplentes)
- ✅ ROC-AUC ≥ 0.80 (boa separabilidade entre classes)

---

## 🔍 Redução de Dimensionalidade

### Motivação
- Features originais: 23
- Possível colinearidade
- Custo computacional
- Overfitting

### Técnicas Aplicadas

**PCA (Principal Component Analysis)**
- Redução linear não-supervisionada
- Preserva variância máxima
- Útil para visualização e eficiência

**LDA (Linear Discriminant Analysis)**
- Redução supervisionada (usa target)
- Maximiza separabilidade entre classes
- Importante para dados desbalanceados

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**
- Visualização em 2D/3D
- Preserva estrutura local
- Não para treinamento (não invertível)

---

## 📁 Arquivos Gerados

### Em `models/`
- `perceptron.pkl`: Modelo do Perceptron
- `decision_tree.pkl`: Árvore otimizada
- `random_forest.pkl`: Ensemble final
- `logistic_regression.pkl`: Regressão logística
- `dimension_reducers.pkl`: PCA, LDA, t-SNE
- `training_results.csv`: Comparativo de modelos

### Em `results/`
- `dimensionality_comparison.csv`: Impacto da redução
- `model_selection.json`: Decisão técnica
- `production_predictions.csv`: Predições em batch
- `mlflow.db`: Database do MLflow (rastreamento)

---

## 🔄 MLflow - Rastreamento de Experimentos

Todos os experimentos são registrados automaticamente no MLflow:

```bash
mlflow ui
# Acesse em http://localhost:5000
```

Rastreado para cada experimento:
- ✅ Parâmetros do modelo (algoritmo, hiperparâmetros)
- ✅ Métricas (Accuracy, Recall, ROC-AUC, Precision, F1)
- ✅ Matriz de confusão
- ✅ Artefatos (modelos .pkl, gráficos, dados)
- ✅ Data e hora de execução
- ✅ Tempo de treinamento

**Localização:** `mlflow.db` (SQLite local)

---

## 🚨 Monitoramento em Produção

### Detecção de Data Drift
Compara distribuição dos dados em produção vs treino:
- Teste Kolmogorov-Smirnov
- Z-score de desvios estatísticos

### Detecção de Model Drift
Monitora degradação de performance:
- Queda em Recall (não detecta inadimplentes)
- Queda em ROC-AUC (pior separação)
- Mudança em Precision

### Estratégia de Re-treinamento
- **Frequência**: Mensal ou ao detectar drift
- **Dados**: Últimas 3 meses + base histórica
- **Validação**: Comparar desempenho vs modelo em produção

---

## � Documentação

| Arquivo | Conteúdo |
|---------|----------|
| [RELATORIO_TECNICO.md](RELATORIO_TECNICO.md) | 📋 **PRINCIPAL** - Análise completa com decisões e justificativas |
| [NAVIGATION.md](NAVIGATION.md) | Mapa visual de navegação do projeto |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Design técnico detalhado (400+ linhas) |
| [docs/QUICK_START.md](docs/QUICK_START.md) | Guia acelerado |
| [VIDEO_PRODUCAO_GUIA.md](VIDEO_PRODUCAO_GUIA.md) | Guia passo-a-passo para demo em vídeo |
| [ROTEIRO_VIDEO.md](ROTEIRO_VIDEO.md) | Script e timing do vídeo |

### 📊 Relatórios em PDF

Os seguintes relatórios em PDF podem ser gerados automaticamente:

```bash
# Gerar Relatório Técnico com análise comparativa
python scripts/generate_technical_report.py
# Resultado: reports/relatorio_tecnico_YYYYMMDD_HHMMSS.pdf (8 páginas)
# Contém: Capa, Resumo Executivo, Gráficos ROC, Matrizes de Confusão,
#         Análise Random Forest, Decisões de Projeto, Próximos Passos
```

**Conteúdo do Relatório Técnico em PDF:**
1. ✅ Resumo Executivo com métricas
2. ✅ Comparação de 4 modelos (Perceptron, Decision Tree, Random Forest, Logistic Regression)
3. ✅ Gráficos ROC para cada modelo
4. ✅ Matrizes de confusão detalhadas
5. ✅ Análise completa do Random Forest selecionado
6. ✅ Decisões de projeto e justificativas
7. ✅ Recomendações para produção

---

## 🌐 APIs e Interfaces

### Flask REST API (`api/`)

Servidor robusto para integração em produção:

```bash
python api/app.py
# http://localhost:5001
```

**Endpoints:**
- `GET /health` - Health check
- `GET /info` - Informações do modelo
- `POST /predict` - Predição com features em JSON
- `POST /predict/batch` - Array de amostras
- `POST /predict/csv` - Upload de arquivo CSV
- `GET /example` - Dados de exemplo

**Respostas:**
```json
{
  "prediction": 0,
  "probability": 0.8234,
  "confidence": 82.34,
  "risk_level": "LOW",
  "timestamp": "2026-04-11T10:30:45.123456"
}
```

### Streamlit Dashboard (`dashboard/`)

Interface web intuitiva para usuários:

```bash
streamlit run dashboard/📄_Home.py
# http://localhost:8501
```

**Recursos:**
- 🏠 **Home**: Visão geral e documentação
- 📊 **Predição**: Entrada manual ou perfis pré-configurados
- 📈 **Batch**: Upload CSV, análise em massa
- 📉 **Monitoramento**: Métricas, alertas, drift detection

---

## 👤 Autor

**Francisco Bruno Lopes Grangeiro**

Projeto de Disciplina - Pós-Graduação em Machine Learning com Scikit-Learn

---

## 📄 Licença

[Consulte LICENSE](LICENSE)

---

## 🔗 Referências

- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Default+of+Credit+Card+Clients)
- MLflow: [mlflow.org](https://mlflow.org)
- Scikit-learn: [scikit-learn.org](https://scikit-learn.org)

---

**Última atualização:** Abril 2026
