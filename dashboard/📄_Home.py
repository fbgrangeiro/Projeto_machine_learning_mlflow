"""
Dashboard Streamlit - Entry Point
Interface principal da aplicação
"""
import streamlit as st

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="💳 Predição de Inadimplência",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("🎯 Menu Principal")

st.sidebar.divider()

st.sidebar.title("ℹ️ Informações")
st.sidebar.info("""
**Projeto de Machine Learning**
- Modelo: Random Forest
- Objetivo: Prever inadimplência
- Features: 23 variáveis
- Classes: Bom Pagador / Inadimplente

**Documentação:**
- [Arquitetura](../docs/ARCHITECTURE.md)
- [Quick Start](../docs/QUICK_START.md)
- [API](../docs/API_REFERENCE.md)
""")

# ============================================================================
# PÁGINA HOME
# ============================================================================

st.title("💳 Sistema de Predição de Inadimplência")
st.subheader("Machine Learning em Produção com Streamlit")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📊 Total de Features", "23")

with col2:
    st.metric("🎯 Classes", "2")

with col3:
    st.metric("⚡ Modelo", "Random Forest")

st.divider()

st.markdown("""
## 🎯 Objetivo

Prever se um cliente de cartão de crédito será inadimplente no próximo mês baseado em:
- **Dados Demográficos**: Sexo, Educação, Estado civil, Idade
- **Dados Financeiros**: Limite de crédito, Valores de fatura, Valores pagos
- **Histórico de Pagamento**: 6 meses de status

## 📌 Classes

- **Classe 0 (Bom Pagador)**: Baixo risco, cliente confiável
- **Classe 1 (Inadimplente)**: RISCO, cliente pode não pagar

## 🚀 Como Usar

Navegue pelos abas à esquerda (ou use o menu):

1. **📊 Predição**: Avalie um cliente individual
2. **📈 Análise em Batch**: Processe múltiplos clientes via CSV
3. **📉 Monitoramento**: Veja métricas de performance
4. **🔧 Configurações**: Ajuste parâmetros da aplicação

## 📊 Métricas de Sucesso

- **Recall ≥ 70%**: Detectar ao menos 70% dos inadimplentes
- **ROC-AUC ≥ 0.80**: Boa separabilidade entre classes
""")

st.divider()

st.markdown("### 📚 Features do Sistema")

features_info = {
    "🎨 Interface Intuitiva": "Dashboard clean e responsivo",
    "⚡ Predições em Tempo Real": "Resultados instantâneos",
    "📈 Análise em Batch": "Processe múltiplas amostras",
    "📊 Visualizações": "Gráficos interativos com Plotly",
    "💾 Rastreamento": "Histórico com MLflow",
    "🔍 Monitoramento": "Alertas de drift de dados/modelo"
}

cols = st.columns(2)
for idx, (feature, desc) in enumerate(features_info.items()):
    cols[idx % 2].write(f"**{feature}**\n{desc}")

st.divider()

st.markdown("""
### 🔗 Links Úteis

- [📖 Documentação Completa](../docs/ARCHITECTURE.md)
- [🔌 API REST](../api/app.py)
- [📖 Guia Rápido](../NAVIGATION.md)
- [⚙️ Configuração](../config/config.py)
""")
