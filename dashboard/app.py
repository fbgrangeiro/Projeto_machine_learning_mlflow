"""
Dashboard Streamlit reorganizado
Raiz para multi-página Streamlit
"""
import streamlit as st

# Setup configuração
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
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎯 Menu Principal")
page = st.sidebar.radio(
    "Selecione uma opção:",
    ["🏠 Home", "📊 Predição", "📈 Análise em Batch", "📉 Monitoramento"],
    index=0
)

st.sidebar.divider()
st.sidebar.title("ℹ️ Informações")
st.sidebar.info("""
**Projeto de Machine Learning**
- Modelo: Random Forest
- Objetivo: Prever inadimplência
- Features: 23 variáveis
- Classes: Bom Pagador / Inadimplente
""")

# ============================================================================
# PÁGINA HOME
# ============================================================================

if page == "🏠 Home":
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
    Prever se um cliente de cartão de crédito será inadimplente no próximo mês.
    
    ## 📌 Classes
    - **Classe 0 (Bom Pagador)**: Baixo risco
    - **Classe 1 (Inadimplente)**: RISCO
    
    ## 🚀 Como Usar
    1. **Predição Individual**: Vá para "📊 Predição"
    2. **Análise em Batch**: Faça upload de um arquivo CSV
    3. **Monitoramento**: Veja métricas de performance
    """)

# ============================================================================
# PÁGINA PREDIÇÃO
# ============================================================================

elif page == "📊 Predição":
    st.title("📊 Predição Individual")
    st.subheader("Avalie o risco de inadimplência de um cliente")
    
    from dashboard.utils import load_model, make_prediction
    import numpy as np
    import plotly.graph_objects as go
    
    model, scaler = load_model()
    st.info("Preencha os dados do cliente para obter uma predição")
    
    # Entrada de dados (simplificada)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        limit_bal = st.number_input("Limite de Crédito", min_value=0, value=150000, step=10000)
        sex = st.radio("Sexo", options=[1, 2], format_func=lambda x: "M" if x == 1 else "F")
        age = st.slider("Idade", min_value=21, max_value=79, value=35)
    
    with col2:
        education = st.radio("Educação", options=[1, 2, 3])
        marriage = st.radio("Estado Civil", options=[1, 2, 3])
    
    with col3:
        bill_amt1 = st.number_input("Fatura (mês -1)", min_value=0, value=50000, step=1000)
        pay_amt1 = st.number_input("Pagamento (mês -1)", min_value=0, value=5000, step=500)
    
    if st.button("🔮 Fazer Predição", type="primary", use_container_width=True):
        # Dados completos (23 features)
        features = np.array([[
            limit_bal, sex, education, marriage, age,
            -1, -1, -1, -1, -1, -1,
            bill_amt1, 45000, 40000, 35000, 30000, 25000,
            pay_amt1, 4500, 4000, 3500, 3000, 2500
        ]])
        
        prediction, probability = make_prediction(model, features, scaler)
        
        st.divider()
        pred_class = int(prediction[0])
        prob_0 = float(probability[0][0])
        prob_1 = float(probability[0][1])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Classificação", "✅ Bom" if pred_class == 0 else "❌ Risco")
        with col2:
            st.metric("Confiança", f"{max(prob_0, prob_1):.1%}")
        with col3:
            st.metric("Probabilidade Risco", f"{prob_1:.1%}")

# ============================================================================
# PÁGINA ANÁLISE EM BATCH
# ============================================================================

elif page == "📈 Análise em Batch":
    st.title("📈 Análise em Batch")
    st.info("Faça upload de um CSV com 23 colunas de features")
    
    uploaded_file = st.file_uploader("📁 Upload CSV", type=['csv'])
    
    if uploaded_file:
        import pandas as pd
        from dashboard.utils import load_model, make_prediction
        
        df = pd.read_csv(uploaded_file)
        st.write(f"**Arquivo:** {len(df)} linhas, {len(df.columns)} colunas")
        
        if len(df.columns) == 23:
            if st.button("🔮 Processar"):
                model, scaler = load_model()
                predictions, probabilities = make_prediction(model, df.values, scaler)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total", len(df))
                with col2:
                    st.metric("Bom Pagador", int((predictions == 0).sum()))
                with col3:
                    st.metric("Risco", int((predictions == 1).sum()))

# ============================================================================
# PÁGINA MONITORAMENTO
# ============================================================================

elif page == "📉 Monitoramento":
    st.title("📉 Monitoramento")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Acurácia", "81.5%")
    with col2:
        st.metric("Recall", "75.3%")
    with col3:
        st.metric("Precision", "68.9%")
    with col4:
        st.metric("ROC-AUC", "0.823")
