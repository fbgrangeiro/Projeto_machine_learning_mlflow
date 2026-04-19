"""
Dashboard Streamlit - Página de Predição Individual
"""
import sys
from pathlib import Path
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Adicionar raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils import load_model, make_prediction

st.set_page_config(page_title="📊 Predição", page_icon="📊", layout="wide")

st.title("📊 Predição Individual")
st.subheader("Avalie o risco de inadimplência de um cliente")

# Carregar modelo
model, scaler = load_model()

st.info("Preencha os dados do cliente para obter uma predição")

tab1, tab2 = st.tabs(["📝 Entrada Manual", "📋 Preenchimento Automático"])

with tab1:
    st.subheader("Dados do Cliente")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📋 Dados Demográficos")
        limit_bal = st.number_input("Limite de Crédito (LIMIT_BAL)", min_value=0, value=150000, step=10000)
        sex = st.radio("Sexo (SEX)", options=[1, 2], format_func=lambda x: "Masculino" if x == 1 else "Feminino")
        education = st.radio("Educação (EDUCATION)", options=[0, 1, 2, 3, 4, 5, 6], 
                            format_func=lambda x: ["Outro", "Pós-Graduação", "Universidade", "Ensino Médio", "Outro1", "Outro2", "Desconhecido"][x])
    
    with col2:
        st.subheader("👨‍👩‍👧 Dados Pessoais")
        marriage = st.radio("Estado Civil (MARRIAGE)", options=[0, 1, 2, 3],
                           format_func=lambda x: ["Outro", "Casado", "Solteiro", "Divorciado"][x])
        age = st.slider("Idade (AGE)", min_value=21, max_value=79, value=35)
    
    with col3:
        st.subheader("💳 Histórico (Últimos 6 meses)")
        st.caption("(-1: Pago, 0: Sem consumo, 1-9: Meses de atraso)")
        
        pay_0 = st.slider("PAY_0 (mês -1)", min_value=-2, max_value=8, value=-1)
        pay_2 = st.slider("PAY_2 (mês -2)", min_value=-2, max_value=8, value=-1)
        pay_3 = st.slider("PAY_3 (mês -3)", min_value=-2, max_value=8, value=-1)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Valores de Fatura (últimos 6 meses)")
        bill_amt1 = st.number_input("BILL_AMT1 (mês -1)", min_value=-200000, value=50000, step=1000)
        bill_amt2 = st.number_input("BILL_AMT2 (mês -2)", min_value=-200000, value=45000, step=1000)
        bill_amt3 = st.number_input("BILL_AMT3 (mês -3)", min_value=-200000, value=40000, step=1000)
        bill_amt4 = st.number_input("BILL_AMT4 (mês -4)", min_value=-200000, value=35000, step=1000)
        bill_amt5 = st.number_input("BILL_AMT5 (mês -5)", min_value=-200000, value=30000, step=1000)
        bill_amt6 = st.number_input("BILL_AMT6 (mês -6)", min_value=-200000, value=25000, step=1000)
    
    with col2:
        st.subheader("💵 Valores Pagos (últimos 6 meses)")
        pay_amt1 = st.number_input("PAY_AMT1 (mês -1)", min_value=0, value=5000, step=500)
        pay_amt2 = st.number_input("PAY_AMT2 (mês -2)", min_value=0, value=4500, step=500)
        pay_amt3 = st.number_input("PAY_AMT3 (mês -3)", min_value=0, value=4000, step=500)
        pay_amt4 = st.number_input("PAY_AMT4 (mês -4)", min_value=0, value=3500, step=500)
        pay_amt5 = st.number_input("PAY_AMT5 (mês -5)", min_value=0, value=3000, step=500)
        pay_amt6 = st.number_input("PAY_AMT6 (mês -6)", min_value=0, value=2500, step=500)
    
    pay_4 = st.slider("PAY_4 (mês -4)", min_value=-2, max_value=8, value=-1, key="pay_4_manual")
    pay_5 = st.slider("PAY_5 (mês -5)", min_value=-2, max_value=8, value=-1, key="pay_5_manual")
    pay_6 = st.slider("PAY_6 (mês -6)", min_value=-2, max_value=8, value=-1, key="pay_6_manual")
    
    if st.button("🔮 Fazer Predição", type="primary", use_container_width=True):
        features = np.array([[
            limit_bal, sex, education, marriage, age,
            pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
            bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6,
            pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6
        ]])
        
        prediction, probability = make_prediction(model, features, scaler)
        
        st.divider()
        st.subheader("🎯 Resultado da Predição")
        
        pred_class = int(prediction[0])
        prob_0 = float(probability[0][0])
        prob_1 = float(probability[0][1])
        
        if pred_class == 0:
            color = "🟢"
            status = "Bom Pagador"
            risk_label = "Baixo Risco"
        else:
            color = "🔴"
            status = "Inadimplente"
            risk_label = "ALTO RISCO"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Classificação", f"{color} {status}")
        
        with col2:
            st.metric("⚠️ Nível de Risco", risk_label)
        
        with col3:
            st.metric("🎯 Confiança", f"{max(prob_0, prob_1):.1%}")
        
        st.divider()
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Bom Pagador', 'Inadimplente'],
                y=[prob_0, prob_1],
                marker_color=['#2ecc71', '#e74c3c'],
                text=[f'{prob_0:.1%}', f'{prob_1:.1%}'],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Distribuição de Probabilidade",
            xaxis_title="Classe",
            yaxis_title="Probabilidade",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        st.subheader("📋 Interpretação")
        
        if pred_class == 0:
            st.success(f"""
            ✅ **Cliente com Baixo Risco**
            
            - Probabilidade de ser bom pagador: **{prob_0:.1%}**
            - Recomendação: ✅ Aprovar crédito
            - Confiança do modelo: {max(prob_0, prob_1):.1%}
            """)
        else:
            st.error(f"""
            ⚠️ **Cliente em Risco de Inadimplência**
            
            - Probabilidade de inadimplência: **{prob_1:.1%}**
            - Recomendação: ❌ Analisar mais dados ou rejeitar crédito
            - Confiança do modelo: {max(prob_0, prob_1):.1%}
            """)

with tab2:
    st.subheader("Perfis Pré-configurados")
    
    profile = st.selectbox(
        "Escolha um perfil:",
        ["👤 Cliente Excelente", "👤 Cliente Regular", "⚠️ Cliente Risco"]
    )
    
    if profile == "👤 Cliente Excelente":
        features = {
            'LIMIT_BAL': 300000, 'SEX': 1, 'EDUCATION': 2, 'MARRIAGE': 1, 'AGE': 40,
            'PAY_0': -1, 'PAY_2': -1, 'PAY_3': -1, 'PAY_4': -1, 'PAY_5': -1, 'PAY_6': -1,
            'BILL_AMT1': 20000, 'BILL_AMT2': 18000, 'BILL_AMT3': 15000, 'BILL_AMT4': 12000, 'BILL_AMT5': 10000, 'BILL_AMT6': 8000,
            'PAY_AMT1': 30000, 'PAY_AMT2': 25000, 'PAY_AMT3': 20000, 'PAY_AMT4': 15000, 'PAY_AMT5': 12000, 'PAY_AMT6': 10000,
        }
        desc = "Cliente com histórico perfeito de pagamentos e limite alto"
    elif profile == "👤 Cliente Regular":
        features = {
            'LIMIT_BAL': 150000, 'SEX': 2, 'EDUCATION': 2, 'MARRIAGE': 1, 'AGE': 35,
            'PAY_0': 0, 'PAY_2': 0, 'PAY_3': 0, 'PAY_4': -1, 'PAY_5': -1, 'PAY_6': -1,
            'BILL_AMT1': 50000, 'BILL_AMT2': 48000, 'BILL_AMT3': 45000, 'BILL_AMT4': 42000, 'BILL_AMT5': 40000, 'BILL_AMT6': 38000,
            'PAY_AMT1': 5000, 'PAY_AMT2': 4800, 'PAY_AMT3': 4500, 'PAY_AMT4': 4200, 'PAY_AMT5': 4000, 'PAY_AMT6': 3800,
        }
        desc = "Cliente com pagamentos regulares e sem atrasos"
    else:
        features = {
            'LIMIT_BAL': 50000, 'SEX': 1, 'EDUCATION': 1, 'MARRIAGE': 2, 'AGE': 55,
            'PAY_0': 2, 'PAY_2': 2, 'PAY_3': 1, 'PAY_4': 1, 'PAY_5': 1, 'PAY_6': 0,
            'BILL_AMT1': 80000, 'BILL_AMT2': 78000, 'BILL_AMT3': 76000, 'BILL_AMT4': 74000, 'BILL_AMT5': 72000, 'BILL_AMT6': 70000,
            'PAY_AMT1': 1000, 'PAY_AMT2': 900, 'PAY_AMT3': 800, 'PAY_AMT4': 700, 'PAY_AMT5': 600, 'PAY_AMT6': 500,
        }
        desc = "Cliente com múltiplos atrasos e dívida crescente"
    
    st.info(desc)
    st.json(features)
    
    if st.button("🔮 Analisar Perfil", type="primary", use_container_width=True):
        feature_list = [
            features['LIMIT_BAL'], features['SEX'], features['EDUCATION'], 
            features['MARRIAGE'], features['AGE'],
            features['PAY_0'], features['PAY_2'], features['PAY_3'], 
            features['PAY_4'], features['PAY_5'], features['PAY_6'],
            features['BILL_AMT1'], features['BILL_AMT2'], features['BILL_AMT3'],
            features['BILL_AMT4'], features['BILL_AMT5'], features['BILL_AMT6'],
            features['PAY_AMT1'], features['PAY_AMT2'], features['PAY_AMT3'],
            features['PAY_AMT4'], features['PAY_AMT5'], features['PAY_AMT6']
        ]
        
        X = np.array([feature_list])
        prediction, probability = make_prediction(model, X, scaler)
        
        st.divider()
        
        pred_class = int(prediction[0])
        prob_0 = float(probability[0][0])
        prob_1 = float(probability[0][1])
        
        if pred_class == 0:
            st.success(f"✅ Classificado como **Bom Pagador** (confiança: {prob_0:.1%})")
        else:
            st.error(f"⚠️ Classificado como **Inadimplente** (confiança: {prob_1:.1%})")
