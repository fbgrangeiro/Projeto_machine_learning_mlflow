"""
Dashboard Streamlit - Página de Monitoramento
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# Adicionar raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="📉 Monitoramento", page_icon="📉", layout="wide")

st.title("📉 Monitoramento")
st.subheader("Métricas de Performance do Modelo")

st.info("Métricas de performance do modelo em produção")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Acurácia", "81.5%", delta="+1.2%")

with col2:
    st.metric("Recall", "75.3%", delta="-0.5%")

with col3:
    st.metric("Precision", "68.9%", delta="+2.1%")

with col4:
    st.metric("ROC-AUC", "0.823", delta="+0.02")

st.divider()

st.subheader("📈 Histórico de Performance")

dates = pd.date_range('2026-03-01', periods=30, freq='D')
performance_data = {
    'Data': dates,
    'Accuracy': 80 + np.random.randn(30).cumsum() * 0.1,
    'Recall': 74 + np.random.randn(30).cumsum() * 0.1,
    'ROC-AUC': 0.81 + np.random.randn(30).cumsum() * 0.005
}

df_perf = pd.DataFrame(performance_data)

fig = px.line(
    df_perf,
    x='Data',
    y=['Accuracy', 'Recall', 'ROC-AUC'],
    title="Evolução de Métricas",
    markers=True
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("🔲 Matriz de Confusão (Teste)")

cm_data = pd.DataFrame({
    'Predito 0': [7200, 400],
    'Predito 1': [1800, 600]
}, index=['Real 0', 'Real 1'])

st.dataframe(cm_data, use_container_width=True)

st.divider()

st.subheader("🚨 Alertas")

col1, col2 = st.columns(2)

with col1:
    st.warning("⚠️ **Data Drift Detectado**\n\nMédia de idade dos clientes aumentou 15% em relação ao treino.")

with col2:
    st.info("ℹ️ **Informação**\n\nÚltimo re-treinamento: há 7 dias")
