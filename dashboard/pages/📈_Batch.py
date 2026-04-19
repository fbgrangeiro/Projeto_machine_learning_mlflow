"""
Dashboard Streamlit - Página de Análise em Batch
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# Adicionar raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils import load_model, make_prediction

st.set_page_config(page_title="📈 Análise em Batch", page_icon="📈", layout="wide")

st.title("📈 Análise em Batch")
st.subheader("Processe múltiplos clientes de uma vez")

model, scaler = load_model()

uploaded_file = st.file_uploader(
    "📁 Faça upload de um arquivo CSV",
    type=['csv'],
    help="O arquivo deve ter exatamente 23 colunas com as features"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.write(f"**Arquivo carregado:** {len(df)} linhas, {len(df.columns)} colunas")
    
    if len(df.columns) != 23:
        st.error(f"❌ O arquivo deve ter 23 colunas, mas tem {len(df.columns)}")
    else:
        if st.button("🔮 Processar Lote", type="primary", use_container_width=True):
            with st.spinner("Processando..."):
                predictions, probabilities = make_prediction(model, df.values, scaler)
                
                df['prediction'] = predictions
                df['prob_class_0'] = probabilities[:, 0]
                df['prob_class_1'] = probabilities[:, 1]
                df['confidence'] = np.max(probabilities, axis=1)
                df['risk_level'] = df['prediction'].apply(
                    lambda x: "🔴 Alto Risco" if x == 1 else "🟢 Baixo Risco"
                )
                
                st.divider()
                st.subheader("📊 Resumo")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total de Amostras", len(df))
                
                with col2:
                    st.metric("Bom Pagador", int((predictions == 0).sum()))
                
                with col3:
                    st.metric("Inadimplente", int((predictions == 1).sum()))
                
                with col4:
                    pct_risco = (predictions == 1).sum() / len(predictions) * 100
                    st.metric("% em Risco", f"{pct_risco:.1f}%")
                
                st.divider()
                
                fig = px.pie(
                    names=['Bom Pagador', 'Inadimplente'],
                    values=[(predictions == 0).sum(), (predictions == 1).sum()],
                    color=['rgb(46, 204, 113)', 'rgb(231, 76, 60)'],
                    title="Distribuição de Predições"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                st.subheader("📋 Resultados Detalhados")
                
                st.dataframe(
                    df[['prediction', 'risk_level', 'confidence', 'prob_class_0', 'prob_class_1']],
                    use_container_width=True
                )
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Baixar Resultados (CSV)",
                    data=csv,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
