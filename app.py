import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Importar suas classes do arquivo prediction_system.py
from prediction_system import SmartPredictionSystem, BacktestingEngine, example_usage

# Configuração da página
st.set_page_config(page_title="Football Over 0.5 HT Predictor", page_icon="⚽")

# Título
st.title("⚽ Football Over 0.5 HT Predictor")
st.markdown("Sistema de Predição Inteligente com Backtesting 365 dias")

# Criar instância do sistema
system = SmartPredictionSystem()

# Executar exemplo
if st.button("🚀 Executar Exemplo de Predição"):
    with st.spinner("Executando..."):
        result = example_usage()
        
        st.success("Predição concluída!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Taxa Final", f"{result['confidence_percentage']:.1f}%")
        
        with col2:
            st.metric("Probabilidade", f"{result['final_probability']:.2f}")
        
        st.subheader("Breakdown da Análise")
        st.json(result['breakdown'])
        
        st.subheader("Detalhes da Análise")
        st.json(result['analysis_details'])
