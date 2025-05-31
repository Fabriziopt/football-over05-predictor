# Substitua a seção no tab1 após as métricas resumo
if predictions:
    # Separar previsões por categoria
    predictions_with_ml = [p for p in predictions if p['status'] == 'PREDICTED']
    predictions_limited = [p for p in predictions if p['status'] == 'LIMITED_DATA']
    predictions_no_data = [p for p in predictions if p['status'] == 'NO_DATA']
    
    # Métricas resumo (suas métricas atuais)
    col1, col2, col3, col4 = st.columns(4)
    
    total_games = len(predictions)
    high_confidence = len([p for p in predictions_with_ml if p['confidence'] > 70])
    over_predictions = len([p for p in predictions_with_ml if p['prediction'] == 'OVER 0.5'])
    avg_confidence = sum([p['confidence'] for p in predictions_with_ml]) / len(predictions_with_ml) if predictions_with_ml else 0
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎮 Total de Jogos</h3>
            <h1>{total_games}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Alta Confiança</h3>
            <h1>{high_confidence}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Over 0.5</h3>
            <h1>{over_predictions}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💯 Confiança Média</h3>
            <h1>{avg_confidence:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)
    
    # ✅ SEÇÃO 1: MELHORES APOSTAS
    st.subheader("🏆 Melhores Apostas (Alta Confiança)")
    
    best_bets = [p for p in predictions_with_ml if p['prediction'] == 'OVER 0.5' and p['confidence'] > 65]
    best_bets.sort(key=lambda x: x['confidence'], reverse=True)
    
    if best_bets:
        for i, pred in enumerate(best_bets[:10]):
            try:
                utc_time = datetime.strptime(pred['kickoff'][:16], '%Y-%m-%dT%H:%M')
                hora_portugal = utc_time.strftime('%H:%M')
            except:
                hora_portugal = pred['kickoff'][11:16]
            
            confidence_class = "accuracy-high" if pred['confidence'] > 75 else "accuracy-medium"
            
            # Mostrar features avançadas
            advanced_info = ""
            if 'advanced_features' in pred:
                adv = pred['advanced_features']
                advanced_info = f"""
                <p><strong>🧠 Análise Avançada:</strong></p>
                <p>• Consistência: Casa {adv.get('home_consistency', 0):.2f} | Fora {adv.get('away_consistency', 0):.2f}</p>
                <p>• Combined Score: {adv.get('combined_score', 0):.3f}</p>
                <p>• Momentum: Casa {adv.get('home_momentum', 0):.1%} | Fora {adv.get('away_momentum', 0):.1%}</p>
                """
            
            st.markdown(f"""
            <div class="prediction-card">
                <h3>⚽ {pred['home_team']} vs {pred['away_team']}</h3>
                <p><strong>🏆 Liga:</strong> {pred['league']} ({pred['country']})</p>
                <p><strong>🕐 Horário PT:</strong> {hora_portugal}</p>
                <hr style="opacity: 0.3;">
                <p><strong>🎯 Previsão ML:</strong> {pred['prediction']}</p>
                <p><strong>💯 Confiança:</strong> <span class="{confidence_class}">{pred['confidence']:.1f}%</span></p>
                <p><strong>📊 Probabilidades:</strong> Over {pred['probability_over']:.1f}% | Under {pred['probability_under']:.1f}%</p>
                {advanced_info}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("🤷 Nenhuma aposta OVER 0.5 com boa confiança encontrada hoje")
    
    # ✅ SEÇÃO 2: TODAS AS PREVISÕES ML
    st.subheader(f"🤖 Todas as Previsões ML ({len(predictions_with_ml)} jogos)")
    
    if predictions_with_ml:
        # Filtros
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_confidence = st.slider("Confiança mínima", 0, 100, 50)
        with col2:
            filter_prediction = st.selectbox("Filtrar por", ["Todas", "OVER 0.5", "UNDER 0.5"])
        with col3:
            sort_by = st.selectbox("Ordenar por", ["Confiança", "Horário", "Liga"])
        
        # Aplicar filtros
        filtered_predictions = predictions_with_ml.copy()
        
        if min_confidence > 0:
            filtered_predictions = [p for p in filtered_predictions if p['confidence'] >= min_confidence]
        
        if filter_prediction != "Todas":
            filtered_predictions = [p for p in filtered_predictions if p['prediction'] == filter_prediction]
        
        # Ordenar
        if sort_by == "Confiança":
            filtered_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        elif sort_by == "Horário":
            filtered_predictions.sort(key=lambda x: x['kickoff'])
        else:  # Liga
            filtered_predictions.sort(key=lambda x: x['league'])
        
        st.info(f"Mostrando {len(filtered_predictions)} de {len(predictions_with_ml)} jogos")
        
        # Mostrar jogos filtrados
        for pred in filtered_predictions:
            try:
                utc_time = datetime.strptime(pred['kickoff'][:16], '%Y-%m-%dT%H:%M')
                hora_portugal = utc_time.strftime('%H:%M')
            except:
                hora_portugal = pred['kickoff'][11:16]
            
            # Cor do card baseada na previsão
            card_color = "background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);" if pred['prediction'] == 'OVER 0.5' else "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);"
            
            confidence_class = "accuracy-high" if pred['confidence'] > 75 else "accuracy-medium" if pred['confidence'] > 60 else "accuracy-low"
            
            st.markdown(f"""
            <div style="{card_color} color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <h4>⚽ {pred['home_team']} vs {pred['away_team']}</h4>
                <p>🏆 {pred['league']} | 🕐 {hora_portugal} | 🌍 {pred['country']}</p>
                <p><strong>🎯 {pred['prediction']}</strong> | <span class="{confidence_class}">💯 {pred['confidence']:.1f}%</span></p>
                <p>📊 Over: {pred['probability_over']:.1f}% | Under: {pred['probability_under']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ✅ SEÇÃO 3: JOGOS COM DADOS LIMITADOS
    if predictions_limited:
        st.subheader(f"⚠️ Jogos com Dados Limitados ({len(predictions_limited)} jogos)")
        st.info("Estes times têm menos de 3 jogos no banco de dados")
        
        for pred in predictions_limited:
            try:
                utc_time = datetime.strptime(pred['kickoff'][:16], '%Y-%m-%dT%H:%M')
                hora_portugal = utc_time.strftime('%H:%M')
            except:
                hora_portugal = pred['kickoff'][11:16]
            
            home_games = pred['home_stats']['games']
            away_games = pred['away_stats']['games']
            
            st.markdown(f"""
            <div style="background: #ffc107; color: black; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                <h4>⚽ {pred['home_team']} vs {pred['away_team']}</h4>
                <p>🏆 {pred['league']} | 🕐 {hora_portugal}</p>
                <p>📊 Dados: Casa {home_games} jogos | Fora {away_games} jogos</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ✅ SEÇÃO 4: JOGOS SEM DADOS
    if predictions_no_data:
        st.subheader(f"❌ Jogos Sem Dados Históricos ({len(predictions_no_data)} jogos)")
        st.info("Estes times não estão no banco de dados de treinamento")
        
        for pred in predictions_no_data:
            try:
                utc_time = datetime.strptime(pred['kickoff'][:16], '%Y-%m-%dT%H:%M')
                hora_portugal = utc_time.strftime('%H:%M')
            except:
                hora_portugal = pred['kickoff'][11:16]
            
            st.markdown(f"""
            <div style="background: #dc3545; color: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                <h4>⚽ {pred['home_team']} vs {pred['away_team']}</h4>
                <p>🏆 {pred['league']} | 🕐 {hora_portugal}</p>
                <p>❌ Times não encontrados no banco de dados</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ✅ ESTATÍSTICAS FINAIS
    st.subheader("📊 Estatísticas do Dia")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Jogos com ML", len(predictions_with_ml))
        st.metric("Dados Limitados", len(predictions_limited))
    
    with col2:
        st.metric("Sem Dados", len(predictions_no_data))
        if predictions_with_ml:
            over_rate = (len([p for p in predictions_with_ml if p['prediction'] == 'OVER 0.5']) / len(predictions_with_ml)) * 100
            st.metric("Taxa Over ML", f"{over_rate:.1f}%")
    
    with col3:
        if predictions_with_ml:
            max_conf = max([p['confidence'] for p in predictions_with_ml])
            min_conf = min([p['confidence'] for p in predictions_with_ml])
            st.metric("Máx Confiança", f"{max_conf:.1f}%")
            st.metric("Mín Confiança", f"{min_conf:.1f}%")

else:
    st.info("🤷 Nenhuma previsão disponível (times sem dados históricos)")
