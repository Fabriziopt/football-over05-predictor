def predict_with_strategy(fixtures, league_models, min_confidence=60, no_limit=True):
    """Faz previsões com estratégia inteligente e tratamento robusto - SEM LIMITE de apostas por dia"""
    
    if not league_models:
        return []
    
    predictions = []
    
    for fixture in fixtures:
        try:
            if fixture['fixture']['status']['short'] not in ['NS', 'TBD']:
                continue
            
            league_id = fixture['league']['id']
            
            if league_id not in league_models:
                continue
            
            model_data = league_models[league_id]
            model = model_data['model']
            scaler = model_data['scaler']
            feature_cols = model_data['feature_cols']
            team_stats = model_data['team_stats']
            league_over_rate = model_data['league_over_rate']
            best_threshold = model_data.get('best_threshold', 0.5)
            
            home_id = fixture['teams']['home']['id']
            away_id = fixture['teams']['away']['id']
            
            if home_id not in team_stats or away_id not in team_stats:
                continue
            
            home_stats = team_stats[home_id]
            away_stats = team_stats[away_id]
            
            # Calcular a dinâmica Casa vs Fora
            home_away_dynamic = analyze_home_away_dynamic(home_stats, away_stats)
            
            # Poisson calculation (modelo melhorado)
            home_expected = home_stats['home_attack_strength'] * away_stats['away_defense_strength'] * 0.5
            away_expected = away_stats['away_attack_strength'] * home_stats['home_defense_strength'] * 0.5
            
            poisson_calc = calculate_poisson_probabilities(home_expected * 2, away_expected * 2, improved=True)
            
            # Calcular pontuação unificada
            unified = calculate_unified_score(
                home_stats, 
                away_stats, 
                league_over_rate,
                poisson_calc['poisson_over_05']
            )
            
            # Criar features básicas
            features = {
                'home_over_rate': home_stats['over_rate'],
                'away_over_rate': away_stats['over_rate'],
                'home_home_over_rate': home_stats['home_over_rate'],
                'away_away_over_rate': away_stats['away_over_rate'],
                'league_over_rate': league_over_rate,
                'home_attack_strength': home_stats['home_attack_strength'],
                'home_defense_strength': home_stats['home_defense_strength'],
                'away_attack_strength': away_stats['away_attack_strength'],
                'away_defense_strength': away_stats['away_defense_strength'],
                'poisson_over_05': poisson_calc['poisson_over_05'],
                'expected_goals_ht': poisson_calc['expected_goals_ht'],
                'prob_0_0': poisson_calc.get('prob_0_0', 1 - poisson_calc['poisson_over_05']),
                'prob_exact_1': poisson_calc.get('prob_exact_1', 0.3),
                'prob_2_plus': poisson_calc.get('prob_2_plus', 0.2),
                'combined_over_rate': (home_stats['home_over_rate'] + away_stats['away_over_rate']) / 2,
                'attack_index': (home_stats['home_attack_strength'] + away_stats['away_attack_strength']) / 2,
                'game_pace_index': (home_expected + away_expected),
                'over_rate_vs_league': ((home_stats['home_over_rate'] + away_stats['away_over_rate']) / 2) / max(league_over_rate, 0.01),
                'expected_vs_league': poisson_calc['expected_goals_ht'] / 0.5,
                'home_games_played': home_stats['home_games'],
                'away_games_played': away_stats['away_games'],
                'min_games': min(home_stats['home_games'], away_stats['away_games']),
                'unified_score': unified['unified_score'] / 100  # Normalizado para 0-1
            }
            
            # Adicionar features avançadas quando disponíveis
            if 'recent_over_rate' in home_stats:
                features['home_recent_over_rate'] = home_stats['recent_over_rate']
            if 'recent_over_rate' in away_stats:
                features['away_recent_over_rate'] = away_stats['recent_over_rate']
            if 'home_trend' in home_stats:
                features['home_trend'] = home_stats['home_trend']
            if 'away_trend' in away_stats:
                features['away_trend'] = away_stats['away_trend']
            
            # Adicionar dinâmica casa-fora
            features['home_dominance'] = home_away_dynamic.get('home_dominance', 1.0)
            features['away_threat'] = home_away_dynamic.get('away_threat', 1.0)
            features['matchup_balance'] = home_away_dynamic.get('matchup_balance', 0.5)
            
            # Adicionar tendência da liga se disponível
            if 'league_trend' in model_data:
                features['league_trend'] = model_data['league_trend'].get('overall_trend', 0)
            
            # Informações sobre outliers
            features['home_has_outliers'] = 1 if home_stats.get('home_has_outliers', False) else 0
            features['away_has_outliers'] = 1 if away_stats.get('away_has_outliers', False) else 0
            
            # Garantir que todas as features do modelo estão presentes
            for col in feature_cols:
                if col not in features:
                    features[col] = 0.0  # Valor neutro
            
            # Fazer previsão
            X = pd.DataFrame([features])[feature_cols]
            
            # Tratar missing features
            X = X.fillna(0.0)   # Tratar NaN
            
            X_scaled = scaler.transform(X)
            
            pred_proba = model.predict_proba(X_scaled)[0]
            confidence = pred_proba[1] * 100
            
            # Aplicar threshold otimizado
            pred_class = 1 if pred_proba[1] >= best_threshold else 0
            
            # Calcular indicadores de força
            game_vs_league_percent = (features['combined_over_rate'] / max(league_over_rate, 0.01) - 1) * 100
            
            # Calcular odds justas e análise de valor
            fair_odd = calculate_fair_odds(poisson_calc['poisson_over_05'] * 100)
            
            # Análise avançada do confronto
            matchup_analysis = {
                'home_dominance_score': features['home_dominance'] * 10,  # 0-10 escala
                'away_threat_score': features['away_threat'] * 10,  # 0-10 escala
                'matchup_balance': features['matchup_balance'],  # 0-1 onde 0.5 é equilibrado
                'expected_flow': 'Dominante em Casa' if features['matchup_balance'] < 0.4 else 
                               ('Equilíbrio' if features['matchup_balance'] <= 0.6 else 'Forte Fora')
            }
            
            # Usar a pontuação unificada para a decisão final
            final_confidence = unified['unified_score']
            final_risk = unified['risk_level']
            
            prediction = {
                'home_team': fixture['teams']['home']['name'],
                'away_team': fixture['teams']['away']['name'],
                'home_team_stats': home_stats,
                'away_team_stats': away_stats,
                'league': fixture['league']['name'],
                'country': fixture['league']['country'],
                'league_over_rate': league_over_rate * 100,
                'kickoff': fixture['fixture']['date'],
                'prediction': 'OVER 0.5' if pred_class == 1 else 'UNDER 0.5',
                'confidence': confidence,
                'unified_score': unified['unified_score'],
                'risk_level': unified['risk_level'],
                'ml_probability': pred_proba[1] * 100,
                'poisson_probability': poisson_calc['poisson_over_05'] * 100,
                'expected_goals_ht': poisson_calc['expected_goals_ht'],
                'game_vs_league_percent': game_vs_league_percent,
                'game_vs_league_ratio': features['combined_over_rate'] / max(league_over_rate, 0.01),
                'model_metrics': model_data['test_metrics'],
                'top_features': model_data['top_features'],
                'fair_odds': fair_odd,
                'matchup_analysis': matchup_analysis,
                'fixture_id': fixture['fixture']['id']
            }
            
            # Se estamos no modo sem limite, incluímos todos acima da confiança mínima
            if final_confidence >= min_confidence:
                predictions.append(prediction)
            
        except Exception as e:
            continue
    
    # Ordenar por pontuação unificada
    predictions.sort(key=lambda x: x['unified_score'], reverse=True)
    
    return predictions

def calculate_fair_odds(poisson_probability_percentage):
    """Calcula a odd justa baseada na probabilidade Poisson"""
    try:
        # Converter Poisson para probabilidade (0-1)
        probability = poisson_probability_percentage / 100
        
        # Odd justa = 1 / probabilidade
        fair_odd = 1 / probability
        
        return round(fair_odd, 2)
    except:
        return 0.0

def display_smart_prediction(pred):
    """Exibe previsão com análise inteligente e odd justa"""
    
    try:
        with st.container():
            # Header
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.subheader(f"⚽ {pred['home_team']} vs {pred['away_team']}")
            
            with col2:
                # Mostrar pontuação unificada
                if pred['unified_score'] > 75:
                    st.success(f"**{pred['unified_score']:.1f}**")
                elif pred['unified_score'] > 65:
                    st.info(f"**{pred['unified_score']:.1f}**")
                else:
                    st.warning(f"**{pred['unified_score']:.1f}**")
                st.caption(f"Risco: {pred['risk_level']}")
            
            with col3:
                # Comparação com liga
                if pred['game_vs_league_percent'] > 20:
                    st.write("🔥 **+{:.0f}%**".format(pred['game_vs_league_percent']))
                elif pred['game_vs_league_percent'] < -20:
                    st.write("❄️ **{:.0f}%**".format(pred['game_vs_league_percent']))
                else:
                    st.write("➖ **{:+.0f}%**".format(pred['game_vs_league_percent']))
            
            # Análise detalhada
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"🏆 **Liga:** {pred['league']} ({pred['country']})")
                st.write(f"📊 **Média da Liga:** {pred['league_over_rate']:.1f}%")
                
            with col2:
                st.write(f"🏠 **{pred['home_team']}**")
                st.write(f"- Casa: {pred['home_team_stats']['home_over_rate']*100:.1f}%")
                st.write(f"- Força Ataque: {pred['home_team_stats']['home_attack_strength']:.2f}")
                
            with col3:
                st.write(f"✈️ **{pred['away_team']}**")
                st.write(f"- Fora: {pred['away_team_stats']['away_over_rate']*100:.1f}%")
                st.write(f"- Força Ataque: {pred['away_team_stats']['away_attack_strength']:.2f}")
            
            # Análise de Confronto
            st.markdown("### 🔄 Dinâmica do Confronto")
            col1, col2 = st.columns(2)
            
            with col1:
                matchup = pred['matchup_analysis']
                flow = matchup['expected_flow']
                flow_icon = "🏠" if flow == "Dominante em Casa" else ("⚖️" if flow == "Equilíbrio" else "✈️")
                
                st.write(f"**Dinâmica:** {flow_icon} {flow}")
                st.write(f"**Dominância Casa:** {matchup['home_dominance_score']:.1f}/10")
                st.write(f"**Ameaça Fora:** {matchup['away_threat_score']:.1f}/10")
            
            with col2:
                # Calcular força do indicador Game vs Liga
                game_vs_league = pred['game_vs_league_percent']
                if game_vs_league > 20:
                    indicator = f"🔥 **Superior à Liga: +{game_vs_league:.1f}%**"
                elif game_vs_league < -20:
                    indicator = f"❄️ **Inferior à Liga: {game_vs_league:.1f}%**"
                else:
                    indicator = f"➖ **Próximo à Média: {game_vs_league:+.1f}%**"
                
                st.write(indicator)
                # Adicionar outras métricas relevantes
                st.write(f"**Jogo vs Liga:** {pred['game_vs_league_ratio']:.2f}x")
                
                # Indicador de confiança contextual
                if pred['unified_score'] > 70 and pred['game_vs_league_ratio'] > 1.1:
                    st.write("📈 **Alta Confiança Contextual**")
                elif pred['unified_score'] > 65:
                    st.write("🔍 **Confiança Moderada**")
                else:
                    st.write("⚠️ **Acompanhar Outros Fatores**")
            
            # Previsões com Odd Justa
            st.markdown("### 🎯 Análise Preditiva")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("ML Probability", f"{pred['ml_probability']:.1f}%")
            with col2:
                st.metric("Poisson Probability", f"{pred['poisson_probability']:.1f}%")
            with col3:
                st.metric("Gols Esperados HT", f"{pred['expected_goals_ht']:.2f}")
            with col4:
                # Calcular e mostrar odd justa baseada no Poisson
                fair_odd = pred['fair_odds']
                st.metric("💰 Odd Justa", f"{fair_odd}")
            
            # Recomendação com odd justa e análise de valor
            col1, col2 = st.columns(2)
            
            with col1:
                if pred['prediction'] == 'OVER 0.5':
                    if pred['unified_score'] > 70 and pred['game_vs_league_ratio'] > 1.1:
                        st.success(f"✅ **APOSTAR: {pred['prediction']} HT** (Alta Confiança) | **Odd Justa: {fair_odd}**")
                    else:
                        st.info(f"📊 **Considerar: {pred['prediction']} HT** (Confiança Moderada) | **Odd Justa: {fair_odd}**")
            
            with col2:
                # Guia de valor para aposta
                st.write("**💰 Guia de Valor:**")
                st.write(f"- Valor excelente: Odd > {(fair_odd*1.1):.2f}")
                st.write(f"- Valor bom: Odd > {fair_odd:.2f}")
                st.write(f"- Sem valor: Odd < {(fair_odd*0.9):.2f}")
            
            # Informações sobre outliers
            if pred['home_team_stats'].get('home_has_outliers', False) or pred['away_team_stats'].get('away_has_outliers', False):
                st.warning("⚠️ **Atenção**: Detectados outliers nas estatísticas de uma ou ambas equipes. As previsões foram ajustadas para compensar.")
            
            # Detalhes adicionais
            with st.expander("📊 Detalhes Avançados"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🏠 Performance em Casa:**")
                    st.write(f"- Gols marcados: {pred['home_team_stats']['home_goals_scored']:.2f}")
                    st.write(f"- Gols sofridos: {pred['home_team_stats']['home_goals_conceded']:.2f}")
                    if 'recent_over_rate' in pred['home_team_stats']:
                        st.write(f"- Forma recente: {pred['home_team_stats']['recent_over_rate']*100:.1f}%")
                
                with col2:
                    st.write("**✈️ Performance Fora:**")
                    st.write(f"- Gols marcados: {pred['away_team_stats']['away_goals_scored']:.2f}")
                    st.write(f"- Gols sofridos: {pred['away_team_stats']['away_goals_conceded']:.2f}")
                    if 'recent_over_rate' in pred['away_team_stats']:
                        st.write(f"- Forma recente: {pred['away_team_stats']['recent_over_rate']*100:.1f}%")
                
                st.write("**📊 Métricas do Modelo:**")
                st.write(f"- Acurácia: {pred['model_metrics']['accuracy']*100:.1f}%")
                st.write(f"- F1-Score: {pred['model_metrics']['f1_score']*100:.1f}%")
                st.write(f"- Precisão: {pred['model_metrics']['precision']*100:.1f}%")
                
                # Componentes da pontuação unificada
                st.write("**🔢 Componentes da Pontuação:**")
                components = pred.get('unified_score_components', {})
                if components:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Times", f"{components.get('team_score', 0):.1f}")
                    with col2:
                        st.metric("Poisson", f"{components.get('poisson_score', 0):.1f}")
                    with col3:
                        st.metric("vs Liga", f"{components.get('vs_league_score', 0):.1f}")
                    with col4:
                        st.metric("Tendência", f"{components.get('trend_recent_score', 0):.1f}")
            
            st.markdown("---")
            
    except Exception as e:
        st.error(f"❌ Erro ao exibir previsão: {str(e)}")

def create_excel_download(df, filename):
    """Cria arquivo Excel para download"""
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Dados')
        output.seek(0)
        return output.getvalue()
    except:
        return None

def display_league_summary(league_models):
    """Exibe resumo das ligas com visualizações avançadas"""
    
    try:
        st.header("📊 Análise das Ligas")
        
        if not league_models:
            st.warning("⚠️ Nenhum modelo treinado!")
            return
        
        # Dados para análise
        league_data = []
        for league_id, model_data in league_models.items():
            try:
                # Extrair tendência da liga se disponível
                league_trend = model_data.get('league_trend', {}).get('overall_trend', 0)
                trend_direction = "↗️" if league_trend > 0.05 else ("↘️" if league_trend < -0.05 else "→")
                
                # Tipo de modelo usado
                model_type = model_data.get('model_type', 'Desconhecido')
                if 'Voting' in model_type:
                    model_type = "🔄 Ensemble"
                elif 'Calibrated' in model_type:
                    model_type = "📊 Calibrado"
                
                league_data.append({
                    'Liga': model_data['league_name'],
                    'Over 0.5 HT %': round(model_data['league_over_rate'] * 100, 1),
                    'Tendência': trend_direction,
                    'Jogos': model_data['total_matches'],
                    'F1-Score': round(model_data['test_metrics']['f1_score'] * 100, 1),
                    'Acurácia': round(model_data['test_metrics']['accuracy'] * 100, 1),
                    'Precisão': round(model_data['test_metrics']['precision'] * 100, 1),
                    'Recall': round(model_data['test_metrics']['recall'] * 100, 1),
                    'Threshold Ótimo': round(model_data['best_threshold'], 3),
                    'Modelo': model_type
                })
            except:
                continue
        
        if not league_data:
            st.warning("⚠️ Nenhum dado para exibir!")
            return
        
        df_leagues = pd.DataFrame(league_data)
        
        # Estatísticas gerais
        avg_accuracy = df_leagues['Acurácia'].mean()
        avg_f1 = df_leagues['F1-Score'].mean()
        avg_over_rate = df_leagues['Over 0.5 HT %'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Ligas Analisadas", len(df_leagues))
        with col2:
            st.metric("Acurácia Média", f"{avg_accuracy:.1f}%")
        with col3:
            st.metric("F1-Score Médio", f"{avg_f1:.1f}%")
        with col4:
            st.metric("Over 0.5 HT Médio", f"{avg_over_rate:.1f}%")
        
        # Botão de download no topo
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            excel_data = create_excel_download(df_leagues, "analise_ligas.xlsx")
            if excel_data:
                st.download_button(
                    label="📥 Download Excel - Todas as Ligas",
                    data=excel_data,
                    file_name=f"analise_ligas_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        # Visualizações em colunas
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Top 15 Ligas - Taxa Over 0.5 HT")
            top_leagues = df_leagues.sort_values('Over 0.5 HT %', ascending=False).head(15)
            chart_data = top_leagues.set_index('Liga')['Over 0.5 HT %']
            st.bar_chart(chart_data)
            
            # Download Top 15
            excel_top = create_excel_download(top_leagues, "top_15_ligas.xlsx")
            if excel_top:
                st.download_button(
                    label="📥 Download Top 15 Ligas",
                    data=excel_top,
                    file_name=f"top_15_ligas_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col2:
            st.subheader("🎯 Performance dos Modelos")
            performance_data = df_leagues[['Liga', 'Acurácia', 'F1-Score']].set_index('Liga').head(15)
            st.line_chart(performance_data)
            
            # Download Performance
            perf_df = df_leagues[['Liga', 'Acurácia', 'F1-Score', 'Precisão', 'Recall', 'Modelo']].sort_values('F1-Score', ascending=False)
            excel_perf = create_excel_download(perf_df, "performance_modelos.xlsx")
            if excel_perf:
                st.download_button(
                    label="📥 Download Performance",
                    data=excel_perf,
                    file_name=f"performance_modelos_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        # Análise por continente/região
        st.subheader("🌍 Análise por Região")
        
        # Extrair região de cada liga (usando o país)
        def get_region(league_name):
            name_lower = league_name.lower()
            if any(region in name_lower for region in ['england', 'spain', 'italy', 'germany', 'france', 'portugal', 'netherlands', 'belgium']):
                return 'Europa Ocidental'
            elif any(region in name_lower for region in ['brazil', 'argentina', 'colombia', 'chile', 'mexico']):
                return 'América Latina'
            elif any(region in name_lower for region in ['usa', 'canada', 'mls']):
                return 'América do Norte'
            elif any(region in name_lower for region in ['japan', 'china', 'korea', 'australia']):
                return 'Ásia-Pacífico'
            elif any(region in name_lower for region in ['sweden', 'norway', 'denmark', 'finland', 'iceland']):
                return 'Europa Nórdica'
            elif any(region in name_lower for region in ['poland', 'czech', 'hungary', 'romania', 'bulgaria', 'croatia', 'serbia']):
                return 'Europa Oriental'
            else:
                return 'Outras Regiões'
        
        # Adicionar coluna de região
        df_leagues['Região'] = df_leagues['Liga'].apply(get_region)
        
        # Agrupar por região
        region_analysis = df_leagues.groupby('Região').agg({
            'Over 0.5 HT %': 'mean',
            'F1-Score': 'mean',
            'Acurácia': 'mean',
            'Liga': 'count'
        }).rename(columns={'Liga': 'Número de Ligas'}).reset_index()
        
        # Exibir análise por região
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(region_analysis.sort_values('Over 0.5 HT %', ascending=False))
        
        with col2:
            # Gráfico de barras para taxa over por região
            chart_data = region_analysis.set_index('Região')['Over 0.5 HT %'].sort_values(ascending=False)
            st.bar_chart(chart_data)
        
        # Tabela resumo
        st.subheader("📋 Resumo Detalhado de Todas as Ligas")
        df_display = df_leagues.sort_values('F1-Score', ascending=False)
        st.dataframe(df_display, use_container_width=True)
        
        # Análise de qualidade com downloads
        st.subheader("📊 Análise de Qualidade dos Modelos")
        
        high_quality = df_leagues[df_leagues['F1-Score'] >= 80]
        medium_quality = df_leagues[(df_leagues['F1-Score'] >= 70) & (df_leagues['F1-Score'] < 80)]
        low_quality = df_leagues[df_leagues['F1-Score'] < 70]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🟢 Alta Qualidade (F1 ≥ 80%)", len(high_quality))
            if len(high_quality) > 0:
                excel_high = create_excel_download(high_quality, "ligas_alta_qualidade.xlsx")
                if excel_high:
                    st.download_button(
                        label="📥 Download Alta Qualidade",
                        data=excel_high,
                        file_name=f"ligas_alta_qualidade_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="high_quality"
                    )
        
        with col2:
            st.metric("🟡 Média Qualidade (F1 70-80%)", len(medium_quality))
            if len(medium_quality) > 0:
                excel_med = create_excel_download(medium_quality, "ligas_media_qualidade.xlsx")
                if excel_med:
                    st.download_button(
                        label="📥 Download Média Qualidade",
                        data=excel_med,
                        file_name=f"ligas_media_qualidade_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="medium_quality"
                    )
        
        with col3:
            st.metric("🔴 Baixa Qualidade (F1 < 70%)", len(low_quality))
            if len(low_quality) > 0:
                excel_low = create_excel_download(low_quality, "ligas_baixa_qualidade.xlsx")
                if excel_low:
                    st.download_button(
                        label="📥 Download Baixa Qualidade",
                        data=excel_low,
                        file_name=f"ligas_baixa_qualidade_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="low_quality"
                    )
        
    except Exception as e:
        st.error(f"❌ Erro ao exibir resumo: {str(e)}")

def run_backtesting(league_models, historical_df, days_back=90, min_confidence=60):
    """
    Executa backtesting para avaliar o desempenho do modelo em dados históricos.
    
    Parâmetros:
    - league_models: dicionário com os modelos treinados
    - historical_df: DataFrame com dados históricos
    - days_back: quantos dias para trás analisar
    - min_confidence: confiança mínima para considerar uma aposta
    
    Retorna:
    - DataFrame com resultados do backtesting
    - Estatísticas de desempenho
    """
    if not league_models or historical_df.empty:
        return pd.DataFrame(), {"accuracy": 0, "total_bets": 0}
    
    try:
        # Filtrar jogos dentro do período
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Garantir que temos data em formato datetime
        if 'date' in historical_df.columns:
            historical_df['date'] = pd.to_datetime(historical_df['date'], errors='coerce')
            test_df = historical_df[(historical_df['date'] >= start_date) & 
                                  (historical_df['date'] <= end_date)].copy()
        else:
            # Se não tiver data, usa todo o dataset (menos ideal)
            test_df = historical_df.copy()
        
        if test_df.empty:
            return pd.DataFrame(), {"accuracy": 0, "total_bets": 0}
        
        # Preparar lista para resultados
        backtest_results = []
        
        # Agrupar por liga para facilitar o processamento
        league_groups = test_df.groupby('league_id')
        
        for league_id, league_df in league_groups:
            # Verificar se temos modelo para esta liga
            if league_id not in league_models:
                continue
                
            model_data = league_models[league_id]
            
            # Ordenar jogos por data se disponível
            if 'date' in league_df.columns:
                league_df = league_df.sort_values('date')
            
            # Para cada jogo na liga
            for idx, row in league_df.iterrows():
                try:
                    # Evitar usar o próprio jogo no treinamento (simular predição real)
                    # Na prática, deveria usar apenas jogos anteriores à data do jogo atual
                    # Aqui simplificamos usando o modelo treinado em todos os dados
                    
                    # Obter estatísticas dos times
                    home_id = row['home_team_id']
                    away_id = row['away_team_id']
                    
                    if home_id not in model_data['team_stats'] or away_id not in model_data['team_stats']:
                        continue
                        
                    home_stats = model_data['team_stats'][home_id]
                    away_stats = model_data['team_stats'][away_id]
                    
                    # Calcular previsão
                    # Poisson calculation
                    home_expected = home_stats['home_attack_strength'] * away_stats['away_defense_strength'] * 0.5
                    away_expected = away_stats['away_attack_strength'] * home_stats['home_defense_strength'] * 0.5
                    
                    poisson_calc = calculate_poisson_probabilities(home_expected * 2, away_expected * 2, improved=True)
                    
                    # Unificar pontuação
                    unified = calculate_unified_score(
                        home_stats, 
                        away_stats, 
                        model_data['league_over_rate'],
                        poisson_calc['poisson_over_05']
                    )
                    
                    # Determinar previsão baseada na pontuação unificada
                    confidence = unified['unified_score']
                    prediction = 1 if confidence >= min_confidence else 0
                    
                    # Resultado real
                    actual = row['over_05']
                    
                    # Armazenar resultado
                    backtest_results.append({
                        'date': row['date'] if 'date' in row else None,
                        'league_id': league_id,
                        'league_name': model_data['league_name'],
                        'home_team': row['home_team'],
                        'away_team': row['away_team'],
                        'prediction': prediction,
                        'actual': actual,
                        'correct': prediction == actual,
                        'confidence': confidence,
                        'unified_score': unified['unified_score'],
                        'risk_level': unified['risk_level'],
                        'vs_league': unified['vs_league_raw'],
                        'poisson_prob': poisson_calc['poisson_over_05'] * 100
                    })
                    
                except Exception as e:
                    continue
        
        # Criar DataFrame de resultados
        results_df = pd.DataFrame(backtest_results)
        
        if results_df.empty:
            return pd.DataFrame(), {"accuracy": 0, "total_bets": 0}
        
        # Calcular estatísticas
        stats = {
            "total_bets": len(results_df),
            "correct_bets": results_df['correct'].sum(),
            "accuracy": results_df['correct'].mean() * 100 if len(results_df) > 0 else 0,
            "avg_confidence": results_df['confidence'].mean() if len(results_df) > 0 else 0
        }
        
        # Adicionar estatísticas por nível de confiança
        confidence_levels = [
            (50, 60), (60, 70), (70, 80), (80, 90), (90, 101)
        ]
        
        for low, high in confidence_levels:
            mask = (results_df['confidence'] >= low) & (results_df['confidence'] < high)
            level_df = results_df[mask]
            
            if len(level_df) > 0:
                stats[f"conf_{low}_{high}_count"] = len(level_df)
                stats[f"conf_{low}_{high}_accuracy"] = level_df['correct'].mean() * 100
            else:
                stats[f"conf_{low}_{high}_count"] = 0
                stats[f"conf_{low}_{high}_accuracy"] = 0
        
        # Adicionar estatísticas por nível de vs_league
        vs_league_levels = [
            (-100, -20), (-20, 0), (0, 20), (20, 40), (40, 100)
        ]
        
        for low, high in vs_league_levels:
            mask = (results_df['vs_league'] >= low) & (results_df['vs_league'] < high)
            level_df = results_df[mask]
            
            if len(level_df) > 0:
                stats[f"vs_league_{low}_{high}_count"] = len(level_df)
                stats[f"vs_league_{low}_{high}_accuracy"] = level_df['correct'].mean() * 100
            else:
                stats[f"vs_league_{low}_{high}_count"] = 0
                stats[f"vs_league_{low}_{high}_accuracy"] = 0
        
        return results_df, stats
        
    except Exception as e:
        return pd.DataFrame(), {"accuracy": 0, "total_bets": 0, "error": str(e)}

def display_backtesting_results(results_df, stats):
    """
    Exibe os resultados do backtesting de forma visual e intuitiva.
    """
    if results_df.empty:
        st.warning("⚠️ Não há dados suficientes para o backtesting")
        return
    
    st.header("🧪 Resultados do Backtesting")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Apostas", stats["total_bets"])
    with col2:
        st.metric("Apostas Corretas", stats["correct_bets"])
    with col3:
        st.metric("Taxa de Acerto", f"{stats['accuracy']:.1f}%")
    with col4:
        st.metric("Confiança Média", f"{stats['avg_confidence']:.1f}")
    
    # Gráficos de desempenho
    st.subheader("📊 Desempenho por Nível de Confiança")
    
    # Preparar dados para gráfico de confiança
    conf_data = []
    for low, high in [(50, 60), (60, 70), (70, 80), (80, 90), (90, 101)]:
        count = stats.get(f"conf_{low}_{high}_count", 0)
        accuracy = stats.get(f"conf_{low}_{high}_accuracy", 0)
        
        if count > 0:
            conf_data.append({
                'Nível': f"{low}-{high-1}%",
                'Apostas': count,
                'Acerto': accuracy
            })
    
    if conf_data:
        conf_df = pd.DataFrame(conf_data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(conf_df.set_index('Nível')['Acerto'])
        with col2:
            st.bar_chart(conf_df.set_index('Nível')['Apostas'])
    
    # Gráfico vs_league
    st.subheader("📈 Desempenho por Diferença vs Liga")
    
    # Preparar dados para gráfico vs_league
    vs_league_data = []
    for low, high in [(-100, -20), (-20, 0), (0, 20), (20, 40), (40, 100)]:
        count = stats.get(f"vs_league_{low}_{high}_count", 0)
        accuracy = stats.get(f"vs_league_{low}_{high}_accuracy", 0)
        
        if count > 0:
            label = f"{low}% a {high}%"
            if low == -100:
                label = f"< {high}%"
            if high == 100:
                label = f"> {low}%"
                
            vs_league_data.append({
                'Faixa': label,
                'Apostas': count,
                'Acerto': accuracy
            })
    
    if vs_league_data:
        vs_league_df = pd.DataFrame(vs_league_data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(vs_league_df.set_index('Faixa')['Acerto'])
        with col2:
            st.bar_chart(vs_league_df.set_index('Faixa')['Apostas'])
    
    # Tabela de resultados recentes
    st.subheader("📋 Resultados Detalhados")
    
    # Ordenar por data se disponível
    if 'date' in results_df.columns:
        display_df = results_df.sort_values('date', ascending=False).head(50).copy()
    else:
        display_df = results_df.head(50).copy()
    
    # Formatar para exibição
    display_df['Resultado'] = display_df['correct'].map({True: "✅ Correto", False: "❌ Incorreto"})
    display_df['Confiança'] = display_df['confidence'].round(1).astype(str) + '%'
    display_df['vs Liga'] = display_df['vs_league'].round(1).astype(str) + '%'
    display_df['Data'] = display_df['date'].dt.strftime('%d/%m/%Y') if 'date' in display_df.columns else "N/A"
    
    # Selecionar colunas para exibição
    cols_to_show = ['Data', 'league_name', 'home_team', 'away_team', 
                   'Confiança', 'vs Liga', 'risk_level', 'Resultado']
    
    st.dataframe(display_df[cols_to_show], use_container_width=True)
    
    # Download dos resultados
    excel_data = create_excel_download(results_df, "backtesting_results.xlsx")
    if excel_data:
        st.download_button(
            label="📥 Download Resultados Completos",
            data=excel_data,
            file_name=f"backtesting_resultados_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

def main():
    st.title("⚽ HT Goals AI Ultimate - Sistema Avançado")
    st.markdown("🎯 **Versão Premium - Máxima taxa de acerto para Over 0.5 HT**")
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configurações")
        
        # Status da API
        try:
            conn_ok, conn_msg = test_api_connection()
            if conn_ok:
                st.success("✅ API conectada")
            else:
                st.error(f"❌ {conn_msg}")
        except:
            st.error("❌ Erro ao testar API")
        
        # Status dos modelos
        if st.session_state.models_trained and st.session_state.league_models:
            st.success(f"✅ {len(st.session_state.league_models)} ligas treinadas")
            
            # Botão para carregar backup se disponível
            if st.session_state.models_backup:
                if st.button("🔄 Carregar Backup"):
                    load_training_progress()
                    st.success("✅ Backup carregado!")
                    st.rerun()
        else:
            st.warning("⚠️ Modelos não treinados")
        
        st.markdown("---")
        
        # Configurações
        st.markdown("### 📊 Parâmetros")
        
        min_matches_per_league = st.slider(
            "Mínimo jogos por liga:",
            min_value=20,
            max_value=100,
            value=30,
            help="30 jogos é o equilíbrio ideal entre quantidade de ligas e qualidade dos modelos"
        )
        
        min_confidence = st.slider(
            "Confiança mínima:",
            min_value=50,
            max_value=80,
            value=60,
            help="60% permite mais oportunidades com boa taxa de acerto"
        )
        
        include_all_seasons = st.checkbox("📅 Múltiplas Temporadas", value=True,
                                       help="Inclui até 3 temporadas nos dados para análise mais completa")
        
        use_cache = st.checkbox("💾 Usar cache", value=True,
                             help="Usa dados salvos anteriormente para maior rapidez")
        
        # Mostrar erros de treinamento se houver
        if st.session_state.training_errors:
            with st.expander("⚠️ Erros de Treinamento"):
                for error in st.session_state.training_errors[-5:]:  # Últimos 5
                    st.write(f"• {error}")
    
    # Tabs principais
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🤖 Treinar", "📊 Análise Ligas", "🎯 Previsões", "📈 Dashboard", "📚 Documentação"])
    
    with tab1:
        st.header("🤖 Treinamento Avançado com Multi-Temporadas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **✅ Sistema Premium:**
            - Análise de múltiplas temporadas
            - Backup automático do progresso
            - Análise casa vs fora detalhada
            - Forma recente das equipes
            - Tendências de crescimento/queda
            """)
        
        with col2:
            st.success("""
            **🎯 Melhorias:**
            - Modelos ensemble para maior precisão
            - Calibração de probabilidades
            - Análise de dinâmica de confrontos
            - Métricas avançadas por região
            - Grid search para threshold ótimo
            """)
        
        if st.button("🚀 TREINAR SISTEMA AVANÇADO", type="primary", use_container_width=True):
            
            # Limpar erros anteriores
            st.session_state.training_errors = []
            st.session_state.training_in_progress = True
            
            try:
                with st.spinner("📥 Carregando dados históricos..."):
                    df = collect_historical_data_smart(
                        days=None, 
                        use_cached=use_cache, 
                        seasonal=True,
                        include_all_seasons=include_all_seasons
                    )
                
                if df.empty:
                    st.error("❌ Nenhum dado disponível para treinamento")
                    st.session_state.training_in_progress = False
                    st.stop()
                
                st.success(f"✅ {len(df)} jogos carregados")
                
                # Mostrar distribuição de temporadas
                if 'season' in df.columns:
                    seasons = df['season'].value_counts()
                    st.write("📅 **Distribuição por Temporada:**")
                    for season, count in seasons.items():
                        st.write(f"- {season}: {count} jogos")
                
                # Agrupar por liga
                league_groups = df.groupby(['league_id', 'league_name', 'country'])
                
                st.info(f"🎯 Encontradas {len(league_groups)} ligas para análise")
                
                league_models = {}
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results_summary = []
                successful_leagues = 0
                
                for idx, ((league_id, league_name, country), league_df) in enumerate(league_groups):
                    progress = (idx + 1) / len(league_groups)
                    progress_bar.progress(progress)
                    
                    league_full_name = f"{league_name} ({country})"
                    status_text.text(f"🔄 Treinando: {league_full_name}")
                    
                    if len(league_df) < min_matches_per_league:
                        continue
                    
                    # Treinar modelo avançado
                    model_data, message = train_complete_model_with_validation(
                        league_df, league_id, league_full_name, min_matches_per_league
                    )
                    
                    if model_data:
                        league_models[league_id] = model_data
                        successful_leagues += 1
                        st.success(message)
                        
                        results_summary.append({
                            'Liga': league_full_name,
                            'Jogos': len(league_df),
                            'Acurácia': model_data['test_metrics']['accuracy'],
                            'F1-Score': model_data['test_metrics']['f1_score'],
                            'Tipo Modelo': model_data.get('model_type', 'Desconhecido')
                        })
                        
                        # Salvar progresso a cada 5 ligas
                        if successful_leagues % 5 == 0:
                            save_training_progress(league_models, f"backup_{successful_leagues}")
                    else:
                        st.warning(message)
                
                progress_bar.empty()
                status_text.empty()
                
                if league_models:
                    # Salvar progresso final
                    save_training_progress(league_models, "final")
                    
                    st.session_state.league_models = league_models
                    st.session_state.models_trained = True
                    
                    # Resumo final
                    st.success(f"🎉 {len(league_models)} ligas treinadas com sucesso!")
                    
                    if results_summary:
                        avg_accuracy = np.mean([r['Acurácia'] for r in results_summary])
                        avg_f1 = np.mean([r['F1-Score'] for r in results_summary])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Ligas Treinadas", len(league_models))
                        with col2:
                            st.metric("Acurácia Média", f"{avg_accuracy:.1%}")
                        with col3:
                            st.metric("F1-Score Médio", f"{avg_f1:.1%}")
                        
                        # Mostrar distribuição de tipos de modelo
                        model_types = pd.DataFrame(results_summary)['Tipo Modelo'].value_counts()
                        st.write("🤖 **Distribuição de Modelos:**")
                        for model_type, count in model_types.items():
                            st.write(f"- {model_type}: {count} ligas")
                    
                    st.balloons()
                else:
                    st.error("❌ Nenhuma liga foi treinada com sucesso!")
                    
            except Exception as e:
                st.error(f"❌ Erro geral no treinamento: {str(e)}")
                st.error("Detalhes técnicos:")
                st.code(traceback.format_exc())
                
                # Tentar carregar backup se disponível
                if st.session_state.models_backup:
                    st.warning("🔄 Tentando carregar backup...")
                    if load_training_progress():
                        st.success("✅ Backup carregado com sucesso!")
            
            finally:
                st.session_state.training_in_progress = False
    
    with tab2:
        if not st.session_state.models_trained:
            st.warning("⚠️ Treine o sistema primeiro!")
            if st.session_state.models_backup:
                if st.button("🔄 Carregar Modelos do Backup"):
                    load_training_progress()
                    st.rerun()
        else:
            display_league_summary(st.session_state.league_models)
    
    with tab3:
        st.header("🎯 Previsões Inteligentes")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Treine o sistema primeiro!")
            if st.session_state.models_backup:
                if st.button("🔄 Carregar Modelos do Backup", key="pred_backup"):
                    load_training_progress()
                    st.rerun()
            st.stop()
        
        selected_date = st.date_input("📅 Data:", value=datetime.now().date())
        date_str = selected_date.strftime('%Y-%m-%d')
        
        # Adicionar opção para múltiplas datas
        multi_day = st.checkbox("🗓️ Analisar múltiplos dias", 
                              help="Busca jogos para até 3 dias a partir da data selecionada")
        
        fixtures = []
        
        with st.spinner("🔍 Analisando jogos..."):
            # Data selecionada
            fixtures.extend(get_fixtures_cached(date_str))
            
            # Dias adicionais se selecionado
            if multi_day:
                for i in range(1, 4):  # Próximos 3 dias
                    next_date = (selected_date + timedelta(days=i)).strftime('%Y-%m-%d')
                    fixtures.extend(get_fixtures_cached(next_date))
        
        if not fixtures:
            st.info("📅 Nenhum jogo encontrado para esta data")
        else:
            st.info(f"🔍 Encontrados {len(fixtures)} jogos para análise")
            
            # Fazer previsões (SEM LIMITE de apostas)
            predictions = predict_with_strategy(
                fixtures, 
                st.session_state.league_models, 
                min_confidence=min_confidence,
                no_limit=True  # Sem limite de apostas por dia
            )
            
            if not predictions:
                st.info("🤷 Nenhuma previsão acima da confiança mínima encontrada")
                st.write("**Possíveis motivos:**")
                st.write("• Confiança mínima muito alta")
                st.write("• Times não presentes nos dados de treinamento")
                st.write("• Jogos de ligas não treinadas")
            else:
                st.success(f"🎯 {len(predictions)} apostas encontradas!")
                
                # Filtros avançados
                col1, col2, col3 = st.columns(3)
                with col1:
                    show_only_over = st.checkbox("Mostrar apenas OVER 0.5", value=True)
                with col2:
                    sort_by = st.selectbox("Ordenar por:", 
                                         ["Pontuação Unificada", "Vs Liga %", "Poisson %", "Odd Justa"])
                with col3:
                    min_vs_league = st.slider("Mínimo vs Liga", min_value=-50, max_value=50, value=0,
                                          help="Filtrar apostas onde o jogo é X% acima/abaixo da média da liga")
                
                # Aplicar filtros
                filtered_predictions = predictions.copy()
                
                if show_only_over:
                    filtered_predictions = [p for p in filtered_predictions if p['prediction'] == 'OVER 0.5']
                
                # Filtro vs Liga
                filtered_predictions = [p for p in filtered_predictions if p['game_vs_league_percent'] >= min_vs_league]
                
                # Ordenação
                if sort_by == "Vs Liga %":
                    filtered_predictions.sort(key=lambda x: x['game_vs_league_percent'], reverse=True)
                elif sort_by == "Poisson %":
                    filtered_predictions.sort(key=lambda x: x['poisson_probability'], reverse=True)
                elif sort_by == "Odd Justa":
                    filtered_predictions.sort(key=lambda x: x['fair_odds'])
                else:  # Pontuação Unificada (padrão)
                    filtered_predictions.sort(key=lambda x: x['unified_score'], reverse=True)
                
                # Estatísticas
                if filtered_predictions:
                    avg_score = np.mean([p['unified_score'] for p in filtered_predictions])
                    avg_poisson = np.mean([p['poisson_probability'] for p in filtered_predictions])
                    avg_vs_league = np.mean([p['game_vs_league_percent'] for p in filtered_predictions])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Apostas", len(filtered_predictions))
                    with col2:
                        st.metric("Pontuação Média", f"{avg_score:.1f}")
                    with col3:
                        st.metric("Poisson Médio", f"{avg_poisson:.1f}%")
                    with col4:
                        st.metric("vs Liga Médio", f"{avg_vs_league:+.1f}%")
                    
                    # Opção para exportar previsões
                    export_data = []
                    for p in filtered_predictions:
                        export_data.append({
                            'Data': p['kickoff'].split('T')[0],
                            'Hora': p['kickoff'].split('T')[1][:5],
                            'Liga': p['league'],
                            'Casa': p['home_team'],
                            'Fora': p['away_team'],
                            'Previsão': p['prediction'],
                            'Pontuação': f"{p['unified_score']:.1f}",
                            'Risco': p['risk_level'],
                            'Poisson': f"{p['poisson_probability']:.1f}%",
                            'Vs Liga': f"{p['game_vs_league_percent']:+.1f}%",
                            'Odd Justa': p['fair_odds'],
                            'Dinâmica': p['matchup_analysis']['expected_flow']
                        })
                    
                    export_df = pd.DataFrame(export_data)
                    excel_data = create_excel_download(export_df, "previsoes.xlsx")
                    
                    if excel_data:
                        st.download_button(
                            label="📥 Exportar Previsões para Excel",
                            data=excel_data,
                            file_name=f"previsoes_{date_str}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    st.markdown("---")
                    
                    # Mostrar previsões
                    for pred in filtered_predictions:
                        display_smart_prediction(pred)
                else:
                    st.info("🔍 Nenhuma previsão após aplicar filtros")
    
    with tab4:
        st.header("📈 Dashboard de Performance")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Treine o sistema primeiro!")
        else:
            try:
                # Análise geral
                total_leagues = len(st.session_state.league_models)
                total_matches = sum(m['total_matches'] for m in st.session_state.league_models.values())
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Ligas", total_leagues)
                with col2:
                    st.metric("Total Jogos", f"{total_matches:,}")
                with col3:import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import joblib
import os
import traceback
from io import BytesIO
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="⚽ HT Goals AI Ultimate",
    page_icon="🎯",
    layout="wide"
)

# Inicializar session state com valores seguros
if 'league_models' not in st.session_state:
    st.session_state.league_models = {}
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'training_errors' not in st.session_state:
    st.session_state.training_errors = []
if 'models_backup' not in st.session_state:
    st.session_state.models_backup = {}
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None

# Configuração da API
try:
    API_KEY = st.secrets["API_KEY"]
except:
    API_KEY = "474f15de5b22951077ccb71b8d75b95c"

API_BASE_URL = "https://v3.football.api-sports.io"

# Diretório para salvar modelos
MODEL_DIR = "models"
try:
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
except:
    MODEL_DIR = "/tmp/models"
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

def save_training_progress(league_models, step="backup"):
    """Salva progresso do treinamento"""
    try:
        st.session_state.models_backup = league_models.copy()
        st.session_state.last_backup = datetime.now()
        # Tenta salvar em arquivo para persistência entre sessões
        try:
            joblib.dump(league_models, os.path.join(MODEL_DIR, f"league_models_{step}.joblib"))
        except:
            pass
        return True
    except:
        return False

def load_training_progress():
    """Carrega progresso salvo"""
    try:
        # Primeiro tenta carregar da session_state
        if st.session_state.models_backup:
            st.session_state.league_models = st.session_state.models_backup.copy()
            st.session_state.models_trained = True
            return True
        
        # Se não encontrar, tenta carregar do arquivo
        try:
            for filename in ["league_models_final.joblib", "league_models_backup.joblib"]:
                filepath = os.path.join(MODEL_DIR, filename)
                if os.path.exists(filepath):
                    league_models = joblib.load(filepath)
                    st.session_state.league_models = league_models
                    st.session_state.models_backup = league_models.copy()
                    st.session_state.models_trained = True
                    return True
        except:
            pass
        
        return False
    except:
        return False

def get_api_headers():
    """Retorna os headers corretos para API-SPORTS"""
    return {'x-apisports-key': API_KEY}

def test_api_connection():
    """Testa a conectividade com a API"""
    try:
        headers = get_api_headers()
        response = requests.get(f'{API_BASE_URL}/status', headers=headers, timeout=10)
        if response.status_code == 200:
            return True, "Conexão OK"
        else:
            return False, f"Status HTTP: {response.status_code}"
    except Exception as e:
        return False, f"Erro de conexão: {str(e)}"

def get_fixtures_with_retry(date_str, max_retries=3):
    """Busca jogos da API com retry automático e tratamento robusto"""
    headers = get_api_headers()
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                f'{API_BASE_URL}/fixtures',
                headers=headers,
                params={'date': date_str},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                fixtures = data.get('response', [])
                return fixtures
            elif response.status_code == 429:  # Rate limit
                wait_time = 2 ** (attempt + 1)
                st.warning(f"⏳ Rate limit - aguardando {wait_time}s...")
                time.sleep(wait_time)
            else:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    st.error(f"❌ Erro HTTP {response.status_code}")
                    return []
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                st.error(f"❌ Erro na API: {str(e)}")
                return []
    return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_fixtures_cached(date_str):
    """Busca jogos com cache"""
    try:
        return get_fixtures_with_retry(date_str)
    except:
        return []

def load_historical_data():
    """Carrega dados históricos do arquivo local"""
    # Se já temos dados carregados na session_state, usa-os
    if st.session_state.historical_data is not None:
        return st.session_state.historical_data, "✅ Dados carregados da sessão"
        
    data_files = [
        "data/historical_matches_complete.parquet",
        "data/historical_matches.parquet", 
        "data/historical_matches.csv",
        "historical_matches.parquet",
        "historical_matches.csv"
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            try:
                if file_path.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                else:
                    df = pd.read_csv(file_path)
                
                # Validar dados
                if df.empty:
                    continue
                
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df = df.dropna(subset=['date'])
                
                # Criar target se não existir
                if 'over_05' not in df.columns:
                    if 'ht_home' in df.columns and 'ht_away' in df.columns:
                        df['over_05'] = (df['ht_home'] + df['ht_away']) > 0
                    elif 'ht_home_goals' in df.columns and 'ht_away_goals' in df.columns:
                        df['over_05'] = (df['ht_home_goals'] + df['ht_away_goals']) > 0
                    else:
                        continue
                
                # Guardar na session_state
                st.session_state.historical_data = df
                
                return df, f"✅ {len(df)} jogos carregados de {file_path}"
            except Exception as e:
                st.warning(f"⚠️ Erro ao carregar {file_path}: {str(e)}")
                continue
    
    return None, "❌ Nenhum arquivo encontrado"

def get_seasonal_data_period():
    """Calcula período ideal baseado na temporada - agora inclui múltiplas temporadas"""
    current_date = datetime.now()
    current_month = current_date.month
    current_year = current_date.year
    
    # Queremos ao menos 3 temporadas de dados para análise profunda
    # A maioria das ligas começa em agosto, então usamos isso como referência
    start_date = datetime(current_year - 3, 8, 1)  # 3 anos atrás
    days_back = (current_date - start_date).days
    
    # Mínimo de 3 anos
    days_back = max(days_back, 3*365)
    
    return days_back, start_date

def collect_historical_data_smart(days=None, use_cached=True, seasonal=True, include_all_seasons=True):
    """Coleta inteligente com opção para incluir múltiplas temporadas e tratamento robusto"""
    
    if include_all_seasons:
        days, start_date = get_seasonal_data_period()
        st.info(f"📅 Modo Múltiplas Temporadas: Buscando dados desde {start_date.strftime('%d/%m/%Y')} ({days} dias)")
    elif seasonal and days is None:
        days, start_date = get_seasonal_data_period()
        st.info(f"📅 Modo Sazonal: Buscando dados desde {start_date.strftime('%d/%m/%Y')} ({days} dias)")
    elif days is None:
        days = 365
    
    # Tentar carregar cache primeiro
    if use_cached:
        df_cache, message = load_historical_data()
        if df_cache is not None and not df_cache.empty:
            try:
                if 'date' in df_cache.columns:
                    df_cache['date'] = pd.to_datetime(df_cache['date'], errors='coerce')
                    df_cache = df_cache.dropna(subset=['date'])
                    current_date = datetime.now()
                    cutoff_date = current_date - timedelta(days=days)
                    df_filtered = df_cache[df_cache['date'] >= cutoff_date].copy()
                    
                    if len(df_filtered) > 100:  # Mínimo de dados
                        st.success(f"✅ {len(df_filtered)} jogos carregados do cache")
                        return df_filtered
            except Exception as e:
                st.warning(f"⚠️ Erro ao processar cache: {str(e)}")
    
    # Buscar da API se necessário
    st.warning("⚠️ Coletando dados da API...")
    
    # Amostragem inteligente por temporadas
    sample_days = []
    
    # Últimos 30 dias - todos os dias
    for i in range(min(30, days)):
        sample_days.append(i + 1)
    
    # 30-90 dias - a cada 2 dias
    if days > 30:
        for i in range(30, min(90, days), 2):
            sample_days.append(i + 1)
    
    # 90-365 dias - a cada 3 dias
    if days > 90:
        for i in range(90, min(365, days), 3):
            sample_days.append(i + 1)
    
    # Mais de 1 ano - amostragem estratégica por temporada
    if days > 365:
        # Para cada temporada anterior, pegamos pontos estratégicos (meio da temporada, início, fim)
        for year_back in range(1, int(days/365) + 1):
            # Pontos médios da temporada (meses diferentes para garantir diversidade)
            for month in [9, 11, 2, 4]:  # Set, Nov, Fev, Abr - pontos estratégicos da temporada
                for day_offset in [5, 15, 25]:  # Início, meio e fim do mês
                    try:
                        year = current_date.year - year_back
                        sample_date = datetime(year, month, day_offset)
                        days_diff = (current_date - sample_date).days
                        if days_diff > 0 and days_diff <= days:
                            sample_days.append(days_diff)
                    except:
                        continue
    
    # Remover duplicatas e ordenar
    sample_days = sorted(list(set(sample_days)))
    
    all_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    errors_count = 0
    max_errors = 20  # Máximo de erros permitidos - aumentado para coleta mais completa
    
    for idx, day_offset in enumerate(sample_days):
        try:
            date = datetime.now() - timedelta(days=day_offset)
            date_str = date.strftime('%Y-%m-%d')
            
            status_text.text(f"🔍 Coletando dados de {date_str}...")
            
            fixtures = get_fixtures_cached(date_str)
            if fixtures:
                for match in fixtures:
                    try:
                        if match['fixture']['status']['short'] == 'FT' and match.get('score', {}).get('halftime'):
                            match_data = extract_match_features(match)
                            if match_data:
                                all_data.append(match_data)
                    except:
                        continue
            
            progress = (idx + 1) / len(sample_days)
            progress_bar.progress(progress)
            
            # Rate limiting
            if idx % 3 == 0:
                time.sleep(0.5)
                
        except Exception as e:
            errors_count += 1
            if errors_count > max_errors:
                st.error(f"❌ Muitos erros na coleta. Parando com {len(all_data)} jogos.")
                break
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if len(all_data) < 100:
        st.error(f"❌ Dados insuficientes coletados: {len(all_data)} jogos")
        return pd.DataFrame()
    
    st.success(f"✅ {len(all_data)} jogos coletados da API")
    
    # Criar DataFrame e salvar na session_state para futuros usos
    df_result = pd.DataFrame(all_data)
    st.session_state.historical_data = df_result
    
    return df_result

def extract_match_features(match):
    """Extrai features básicas do jogo com validação"""
    try:
        # Validar estrutura
        if not match.get('score', {}).get('halftime'):
            return None
        
        ht_home = match['score']['halftime']['home']
        ht_away = match['score']['halftime']['away']
        
        # Validar se são números válidos
        if ht_home is None or ht_away is None:
            return None
        
        if not isinstance(ht_home, (int, float)) or not isinstance(ht_away, (int, float)):
            return None
        
        features = {
            'date': match['fixture']['date'][:10],
            'timestamp': match['fixture']['timestamp'],
            'league_id': match['league']['id'],
            'league_name': match['league']['name'],
            'country': match['league']['country'],
            'home_team': match['teams']['home']['name'],
            'away_team': match['teams']['away']['name'],
            'home_team_id': match['teams']['home']['id'],
            'away_team_id': match['teams']['away']['id'],
            'ht_home_goals': int(ht_home),
            'ht_away_goals': int(ht_away),
            'ht_total_goals': int(ht_home) + int(ht_away),
            'over_05': 1 if (int(ht_home) + int(ht_away)) > 0 else 0,
            # Adicionar informação de temporada (baseada na data)
            'season': get_season_from_date(match['fixture']['date'][:10])
        }
        
        return features
    except Exception as e:
        return None

def get_season_from_date(date_str):
    """Extrai a temporada a partir da data"""
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        year = date.year
        month = date.month
        
        # Para meses a partir de julho, consideramos a temporada como ano/ano+1
        # Para meses até junho, consideramos a temporada como ano-1/ano
        if month >= 7:
            return f"{year}/{year+1}"
        else:
            return f"{year-1}/{year}"
    except:
        # Fallback para temporada atual
        current_year = datetime.now().year
        return f"{current_year-1}/{current_year}"

def calculate_poisson_probabilities(home_avg, away_avg, improved=True):
    """Calcula probabilidades usando distribuição de Poisson com validação e melhorias"""
    try:
        # Validar inputs
        if home_avg < 0 or away_avg < 0:
            return {'poisson_over_05': 0.5, 'expected_goals_ht': 0.5, 'home_lambda': 0.25, 'away_lambda': 0.25}
        
        # Lambda para cada time (média de gols esperados)
        home_lambda = max(home_avg / 2, 0.01)  # Mínimo 0.01
        away_lambda = max(away_avg / 2, 0.01)
        
        # MELHORIA: Ajuste dinâmico baseado em observações empíricas
        # Estudos mostram que o modelo Poisson simples tende a subestimar empates
        if improved:
            # Ajuste para correlação negativa entre gols (tendência de empate)
            correction_factor = 0.85  # Fator de correção empírico
            home_lambda = home_lambda * correction_factor
            away_lambda = away_lambda * correction_factor
        
        # Probabilidade de 0 gols para cada time
        prob_home_0 = poisson.pmf(0, home_lambda)
        prob_away_0 = poisson.pmf(0, away_lambda)
        
        # Probabilidade de 0-0 no HT
        prob_0_0 = prob_home_0 * prob_away_0
        
        # Probabilidade de Over 0.5 HT
        prob_over_05 = 1 - prob_0_0
        
        # Gols esperados no HT
        expected_goals_ht = home_lambda + away_lambda
        
        # MELHORIA: Cálculo de probabilidades específicas para apostas
        prob_exact_1 = (poisson.pmf(1, home_lambda) * prob_away_0) + (prob_home_0 * poisson.pmf(1, away_lambda))
        prob_2_plus = 1 - prob_0_0 - prob_exact_1
        
        return {
            'poisson_over_05': min(max(prob_over_05, 0), 1),
            'expected_goals_ht': max(expected_goals_ht, 0),
            'home_lambda': home_lambda,
            'away_lambda': away_lambda,
            'prob_0_0': prob_0_0,
            'prob_exact_1': prob_exact_1,
            'prob_2_plus': prob_2_plus
        }
    except:
        return {'poisson_over_05': 0.5, 'expected_goals_ht': 0.5, 'home_lambda': 0.25, 'away_lambda': 0.25}

def remove_outliers_winsorization(team_df, column, lower_percentile=5, upper_percentile=95):
    """
    Aplica winsorização para limitar valores extremos,
    substituindo outliers pelos valores dos percentis especificados.
    """
    if len(team_df) < 5:  # Precisa de dados suficientes
        return team_df
    
    try:
        # Calcular percentis
        lower_bound = np.percentile(team_df[column], lower_percentile)
        upper_bound = np.percentile(team_df[column], upper_percentile)
        
        # Criar cópia para não modificar o original
        df_clean = team_df.copy()
        
        # Aplicar winsorização
        df_clean[column] = df_clean[column].clip(lower=lower_bound, upper=upper_bound)
        
        return df_clean
    except:
        return team_df  # Em caso de erro, retorna os dados originais

def calculate_robust_stats(team_df, goals_column):
    """
    Calcula estatísticas robustas que são menos sensíveis a outliers.
    """
    if len(team_df) < 3:  # Precisa de dados suficientes
        return {
            'mean': team_df[goals_column].mean() if not team_df.empty else 0,
            'median': team_df[goals_column].median() if not team_df.empty else 0,
            'has_outliers': False,
            'outlier_count': 0,
            'winsorized_mean': team_df[goals_column].mean() if not team_df.empty else 0,
            'trimmed_mean': team_df[goals_column].mean() if not team_df.empty else 0
        }
    
    try:
        # Estatísticas básicas
        mean = team_df[goals_column].mean()
        median = team_df[goals_column].median()
        std = team_df[goals_column].std()
        
        # Identificar outliers (método Z-score)
        z_scores = np.abs((team_df[goals_column] - mean) / (std if std > 0 else 1))
        outliers = team_df[z_scores > 2.5]  # Z-score > 2.5 considerado outlier
        
        # Aplicar winsorização
        winsorized_df = remove_outliers_winsorization(team_df, goals_column)
        winsorized_mean = winsorized_df[goals_column].mean()
        
        # Média aparada (remove os 10% extremos)
        # Função trim_mean importada sob demanda para evitar dependência estrita
        try:
            from scipy import stats
            trimmed_mean = stats.trim_mean(team_df[goals_column], 0.1) if len(team_df) >= 10 else mean
        except:
            trimmed_mean = mean  # Fallback se stats não estiver disponível
        
        return {
            'mean': mean,
            'median': median,
            'has_outliers': len(outliers) > 0,
            'outlier_count': len(outliers),
            'winsorized_mean': winsorized_mean,
            'trimmed_mean': trimmed_mean
        }
    except:
        # Fallback seguro
        return {
            'mean': team_df[goals_column].mean() if not team_df.empty else 0,
            'median': team_df[goals_column].median() if not team_df.empty else 0,
            'has_outliers': False,
            'outlier_count': 0,
            'winsorized_mean': team_df[goals_column].mean() if not team_df.empty else 0,
            'trimmed_mean': team_df[goals_column].mean() if not team_df.empty else 0
        }

def calculate_advanced_features(league_df, include_recent_form=True):
    """Calcula features avançadas com tratamento robusto de erros e análise de forma recente"""
    try:
        # Validar DataFrame
        if league_df.empty:
            return pd.DataFrame(), {}, 0.5
        
        # Garantir que over_05 existe
        if 'over_05' not in league_df.columns:
            if 'ht_home_goals' in league_df.columns and 'ht_away_goals' in league_df.columns:
                league_df['over_05'] = (league_df['ht_home_goals'] + league_df['ht_away_goals']) > 0
            else:
                return pd.DataFrame(), {}, 0.5
        
        # Ordenar por data se disponível
        if 'date' in league_df.columns:
            league_df['date'] = pd.to_datetime(league_df['date'], errors='coerce')
            league_df = league_df.sort_values('date').reset_index(drop=True)
        
        # Estatísticas da liga com fallbacks
        league_over_rate = league_df['over_05'].mean() if len(league_df) > 0 else 0.5
        
        # Usar estatísticas robustas para gols da liga
        league_stats = calculate_robust_stats(league_df, 'ht_total_goals')
        league_avg_goals = league_stats['winsorized_mean'] if league_stats['has_outliers'] else league_stats['mean']
        if league_avg_goals == 0:
            league_avg_goals = 1.0  # Fallback para evitar divisão por zero
        
        # Analisar tendência da liga nas últimas temporadas
        league_trend = analyze_league_trend(league_df)
        
        # Estatísticas por time
        team_stats = {}
        try:
            unique_teams = pd.concat([league_df['home_team_id'], league_df['away_team_id']]).unique()
        except:
            return pd.DataFrame(), {}, league_over_rate
        
        for team_id in unique_teams:
            try:
                # Jogos em casa
                home_matches = league_df[league_df['home_team_id'] == team_id]
                # Jogos fora
                away_matches = league_df[league_df['away_team_id'] == team_id]
                # Todos os jogos
                all_matches = pd.concat([home_matches, away_matches])
                
                if len(all_matches) == 0:
                    continue
                
                team_name = home_matches.iloc[0]['home_team'] if len(home_matches) > 0 else away_matches.iloc[0]['away_team']
                
                # MELHORIA: Análise de forma recente (últimos 5 jogos)
                recent_form = {}
                if include_recent_form and len(all_matches) >= 3:
                    # Ordena por data se disponível
                    if 'date' in all_matches.columns:
                        recent_matches = all_matches.sort_values('date', ascending=False).head(5)
                    else:
                        recent_matches = all_matches.tail(5)  # Assume que os últimos são os mais recentes
                    
                    recent_form = {
                        'recent_over_rate': recent_matches['over_05'].mean(),
                        'recent_goals_scored': (
                            recent_matches[recent_matches['home_team_id'] == team_id]['ht_home_goals'].sum() +
                            recent_matches[recent_matches['away_team_id'] == team_id]['ht_away_goals'].sum()
                        ) / len(recent_matches),
                        'recent_goals_conceded': (
                            recent_matches[recent_matches['home_team_id'] == team_id]['ht_away_goals'].sum() +
                            recent_matches[recent_matches['away_team_id'] == team_id]['ht_home_goals'].sum()
                        ) / len(recent_matches)
                    }
                
                # Tratamento de outliers para estatísticas de gols em casa
                home_goals_stats = calculate_robust_stats(home_matches, 'ht_home_goals')
                home_goals_scored = home_goals_stats['winsorized_mean'] if home_goals_stats['has_outliers'] else home_goals_stats['mean']
                if len(home_matches) == 0 or np.isnan(home_goals_scored):
                    home_goals_scored = league_avg_goals/2
                
                # Tratamento de outliers para gols sofridos em casa
                home_conceded_stats = calculate_robust_stats(home_matches, 'ht_away_goals')
                home_goals_conceded = home_conceded_stats['winsorized_mean'] if home_conceded_stats['has_outliers'] else home_conceded_stats['mean']
                if len(home_matches) == 0 or np.isnan(home_goals_conceded):
                    home_goals_conceded = league_avg_goals/2
                
                # Tratamento de outliers para gols marcados fora
                away_goals_stats = calculate_robust_stats(away_matches, 'ht_away_goals')
                away_goals_scored = away_goals_stats['winsorized_mean'] if away_goals_stats['has_outliers'] else away_goals_stats['mean']
                if len(away_matches) == 0 or np.isnan(away_goals_scored):
                    away_goals_scored = league_avg_goals/2
                
                # Tratamento de outliers para gols sofridos fora
                away_conceded_stats = calculate_robust_stats(away_matches, 'ht_home_goals')
                away_goals_conceded = away_conceded_stats['winsorized_mean'] if away_conceded_stats['has_outliers'] else away_conceded_stats['mean']
                if len(away_matches) == 0 or np.isnan(away_goals_conceded):
                    away_goals_conceded = league_avg_goals/2
                
                # MELHORIA: Análise de tendência (melhorando ou piorando)
                team_trend = analyze_team_trend(all_matches, team_id)
                
                team_stats[team_id] = {
                    'team_name': team_name,
                    'games': len(all_matches),
                    'over_rate': all_matches['over_05'].mean(),
                    # Casa
                    'home_games': len(home_matches),
                    'home_over_rate': home_matches['over_05'].mean() if len(home_matches) > 0 else league_over_rate,
                    'home_goals_scored': home_goals_scored,
                    'home_goals_conceded': home_goals_conceded,
                    # Informações sobre outliers em casa
                    'home_has_outliers': home_goals_stats['has_outliers'],
                    'home_outlier_count': home_goals_stats['outlier_count'],
                    # Fora
                    'away_games': len(away_matches),
                    'away_over_rate': away_matches['over_05'].mean() if len(away_matches) > 0 else league_over_rate,
                    'away_goals_scored': away_goals_scored,
                    'away_goals_conceded': away_goals_conceded,
                    # Informações sobre outliers fora
                    'away_has_outliers': away_goals_stats['has_outliers'],
                    'away_outlier_count': away_goals_stats['outlier_count'],
                    # Força ofensiva/defensiva
                    'home_attack_strength': max(home_goals_scored / (league_avg_goals/2 + 0.01), 0.1),
                    'home_defense_strength': max(home_goals_conceded / (league_avg_goals/2 + 0.01), 0.1),
                    'away_attack_strength': max(away_goals_scored / (league_avg_goals/2 + 0.01), 0.1),
                    'away_defense_strength': max(away_goals_conceded / (league_avg_goals/2 + 0.01), 0.1),
                    # Tendência da equipe
                    'overall_trend': team_trend.get('overall_trend', 0),
                    'home_trend': team_trend.get('home_trend', 0),
                    'away_trend': team_trend.get('away_trend', 0),
                    # Forma recente
                    **recent_form
                }
            except Exception as e:
                continue
        
        # Criar features para ML
        features = []
        
        for idx, row in league_df.iterrows():
            try:
                home_id = row['home_team_id']
                away_id = row['away_team_id']
                
                if home_id not in team_stats or away_id not in team_stats:
                    continue
                
                home_stats = team_stats[home_id]
                away_stats = team_stats[away_id]
                
                # Poisson predictions com validação
                home_expected = home_stats['home_attack_strength'] * away_stats['away_defense_strength'] * league_avg_goals/2
                away_expected = away_stats['away_attack_strength'] * home_stats['home_defense_strength'] * league_avg_goals/2
                
                # Usar modelo Poisson melhorado
                poisson_calc = calculate_poisson_probabilities(home_expected * 2, away_expected * 2, improved=True)
                
                # MELHORIA: Análise Casa vs Fora específica
                home_away_dynamic = analyze_home_away_dynamic(home_stats, away_stats)
                
                # Calcular pontuação unificada
                unified_score = calculate_unified_score(
                    home_stats, 
                    away_stats, 
                    league_over_rate,
                    poisson_calc['poisson_over_05']
                )
                
                # Features completas
                feature_row = {
                    # Taxas básicas
                    'home_over_rate': home_stats['over_rate'],
                    'away_over_rate': away_stats['over_rate'],
                    'home_home_over_rate': home_stats['home_over_rate'],
                    'away_away_over_rate': away_stats['away_over_rate'],
                    'league_over_rate': league_over_rate,
                    
                    # Força casa/fora
                    'home_attack_strength': home_stats['home_attack_strength'],
                    'home_defense_strength': home_stats['home_defense_strength'],
                    'away_attack_strength': away_stats['away_attack_strength'],
                    'away_defense_strength': away_stats['away_defense_strength'],
                    
                    # Poisson
                    'poisson_over_05': poisson_calc['poisson_over_05'],
                    'expected_goals_ht': poisson_calc['expected_goals_ht'],
                    'prob_0_0': poisson_calc['prob_0_0'],
                    'prob_exact_1': poisson_calc['prob_exact_1'],
                    'prob_2_plus': poisson_calc['prob_2_plus'],
                    
                    # Combinações
                    'combined_over_rate': (home_stats['home_over_rate'] + away_stats['away_over_rate']) / 2,
                    'attack_index': (home_stats['home_attack_strength'] + away_stats['away_attack_strength']) / 2,
                    'game_pace_index': max(home_expected + away_expected, 0),
                    
                    # Comparação com média da liga
                    'over_rate_vs_league': ((home_stats['home_over_rate'] + away_stats['away_over_rate']) / 2) / max(league_over_rate, 0.01),
                    'expected_vs_league': poisson_calc['expected_goals_ht'] / max(league_avg_goals, 0.01),
                    
                    # Jogos disputados
                    'home_games_played': home_stats['home_games'],
                    'away_games_played': away_stats['away_games'],
                    'min_games': min(home_stats['home_games'], away_stats['away_games']),
                    
                    # MELHORIAS: Novas features
                    # Tendência das equipes
                    'home_trend': home_stats.get('home_trend', 0),
                    'away_trend': away_stats.get('away_trend', 0),
                    'combined_trend': (home_stats.get('home_trend', 0) + away_stats.get('away_trend', 0)) / 2,
                    
                    # Forma recente
                    'home_recent_over_rate': home_stats.get('recent_over_rate', home_stats['over_rate']),
                    'away_recent_over_rate': away_stats.get('recent_over_rate', away_stats['over_rate']),
                    'home_recent_goals': home_stats.get('recent_goals_scored', home_stats['home_goals_scored']),
                    'away_recent_goals': away_stats.get('recent_goals_scored', away_stats['away_goals_scored']),
                    
                    # Dinâmica Casa vs Fora
                    'home_dominance': home_away_dynamic.get('home_dominance', 1.0),
                    'away_threat': home_away_dynamic.get('away_threat', 1.0),
                    'matchup_balance': home_away_dynamic.get('matchup_balance', 0.5),
                    
                    # Tendência da liga
                    'league_trend': league_trend.get('overall_trend', 0),
                    
                    # Pontuação unificada
                    'unified_score': unified_score['unified_score'] / 100,  # Normalizado para 0-1
                    
                    # Outliers
                    'home_has_outliers': 1 if home_stats.get('home_has_outliers', False) else 0,
                    'away_has_outliers': 1 if away_stats.get('away_has_outliers', False) else 0,
                    
                    # Target
                    'target': row['over_05']
                }
                
                features.append(feature_row)
            except Exception as e:
                continue
        
        return pd.DataFrame(features), team_stats, league_over_rate
        
    except Exception as e:
        st.error(f"❌ Erro ao calcular features: {str(e)}")
        return pd.DataFrame(), {}, 0.5

def calculate_unified_score(home_stats, away_stats, league_over_rate, poisson_prob):
    """
    Calcula uma pontuação unificada combinando análise da liga, times e confronto específico.
    Retorna um score de 0-100 e uma classificação de risco.
    
    Pontuação considera:
    1. Desempenho dos times em casa/fora (40%)
    2. Probabilidade Poisson (30%)
    3. Comparação com média da liga (20%)
    4. Tendência e forma recente (10%)
    """
    try:
        # 1. Desempenho Casa/Fora (40%)
        home_strength = home_stats.get('home_over_rate', 0.5) * 100  # % de Over 0.5 HT em casa
        away_strength = away_stats.get('away_over_rate', 0.5) * 100  # % de Over 0.5 HT fora
        team_score = (home_strength + away_strength) / 2  # Média das taxas
        team_score_norm = min(100, max(0, team_score * 1.5))  # Normalizado para 0-100
        
        # 2. Probabilidade Poisson (30%)
        poisson_score = poisson_prob * 100  # Já está em porcentagem
        
        # 3. Comparação com Liga (20%)
        combined_rate = (home_stats.get('home_over_rate', 0.5) + away_stats.get('away_over_rate', 0.5)) / 2
        vs_league = (combined_rate / max(league_over_rate, 0.01) - 1) * 100  # % acima/abaixo da liga
        # Normalizar para 0-100 (onde 0 = -50% abaixo da liga, 100 = +50% acima da liga)
        vs_league_norm = min(100, max(0, (vs_league + 50) * 1))
        
        # 4. Tendência e Forma Recente (10%)
        home_trend = home_stats.get('home_trend', 0)
        away_trend = away_stats.get('away_trend', 0)
        home_recent = home_stats.get('recent_over_rate', home_stats.get('home_over_rate', 0.5))
        away_recent = away_stats.get('recent_over_rate', away_stats.get('away_over_rate', 0.5))
        
        # Calcular score de tendência (-100 a +100)
        trend_score = (home_trend + away_trend) * 50  # Amplificar tendência
        # Calcular score de forma recente (0-100)
        recent_score = (home_recent + away_recent) / 2 * 100
        
        # Combinar tendência e forma recente
        trend_recent_score = (trend_score + recent_score) / 2
        # Normalizar para 0-100
        trend_recent_norm = min(100, max(0, trend_recent_score))
        
        # Calcular pontuação final ponderada
        final_score = (
            (team_score_norm * 0.4) +  # 40% peso
            (poisson_score * 0.3) +    # 30% peso
            (vs_league_norm * 0.2) +   # 20% peso
            (trend_recent_norm * 0.1)  # 10% peso
        )
        
        # Classificar o risco
        if final_score >= 80:
            risk = "Muito Baixo"
        elif final_score >= 70:
            risk = "Baixo"
        elif final_score >= 60:
            risk = "Moderado"
        elif final_score >= 50:
            risk = "Alto"
        else:
            risk = "Muito Alto"
        
        return {
            'unified_score': round(final_score, 1),
            'risk_level': risk,
            'components': {
                'team_score': round(team_score_norm, 1),
                'poisson_score': round(poisson_score, 1),
                'vs_league_score': round(vs_league_norm, 1),
                'trend_recent_score': round(trend_recent_norm, 1)
            },
            'vs_league_raw': round(vs_league, 1)  # Manter o valor original para referência
        }
    except Exception as e:
        # Fallback seguro
        return {
            'unified_score': 50.0,
            'risk_level': "Indeterminado",
            'components': {
                'team_score': 50.0,
                'poisson_score': 50.0,
                'vs_league_score': 50.0,
                'trend_recent_score': 50.0
            },
            'vs_league_raw': 0.0
        }

def analyze_league_trend(league_df):
    """Analisa tendência da liga ao longo do tempo"""
    try:
        if 'date' not in league_df.columns or len(league_df) < 30:
            return {'overall_trend': 0}
        
        # Ordenar por data
        df = league_df.sort_values('date')
        
        # Dividir em dois períodos para comparar
        half_point = len(df) // 2
        first_half = df.iloc[:half_point]
        second_half = df.iloc[half_point:]
        
        # Calcular taxas de over em cada período
        if len(first_half) > 0 and len(second_half) > 0:
            first_rate = first_half['over_05'].mean()
            second_rate = second_half['over_05'].mean()
            
            # Calcular tendência (-1 a 1, onde positivo indica aumento na taxa)
            trend = (second_rate - first_rate) * 2  # Normalizar para escala desejada
            
            return {
                'overall_trend': trend,
                'first_half_rate': first_rate,
                'second_half_rate': second_rate
            }
        
        return {'overall_trend': 0}
    except:
        return {'overall_trend': 0}

def analyze_team_trend(team_matches, team_id):
    """Analisa tendência da equipe ao longo do tempo"""
    try:
        if 'date' not in team_matches.columns or len(team_matches) < 10:
            return {'overall_trend': 0, 'home_trend': 0, 'away_trend': 0}
        
        # Ordenar por data
        df = team_matches.sort_values('date')
        
        # Dividir em dois períodos para comparar
        half_point = len(df) // 2
        first_half = df.iloc[:half_point]
        second_half = df.iloc[half_point:]
        
        # Geral
        overall_trend = 0
        if len(first_half) > 0 and len(second_half) > 0:
            first_rate = first_half['over_05'].mean()
            second_rate = second_half['over_05'].mean()
            overall_trend = (second_rate - first_rate) * 2
        
        # Casa
        home_trend = 0
        home_first = first_half[first_half['home_team_id'] == team_id]
        home_second = second_half[second_half['home_team_id'] == team_id]
        if len(home_first) > 0 and len(home_second) > 0:
            home_first_rate = home_first['over_05'].mean()
            home_second_rate = home_second['over_05'].mean()
            home_trend = (home_second_rate - home_first_rate) * 2
        
        # Fora
        away_trend = 0
        away_first = first_half[first_half['away_team_id'] == team_id]
        away_second = second_half[second_half['away_team_id'] == team_id]
        if len(away_first) > 0 and len(away_second) > 0:
            away_first_rate = away_first['over_05'].mean()
            away_second_rate = away_second['over_05'].mean()
            away_trend = (away_second_rate - away_first_rate) * 2
        
        return {
            'overall_trend': overall_trend,
            'home_trend': home_trend,
            'away_trend': away_trend
        }
    except:
        return {'overall_trend': 0, 'home_trend': 0, 'away_trend': 0}

def analyze_home_away_dynamic(home_stats, away_stats):
    """Analisa a dinâmica específica Casa vs Fora entre as duas equipes"""
    try:
        # Calcular dominância em casa vs ameaça fora
        home_dominance = (home_stats['home_over_rate'] / max(0.1, home_stats['away_over_rate']))
        away_threat = (away_stats['away_over_rate'] / max(0.1, away_stats['home_over_rate']))
        
        # Normalizar (valores acima de 1 indicam força na condição específica)
        home_dominance = max(0.5, min(2.0, home_dominance))
        away_threat = max(0.5, min(2.0, away_threat))
        
        # Equilíbrio do confronto (0 = dominância em casa, 1 = dominância fora)
        matchup_balance = away_threat / (home_dominance + away_threat)
        
        return {
            'home_dominance': home_dominance,
            'away_threat': away_threat,
            'matchup_balance': matchup_balance
        }
    except:
        return {'home_dominance': 1.0, 'away_threat': 1.0, 'matchup_balance': 0.5}

def train_complete_model_with_validation(league_df, league_id, league_name, min_matches=30):
    """Treina modelo com validação completa e tratamento robusto"""
    
    if len(league_df) < min_matches:
        return None, f"❌ {league_name}: {len(league_df)} jogos < {min_matches} mínimo"
    
    try:
        # Preparar features avançadas (incluindo análise de forma recente)
        features_df, team_stats, league_over_rate = calculate_advanced_features(league_df, include_recent_form=True)
        
        if features_df.empty or len(features_df) < min_matches:
            return None, f"❌ {league_name}: Features insuficientes"
        
        # Verificar se temos variação no target
        if features_df['target'].nunique() < 2:
            return None, f"❌ {league_name}: Sem variação no target"
        
        feature_cols = [col for col in features_df.columns if col != 'target']
        X = features_df[feature_cols]
        y = features_df['target']
        
        # Verificar NaN
        if X.isnull().any().any() or y.isnull().any():
            X = X.fillna(X.mean())
            y = y.fillna(0)
        
        # MELHORIA: Usar Time Series Split para dados temporais
        # Isso é mais realista para dados de futebol que têm uma sequência temporal
        try:
            # Primeiramente tentar com TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=3, test_size=int(len(X) * 0.2))
            train_indices, test_indices = list(tscv.split(X))[2]  # Pegar a última divisão
            
            X_train_val, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train_val, y_test = y.iloc[train_indices], y.iloc[test_indices]
            
            # Agora dividir entre treino e validação
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
            )
        except:
            # Fallback para o método tradicional
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
            )
        
        # Escalar dados
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # MELHORIA: Modelos mais avançados com calibração de probabilidade
        # Modelos calibrados fornecem estimativas de probabilidade mais confiáveis
        models = {
            'rf': RandomForestClassifier(
                n_estimators=100, max_depth=8, random_state=42, 
                n_jobs=1, min_samples_split=5, min_samples_leaf=2
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
            ),
            'et': ExtraTreesClassifier(
                n_estimators=100, max_depth=8, random_state=42,
                n_jobs=1, min_samples_split=5, min_samples_leaf=2
            )
        }
        
        # Treinar e validar cada modelo
        best_model = None
        best_score = 0
        results = {}
        
        for name, model in models.items():
            try:
                # Treinar com calibração de probabilidade
                calibrated_model = CalibratedClassifierCV(model, cv=3, method='sigmoid')
                calibrated_model.fit(X_train_scaled, y_train)
                
                # Validar
                val_pred = calibrated_model.predict(X_val_scaled)
                val_acc = accuracy_score(y_val, val_pred)
                val_prec = precision_score(y_val, val_pred, zero_division=0)
                val_rec = recall_score(y_val, val_pred, zero_division=0)
                val_f1 = f1_score(y_val, val_pred, zero_division=0)
                
                results[name] = {
                    'val_accuracy': val_acc,
                    'val_precision': val_prec,
                    'val_recall': val_rec,
                    'val_f1': val_f1,
                    'model': calibrated_model
                }
                
                if val_f1 > best_score:
                    best_score = val_f1
                    best_model = calibrated_model
                    
            except Exception as e:
                st.warning(f"⚠️ Erro treinando {name}: {str(e)}")
                continue
        
        if best_model is None:
            return None, f"❌ {league_name}: Nenhum modelo funcionou"
        
        # MELHORIA: Ensemble dos modelos para decisões mais robustas
        try:
            # Criar ensemble dos melhores modelos
            working_models = [(name, model_data['model']) for name, model_data in results.items() 
                            if model_data['val_f1'] > 0.6]
            
            if len(working_models) >= 2:
                ensemble = VotingClassifier(working_models, voting='soft')
                ensemble.fit(X_train_scaled, y_train)
                
                # Verificar se o ensemble é melhor
                ensemble_pred = ensemble.predict(X_val_scaled)
                ensemble_f1 = f1_score(y_val, ensemble_pred, zero_division=0)
                
                if ensemble_f1 > best_score:
                    best_model = ensemble
                    best_score = ensemble_f1
        except:
            pass  # Se falhar, apenas usa o melhor modelo individual
        
        # Testar melhor modelo
        test_pred = best_model.predict(X_test_scaled)
        test_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, test_pred),
            'precision': precision_score(y_test, test_pred, zero_division=0),
            'recall': recall_score(y_test, test_pred, zero_division=0),
            'f1_score': f1_score(y_test, test_pred, zero_division=0)
        }
        
        # MELHORIA: Análise de threshold otimizado usando grid search
        best_threshold = 0.5
        best_f1 = test_metrics['f1_score']
        
        for threshold in np.arange(0.3, 0.8, 0.02):  # Grid mais fino
            try:
                pred_threshold = (test_pred_proba >= threshold).astype(int)
                f1_threshold = f1_score(y_test, pred_threshold, zero_division=0)
                precision_threshold = precision_score(y_test, pred_threshold, zero_division=0)
                
                # Balancear F1 e precisão para maximizar taxa de acerto
                combined_score = (f1_threshold * 0.7) + (precision_threshold * 0.3)
                
                if combined_score > best_f1:
                    best_f1 = combined_score
                    best_threshold = threshold
            except:
                continue
        
        # Retreinar no dataset completo
        X_scaled = scaler.fit_transform(X)
        best_model.fit(X_scaled, y)
        
        # Feature importance
        try:
            # Extrair importância das features
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = dict(zip(feature_cols, best_model.feature_importances_))
            elif hasattr(best_model, 'estimators_') and hasattr(best_model.estimators_[0], 'feature_importances_'):
                # Para VotingClassifier, pegamos a média das importâncias
                importances = np.zeros(len(feature_cols))
                for estimator in best_model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        importances += estimator.feature_importances_
                feature_importance = dict(zip(feature_cols, importances / len(best_model.estimators_)))
            else:
                # Fallback
                feature_importance = {feature: 1.0/len(feature_cols) for feature in feature_cols}
            
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        except:
            top_features = [('unknown', 1.0)]
        
        # MELHORIA: Análise de desempenho em diferentes cenários
        scenario_analysis = analyze_performance_scenarios(X_test_scaled, y_test, best_model, feature_cols)
        
        # Preparar dados do modelo
        model_data = {
            'model': best_model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'team_stats': team_stats,
            'league_id': league_id,
            'league_name': league_name,
            'league_over_rate': league_over_rate,
            'total_matches': len(league_df),
            'validation_results': results,
            'test_metrics': test_metrics,
            'best_threshold': best_threshold,
            'top_features': top_features,
            'scenario_analysis': scenario_analysis,
            'model_type': type(best_model).__name__
        }
        
        # Adicionar análise de tendência da liga
        league_trend = analyze_league_trend(league_df)
        model_data['league_trend'] = league_trend
        
        return model_data, f"✅ {league_name}: Acc {test_metrics['accuracy']:.1%} | F1 {test_metrics['f1_score']:.1%}"
        
    except Exception as e:
        error_msg = f"❌ {league_name}: {str(e)}"
        st.session_state.training_errors.append(error_msg)
        return None, error_msg

def analyze_performance_scenarios(X_test_scaled, y_test, model, feature_cols):
    """Analisa o desempenho do modelo em diferentes cenários"""
    try:
        # Converter para DataFrame para análise
        X_test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
        X_test_df['target'] = y_test.values
        X_test_df['prediction'] = model.predict(X_test_scaled)
        X_test_df['probability'] = model.predict_proba(X_test_scaled)[:, 1]
        
        # Categorizar em diferentes cenários
        scenarios = {}
        
        # Alta confiança (>75%)
        high_conf = X_test_df[X_test_df['probability'] > 0.75]
        if len(high_conf) > 0:
            scenarios['high_confidence'] = {
                'count': len(high_conf),
                'accuracy': (high_conf['prediction'] == high_conf['target']).mean(),
                'avg_probability': high_conf['probability'].mean()
            }
        
        # Acima da média da liga
        if 'over_rate_vs_league' in X_test_df.columns:
            above_avg = X_test_df[X_test_df['over_rate_vs_league'] > 1.1]
            if len(above_avg) > 0:
                scenarios['above_league_avg'] = {
                    'count': len(above_avg),
                    'accuracy': (above_avg['prediction'] == above_avg['target']).mean(),
                    'avg_probability': above_avg['probability'].mean()
                }
        
        # Alta força de ataque
        if 'attack_index' in X_test_df.columns:
            high_attack = X_test_df[X_test_df['attack_index'] > 1.2]
            if len(high_attack) > 0:
                scenarios['high_attack'] = {
                    'count': len(high_attack),
                    'accuracy': (high_attack['prediction'] == high_attack['target']).mean(),
                    'avg_probability': high_attack['probability'].mean()
                }
        
        # Pontuação unificada alta
        if 'unified_score' in X_test_df.columns:
            high_unified = X_test_df[X_test_df['unified_score'] > 0.7]
            if len(high_unified) > 0:
                scenarios['high_unified_score'] = {
                    'count': len(high_unified),
                    'accuracy': (high_unified['prediction'] == high_unified['target']).mean(),
                    'avg_probability': high_unified['probability'].mean()
                }
        
        return scenarios
    except:
        return {}

def predict_with_strategy(fixtures, league_models, min_confidence=60, no_limit=True):
    """Faz previsões com estratégia inteligente
