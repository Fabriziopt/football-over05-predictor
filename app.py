import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import time

# Configuração da página
st.set_page_config(
    page_title="⚽ Over 0.5 HT Predictor",
    page_icon="⚽",
    layout="wide"
)

# API Key automática (sua API paga)
API_KEY = "1136cc77028d84lfd0efa2a603f81638"

# CSS melhorado
st.markdown("""
<style>
    .header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .premium-badge {
        background: linear-gradient(45deg, #ffd700, #ffed4e);
        color: #333;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .team-over {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .team-under {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .team-balanced {
        background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .league-header {
        background: linear-gradient(45deg, #17a2b8, #007bff);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def get_matches_api(date, api_key=API_KEY):
    """Busca jogos com API paga - múltiplos formatos"""
    base_urls = [
        "https://api.football-data.org/v4/matches",
        "https://api.football-data.org/v2/matches"
    ]
    
    headers = {'X-Auth-Token': api_key}
    date_str = date.strftime('%Y-%m-%d')
    
    # Diferentes formatos de parâmetros para testar
    params_list = [
        {'dateFrom': date_str, 'dateTo': date_str},
        {'date': date_str},
        {'matchday': date_str}
    ]
    
    for base_url in base_urls:
        for params in params_list:
            try:
                response = requests.get(base_url, headers=headers, params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    matches = data.get('matches', [])
                    return matches, f"✅ {len(matches)} jogos encontrados (API v{base_url[-1]})"
                
                elif response.status_code == 400:
                    continue  # Tenta próximo formato
                    
                elif response.status_code == 403:
                    return [], "❌ API Key inválida ou sem permissões"
                    
                elif response.status_code == 429:
                    time.sleep(2)  # Rate limit
                    continue
                    
            except Exception as e:
                continue
    
    return [], "❌ Erro ao conectar com a API"

def get_historical_data_premium(days=30):
    """Busca dados históricos com API paga (até 90 dias)"""
    all_matches = []
    end_date = datetime.now()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(days):
        date = end_date - timedelta(days=i+1)
        status_text.text(f"📊 Coletando dados: {date.strftime('%d/%m/%Y')} ({i+1}/{days})")
        
        matches, message = get_matches_api(date)
        if matches:
            all_matches.extend(matches)
        
        progress_bar.progress((i+1)/days)
        time.sleep(0.1)  # Respeitar rate limit mesmo sendo paga
    
    progress_bar.empty()
    status_text.empty()
    
    return all_matches

def analyze_teams_in_leagues(matches):
    """Analisa equipes individuais dentro de cada liga"""
    team_data = {}
    league_teams = {}
    
    for match in matches:
        if (match.get('status') == 'FINISHED' and 
            match.get('score', {}).get('halfTime')):
            
            league = match['competition']['name']
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            
            ht_score = match['score']['halfTime']
            if ht_score['home'] is not None and ht_score['away'] is not None:
                total_ht_goals = ht_score['home'] + ht_score['away']
                over_05_ht = 1 if total_ht_goals > 0.5 else 0
                
                # Organizar por liga
                if league not in league_teams:
                    league_teams[league] = set()
                
                league_teams[league].add(home_team)
                league_teams[league].add(away_team)
                
                # Analisar equipe da casa
                if home_team not in team_data:
                    team_data[home_team] = {
                        'league': league,
                        'home_games': 0, 'away_games': 0,
                        'home_over': 0, 'away_over': 0,
                        'total_ht_goals_home': 0, 'total_ht_goals_away': 0
                    }
                
                team_data[home_team]['home_games'] += 1
                team_data[home_team]['home_over'] += over_05_ht
                team_data[home_team]['total_ht_goals_home'] += ht_score['home']
                
                # Analisar equipe visitante
                if away_team not in team_data:
                    team_data[away_team] = {
                        'league': league,
                        'home_games': 0, 'away_games': 0,
                        'home_over': 0, 'away_over': 0,
                        'total_ht_goals_home': 0, 'total_ht_goals_away': 0
                    }
                
                team_data[away_team]['away_games'] += 1
                team_data[away_team]['away_over'] += over_05_ht
                team_data[away_team]['total_ht_goals_away'] += ht_score['away']
    
    # Calcular estatísticas das equipes
    team_stats = {}
    
    for team, data in team_data.items():
        total_games = data['home_games'] + data['away_games']
        
        if total_games >= 3:  # Mínimo 3 jogos
            total_over = data['home_over'] + data['away_over']
            over_rate = total_over / total_games
            
            # Média de gols marcados no HT
            avg_goals_home = data['total_ht_goals_home'] / max(data['home_games'], 1)
            avg_goals_away = data['total_ht_goals_away'] / max(data['away_games'], 1)
            avg_goals_total = (data['total_ht_goals_home'] + data['total_ht_goals_away']) / total_games
            
            # Classificação da equipe
            if over_rate >= 0.70:
                classification = "🔥 EQUIPE OVER"
                bet_class = "team-over"
            elif over_rate >= 0.55:
                classification = "📈 OVER Moderada"
                bet_class = "team-over"
            elif over_rate <= 0.30:
                classification = "❄️ EQUIPE UNDER"
                bet_class = "team-under"
            elif over_rate <= 0.45:
                classification = "📉 UNDER Moderada"
                bet_class = "team-under"
            else:
                classification = "⚖️ Equilibrada"
                bet_class = "team-balanced"
            
            team_stats[team] = {
                'league': data['league'],
                'over_rate': over_rate,
                'total_games': total_games,
                'home_games': data['home_games'],
                'away_games': data['away_games'],
                'avg_goals_home': avg_goals_home,
                'avg_goals_away': avg_goals_away,
                'avg_goals_total': avg_goals_total,
                'classification': classification,
                'bet_class': bet_class
            }
    
    return team_stats, league_teams

def predict_matches_by_teams(today_matches, team_stats):
    """Previsões baseadas nas equipes individuais"""
    predictions = []
    
    for match in today_matches:
        if match.get('status') in ['SCHEDULED', 'TIMED']:
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            league = match['competition']['name']
            
            home_stats = team_stats.get(home_team, {})
            away_stats = team_stats.get(away_team, {})
            
            # Calcular probabilidade baseada nas equipes
            home_rate = home_stats.get('over_rate', 0.5)
            away_rate = away_stats.get('over_rate', 0.5)
            
            # Média ponderada (casa tem mais peso)
            combined_rate = (home_rate * 0.6) + (away_rate * 0.4)
            
            # Determinar previsão
            if combined_rate >= 0.65:
                prediction = "✅ OVER 0.5"
                confidence = "ALTA" if combined_rate >= 0.75 else "MÉDIA"
                bet_class = "team-over"
            elif combined_rate <= 0.35:
                prediction = "❌ UNDER 0.5"
                confidence = "ALTA" if combined_rate <= 0.25 else "MÉDIA"
                bet_class = "team-under"
            else:
                prediction = "⚖️ INDEFINIDO"
                confidence = "BAIXA"
                bet_class = "team-balanced"
            
            # Detalhes das equipes
            home_class = home_stats.get('classification', 'Sem dados')
            away_class = away_stats.get('classification', 'Sem dados')
            home_games = home_stats.get('total_games', 0)
            away_games = away_stats.get('total_games', 0)
            
            predictions.append({
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'prediction': prediction,
                'confidence': confidence,
                'probability': f"{combined_rate:.1%}",
                'home_rate': f"{home_rate:.1%}",
                'away_rate': f"{away_rate:.1%}",
                'home_class': home_class,
                'away_class': away_class,
                'home_games': home_games,
                'away_games': away_games,
                'bet_class': bet_class,
                'sort_priority': combined_rate if combined_rate >= 0.55 else 0
            })
    
    # Ordenar por prioridade
    predictions.sort(key=lambda x: x['sort_priority'], reverse=True)
    return predictions

def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>⚽ Over 0.5 HT - Análise por EQUIPES</h1>
        <p>Sistema que analisa equipes individuais dentro de cada liga</p>
        <span class="premium-badge">API PAGA ✨</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar simplificada
    st.sidebar.title("🔧 Configurações")
    st.sidebar.success("🔑 API Key: Configurada automaticamente")
    
    # Data
    selected_date = st.sidebar.date_input(
        "📅 Data para análise:",
        value=datetime.now().date()
    )
    
    # Histórico (expandido para API paga)
    days_history = st.sidebar.slider(
        "📊 Dias de histórico:",
        min_value=7,
        max_value=90,
        value=30,
        help="API paga permite até 90 dias"
    )
    
    # Tabs principais
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 PREVISÕES POR EQUIPES", 
        "🏆 RANKING DE EQUIPES", 
        "📊 ANÁLISE POR LIGA",
        "ℹ️ SOBRE"
    ])
    
    with tab1:
        st.header(f"🎯 Análise de Equipes - {selected_date.strftime('%d/%m/%Y')}")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            analyze_button = st.button(
                "🚀 ANALISAR EQUIPES DO DIA",
                type="primary",
                use_container_width=True
            )
        with col2:
            st.metric("Histórico", f"{days_history} dias")
        
        if analyze_button:
            with st.spinner("🔍 Buscando jogos de hoje..."):
                today_matches, message = get_matches_api(selected_date)
                st.info(message)
            
            if not today_matches:
                st.warning("❌ Nenhum jogo encontrado para esta data")
                return
            
            upcoming_matches = [m for m in today_matches if m.get('status') in ['SCHEDULED', 'TIMED']]
            
            if not upcoming_matches:
                st.info("⏰ Nenhum jogo programado para hoje")
                return
            
            st.success(f"✅ {len(upcoming_matches)} jogos programados!")
            
            with st.spinner(f"📈 Analisando {days_history} dias de histórico..."):
                historical_matches = get_historical_data_premium(days_history)
            
            if not historical_matches:
                st.error("❌ Erro ao buscar dados históricos")
                return
            
            # Analisar equipes
            team_stats, league_teams = analyze_teams_in_leagues(historical_matches)
            
            if not team_stats:
                st.warning("⚠️ Dados insuficientes para análise")
                return
            
            # Gerar previsões por equipes
            predictions = predict_matches_by_teams(upcoming_matches, team_stats)
            
            # Mostrar melhores apostas
            st.header("🏆 MELHORES APOSTAS - ANÁLISE POR EQUIPES")
            
            best_bets = [p for p in predictions if p['sort_priority'] >= 0.55]
            
            if best_bets:
                for bet in best_bets:
                    st.markdown(f"""
                    <div class="{bet['bet_class']}">
                        <h4>⚽ {bet['home_team']} vs {bet['away_team']}</h4>
                        <p><strong>Liga:</strong> {bet['league']}</p>
                        <p><strong>Previsão:</strong> {bet['prediction']} ({bet['probability']})</p>
                        <p><strong>Confiança:</strong> {bet['confidence']}</p>
                        <hr style="border-color: rgba(255,255,255,0.3);">
                        <p><strong>🏠 {bet['home_team']}:</strong> {bet['home_class']} ({bet['home_rate']} - {bet['home_games']} jogos)</p>
                        <p><strong>✈️ {bet['away_team']}:</strong> {bet['away_class']} ({bet['away_rate']} - {bet['away_games']} jogos)</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("🤔 Nenhuma aposta de alta confiança encontrada hoje")
            
            # Todas as previsões
            with st.expander("📋 TODAS AS PREVISÕES"):
                for pred in predictions:
                    st.write(f"**{pred['home_team']} vs {pred['away_team']}**")
                    st.write(f"Liga: {pred['league']} | Previsão: {pred['prediction']} | Probabilidade: {pred['probability']}")
                    st.write(f"Casa: {pred['home_class']} | Fora: {pred['away_class']}")
                    st.write("---")
    
    with tab2:
        st.header("🏆 Ranking das Equipes")
        
        if st.button("📊 Gerar Ranking de Equipes"):
            with st.spinner("Analisando equipes..."):
                historical_matches = get_historical_data_premium(days_history)
                team_stats, league_teams = analyze_teams_in_leagues(historical_matches)
            
            if team_stats:
                # Separar equipes por categoria
                over_teams = []
                under_teams = []
                balanced_teams = []
                
                for team, stats in team_stats.items():
                    team_info = {
                        'Equipe': team,
                        'Liga': stats['league'],
                        'Taxa Over 0.5': f"{stats['over_rate']:.1%}",
                        'Jogos': stats['total_games'],
                        'Casa': stats['home_games'],
                        'Fora': stats['away_games'],
                        'Classificação': stats['classification']
                    }
                    
                    if stats['over_rate'] >= 0.55:
                        over_teams.append(team_info)
                    elif stats['over_rate'] <= 0.45:
                        under_teams.append(team_info)
                    else:
                        balanced_teams.append(team_info)
                
                # Mostrar rankings
                if over_teams:
                    st.subheader("🔥 EQUIPES OVER (Aposte Over 0.5)")
                    over_df = pd.DataFrame(over_teams)
                    over_df = over_df.sort_values('Taxa Over 0.5', ascending=False)
                    st.dataframe(over_df, use_container_width=True)
                
                if under_teams:
                    st.subheader("❄️ EQUIPES UNDER (Evite Over 0.5)")
                    under_df = pd.DataFrame(under_teams)
                    under_df = under_df.sort_values('Taxa Over 0.5', ascending=True)
                    st.dataframe(under_df, use_container_width=True)
                
                if balanced_teams:
                    st.subheader("⚖️ EQUIPES EQUILIBRADAS")
                    balanced_df = pd.DataFrame(balanced_teams)
                    st.dataframe(balanced_df, use_container_width=True)
    
    with tab3:
        st.header("📊 Análise por Liga")
        
        if st.button("🔍 Analisar Ligas e suas Equipes"):
            with st.spinner("Organizando dados por liga..."):
                historical_matches = get_historical_data_premium(days_history)
                team_stats, league_teams = analyze_teams_in_leagues(historical_matches)
            
            if league_teams:
                for league, teams in league_teams.items():
                    if len(teams) >= 3:  # Mostrar apenas ligas com dados suficientes
                        
                        st.markdown(f"""
                        <div class="league-header">
                            🏆 {league}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        league_team_stats = [(team, team_stats[team]) for team in teams if team in team_stats]
                        
                        if league_team_stats:
                            # Ordenar por taxa Over
                            league_team_stats.sort(key=lambda x: x[1]['over_rate'], reverse=True)
                            
                            cols = st.columns(min(3, len(league_team_stats)))
                            
                            for i, (team, stats) in enumerate(league_team_stats[:6]):  # Top 6 por liga
                                with cols[i % 3]:
                                    st.markdown(f"""
                                    <div class="{stats['bet_class']}" style="margin: 0.5rem 0; padding: 1rem;">
                                        <h5>{team}</h5>
                                        <p><strong>{stats['classification']}</strong></p>
                                        <p>Taxa: {stats['over_rate']:.1%}</p>
                                        <p>Jogos: {stats['total_games']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
    
    with tab4:
        st.header("ℹ️ Sistema de Análise por Equipes")
        
        st.write("""
        ### 🎯 **Novidade: Análise Individual de Equipes!**
        
        Agora o sistema analisa **cada equipe separadamente** dentro de suas ligas:
        
        ### 🔍 **Como funciona:**
        - 🏠 **Análise Casa/Fora**: Separa performance em casa e fora
        - 📊 **Estatísticas Individuais**: Cada equipe tem seu próprio perfil
        - 🎯 **Previsões Precisas**: Combina dados das duas equipes do jogo
        - 📈 **Classificação**: 🔥 EQUIPE OVER, ❄️ EQUIPE UNDER, ⚖️ Equilibrada
        
        ### 🏆 **Vantagens da API Paga:**
        - ✅ **Até 90 dias** de histórico
        - ✅ **Dados em tempo real**
        - ✅ **Sem limitações** de requests
        - ✅ **API Key automática**
        
        ### 📊 **Exemplo prático:**
        **Liga UNDER mas com equipes OVER:**
        - Liga Portuguesa: 45% Over geral
        - Mas o Sporting: 75% Over (EQUIPE OVER!)
        - E o Porto: 70% Over (EQUIPE OVER!)
        
        ### 🚀 **Resultado:**
        Quando Sporting vs Porto jogam, mesmo numa "liga UNDER", 
        o sistema recomenda **Over 0.5** porque ambas são **EQUIPES OVER**!
        
        ### 🎯 **Use diariamente** para os melhores insights!
        """)

if __name__ == "__main__":
    main()
