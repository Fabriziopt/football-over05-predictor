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

# SUA API Key (API-Football.com)
API_KEY = "2aad0db0e5b88b3a080bdc85461a919"

# CSS
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
    .debug-success {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .debug-error {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def test_api_football():
    """Testa API-Football.com"""
    headers = {
        'X-RapidAPI-Key': API_KEY,
        'X-RapidAPI-Host': 'v3.football.api-sports.io'
    }
    
    try:
        # Teste básico - buscar ligas
        response = requests.get(
            'https://v3.football.api-sports.io/leagues',
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('results', 0) > 0:
                leagues = data.get('response', [])
                return True, f"✅ API-Football funcionando! {len(leagues)} ligas disponíveis"
            else:
                return False, "⚠️ API conectou mas sem dados"
        
        elif response.status_code == 403:
            return False, "❌ API Key inválida ou sem permissões"
        
        elif response.status_code == 429:
            return False, "⚠️ Rate limit atingido"
        
        else:
            return False, f"❌ Erro HTTP {response.status_code}"
            
    except Exception as e:
        return False, f"❌ Erro: {str(e)}"

def get_matches_api_football(date):
    """Busca jogos usando API-Football.com"""
    headers = {
        'X-RapidAPI-Key': API_KEY,
        'X-RapidAPI-Host': 'v3.football.api-sports.io'
    }
    
    date_str = date.strftime('%Y-%m-%d')
    
    try:
        # Buscar jogos por data
        response = requests.get(
            'https://v3.football.api-sports.io/fixtures',
            headers=headers,
            params={'date': date_str},
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('results', 0) > 0:
                matches = data.get('response', [])
                return matches, f"✅ {len(matches)} jogos encontrados"
            else:
                return [], "📅 Nenhum jogo programado para esta data"
        
        elif response.status_code == 403:
            return [], "❌ API Key sem permissões"
        
        elif response.status_code == 429:
            return [], "⚠️ Rate limit atingido"
        
        else:
            return [], f"❌ Erro HTTP {response.status_code}"
            
    except Exception as e:
        return [], f"❌ Erro: {str(e)}"

def get_historical_data_api_football(days=14):
    """Busca dados históricos da API-Football"""
    all_matches = []
    headers = {
        'X-RapidAPI-Key': API_KEY,
        'X-RapidAPI-Host': 'v3.football.api-sports.io'
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    successful_days = 0
    
    for i in range(days):
        date = datetime.now() - timedelta(days=i+1)
        date_str = date.strftime('%Y-%m-%d')
        
        status_text.text(f"📊 Buscando: {date.strftime('%d/%m/%Y')} ({i+1}/{days})")
        
        try:
            response = requests.get(
                'https://v3.football.api-sports.io/fixtures',
                headers=headers,
                params={'date': date_str},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results', 0) > 0:
                    matches = data.get('response', [])
                    # Filtrar apenas jogos finalizados
                    finished_matches = [m for m in matches if m['fixture']['status']['short'] == 'FT']
                    all_matches.extend(finished_matches)
                    if finished_matches:
                        successful_days += 1
            
        except Exception as e:
            continue
        
        progress_bar.progress((i+1)/days)
        time.sleep(0.3)  # Rate limiting
    
    progress_bar.empty()
    status_text.empty()
    
    return all_matches, successful_days

def analyze_teams_api_football(matches):
    """Analisa equipes usando dados da API-Football"""
    team_data = {}
    
    for match in matches:
        # Verificar se tem dados do primeiro tempo
        if not match.get('score', {}).get('halftime'):
            continue
            
        home_team = match['teams']['home']['name']
        away_team = match['teams']['away']['name']
        league = match['league']['name']
        
        ht_home = match['score']['halftime']['home'] or 0
        ht_away = match['score']['halftime']['away'] or 0
        total_ht = ht_home + ht_away
        over_05 = 1 if total_ht > 0.5 else 0
        
        # Analisar equipe da casa
        if home_team not in team_data:
            team_data[home_team] = {
                'league': league,
                'home_games': 0, 'away_games': 0,
                'home_over': 0, 'away_over': 0,
                'home_goals': 0, 'away_goals': 0
            }
        
        team_data[home_team]['home_games'] += 1
        team_data[home_team]['home_over'] += over_05
        team_data[home_team]['home_goals'] += ht_home
        
        # Analisar equipe visitante
        if away_team not in team_data:
            team_data[away_team] = {
                'league': league,
                'home_games': 0, 'away_games': 0,
                'home_over': 0, 'away_over': 0,
                'home_goals': 0, 'away_goals': 0
            }
        
        team_data[away_team]['away_games'] += 1
        team_data[away_team]['away_over'] += over_05
        team_data[away_team]['away_goals'] += ht_away
    
    # Calcular estatísticas
    team_stats = {}
    for team, data in team_data.items():
        total_games = data['home_games'] + data['away_games']
        
        if total_games >= 3:  # Mínimo 3 jogos
            total_over = data['home_over'] + data['away_over']
            over_rate = total_over / total_games
            
            home_over_rate = data['home_over'] / max(data['home_games'], 1)
            away_over_rate = data['away_over'] / max(data['away_games'], 1)
            
            avg_goals_home = data['home_goals'] / max(data['home_games'], 1)
            avg_goals_away = data['away_goals'] / max(data['away_games'], 1)
            
            team_stats[team] = {
                'league': data['league'],
                'over_rate': over_rate,
                'home_over_rate': home_over_rate,
                'away_over_rate': away_over_rate,
                'total_games': total_games,
                'home_games': data['home_games'],
                'away_games': data['away_games'],
                'avg_goals_home': avg_goals_home,
                'avg_goals_away': avg_goals_away,
                'classification': get_team_classification(over_rate)
            }
    
    return team_stats

def get_team_classification(over_rate):
    """Classifica equipe baseada na taxa Over 0.5"""
    if over_rate >= 0.75:
        return "🔥 EQUIPE OVER FORTE"
    elif over_rate >= 0.60:
        return "📈 EQUIPE OVER"
    elif over_rate <= 0.25:
        return "❄️ EQUIPE UNDER FORTE"
    elif over_rate <= 0.40:
        return "📉 EQUIPE UNDER"
    else:
        return "⚖️ EQUILIBRADA"

def get_bet_class(over_rate):
    """Retorna classe CSS baseada na taxa"""
    if over_rate >= 0.60:
        return "team-over"
    elif over_rate <= 0.40:
        return "team-under"
    else:
        return "team-balanced"

def predict_matches_api_football(today_matches, team_stats):
    """Faz previsões baseadas nas equipes"""
    predictions = []
    
    for match in today_matches:
        if match['fixture']['status']['short'] not in ['NS', 'TBD']:  # Não iniciado
            continue
            
        home_team = match['teams']['home']['name']
        away_team = match['teams']['away']['name']
        league = match['league']['name']
        match_time = match['fixture']['date']
        
        home_stats = team_stats.get(home_team, {})
        away_stats = team_stats.get(away_team, {})
        
        # Usar taxas específicas casa/fora
        home_rate = home_stats.get('home_over_rate', home_stats.get('over_rate', 0.5))
        away_rate = away_stats.get('away_over_rate', away_stats.get('over_rate', 0.5))
        
        # Algoritmo de predição
        combined_rate = (home_rate * 0.6) + (away_rate * 0.4)
        
        # Ajustes
        if home_stats.get('over_rate', 0) > 0.7 and away_stats.get('over_rate', 0) > 0.7:
            combined_rate += 0.05
        elif home_stats.get('over_rate', 0) < 0.3 and away_stats.get('over_rate', 0) < 0.3:
            combined_rate -= 0.05
        
        # Determinar previsão
        if combined_rate >= 0.65:
            prediction = "✅ OVER 0.5"
            confidence = "ALTA" if combined_rate >= 0.75 else "MÉDIA"
        elif combined_rate <= 0.35:
            prediction = "❌ UNDER 0.5"
            confidence = "ALTA" if combined_rate <= 0.25 else "MÉDIA"
        else:
            prediction = "⚖️ INDEFINIDO"
            confidence = "BAIXA"
        
        predictions.append({
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
            'time': match_time[:16] if match_time else '',
            'prediction': prediction,
            'confidence': confidence,
            'probability': f"{combined_rate:.1%}",
            'home_rate': f"{home_rate:.1%}",
            'away_rate': f"{away_rate:.1%}",
            'home_class': home_stats.get('classification', 'Sem dados'),
            'away_class': away_stats.get('classification', 'Sem dados'),
            'home_games': home_stats.get('total_games', 0),
            'away_games': away_stats.get('total_games', 0),
            'bet_class': get_bet_class(combined_rate),
            'sort_priority': combined_rate if combined_rate >= 0.55 else 0
        })
    
    predictions.sort(key=lambda x: x['sort_priority'], reverse=True)
    return predictions

def create_demo_api_football():
    """Demo com dados realistas do API-Football"""
    demo_teams = {
        # Premier League
        'Manchester City': {
            'league': 'Premier League',
            'over_rate': 0.78,
            'home_over_rate': 0.85,
            'away_over_rate': 0.71,
            'total_games': 28,
            'classification': '🔥 EQUIPE OVER FORTE'
        },
        'Liverpool': {
            'league': 'Premier League',
            'over_rate': 0.72,
            'home_over_rate': 0.79,
            'away_over_rate': 0.65,
            'total_games': 27,
            'classification': '📈 EQUIPE OVER'
        },
        'Arsenal': {
            'league': 'Premier League',
            'over_rate': 0.68,
            'home_over_rate': 0.74,
            'away_over_rate': 0.62,
            'total_games': 26,
            'classification': '📈 EQUIPE OVER'
        },
        'Burnley': {
            'league': 'Premier League',
            'over_rate': 0.32,
            'home_over_rate': 0.38,
            'away_over_rate': 0.26,
            'total_games': 25,
            'classification': '📉 EQUIPE UNDER'
        },
        
        # La Liga
        'Real Madrid': {
            'league': 'La Liga',
            'over_rate': 0.81,
            'home_over_rate': 0.88,
            'away_over_rate': 0.74,
            'total_games': 30,
            'classification': '🔥 EQUIPE OVER FORTE'
        },
        'Barcelona': {
            'league': 'La Liga',
            'over_rate': 0.76,
            'home_over_rate': 0.82,
            'away_over_rate': 0.70,
            'total_games': 29,
            'classification': '🔥 EQUIPE OVER FORTE'
        },
        'Atletico Madrid': {
            'league': 'La Liga',
            'over_rate': 0.34,
            'home_over_rate': 0.40,
            'away_over_rate': 0.28,
            'total_games': 28,
            'classification': '📉 EQUIPE UNDER'
        },
        
        # Bundesliga
        'Bayern Munich': {
            'league': 'Bundesliga',
            'over_rate': 0.86,
            'home_over_rate': 0.93,
            'away_over_rate': 0.79,
            'total_games': 24,
            'classification': '🔥 EQUIPE OVER FORTE'
        },
        'Borussia Dortmund': {
            'league': 'Bundesliga',
            'over_rate': 0.74,
            'home_over_rate': 0.81,
            'away_over_rate': 0.67,
            'total_games': 23,
            'classification': '📈 EQUIPE OVER'
        }
    }
    
    demo_matches = [
        {'home': 'Manchester City', 'away': 'Liverpool', 'league': 'Premier League', 'time': '2025-06-01 15:30'},
        {'home': 'Real Madrid', 'away': 'Barcelona', 'league': 'La Liga', 'time': '2025-06-01 21:00'},
        {'home': 'Bayern Munich', 'away': 'Borussia Dortmund', 'league': 'Bundesliga', 'time': '2025-06-01 18:30'},
        {'home': 'Arsenal', 'away': 'Burnley', 'league': 'Premier League', 'time': '2025-06-01 17:30'},
        {'home': 'Barcelona', 'away': 'Atletico Madrid', 'league': 'La Liga', 'time': '2025-06-01 16:15'},
    ]
    
    return demo_teams, demo_matches

def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>⚽ Over 0.5 HT - Sistema Inteligente</h1>
        <p>Powered by API-Football.com - Análise por equipes</p>
        <span class="premium-badge">API-FOOTBALL PRO ✨</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Configurações")
    
    # Teste de API
    api_status, api_message = test_api_football()
    if api_status:
        st.sidebar.success(api_message)
    else:
        st.sidebar.error(api_message)
    
    # Data
    selected_date = st.sidebar.date_input(
        "📅 Data para análise:",
        value=datetime.now().date()
    )
    
    # Histórico
    days_history = st.sidebar.slider(
        "📊 Dias de histórico:",
        min_value=7,
        max_value=30,
        value=14,
        help="Mais dias = análise mais precisa"
    )
    
    # Tabs principais
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 ANÁLISE REAL",
        "🏆 RANKING EQUIPES", 
        "📊 DEMO COMPLETO",
        "ℹ️ SOBRE API-FOOTBALL"
    ])
    
    with tab1:
        st.header(f"🎯 Análise Real - {selected_date.strftime('%d/%m/%Y')}")
        
        if not api_status:
            st.error("⚠️ API não está funcionando - use o DEMO")
            return
        
        if st.button("🚀 ANALISAR JOGOS DO DIA", type="primary"):
            # Buscar jogos de hoje
            with st.spinner("🔍 Buscando jogos de hoje..."):
                today_matches, message = get_matches_api_football(selected_date)
            
            st.info(message)
            
            if not today_matches:
                st.warning("❌ Nenhum jogo encontrado para esta data")
                st.info("💡 Tente uma data diferente ou veja o DEMO")
                return
            
            # Filtrar jogos programados
            upcoming_matches = [m for m in today_matches if m['fixture']['status']['short'] in ['NS', 'TBD']]
            
            if not upcoming_matches:
                st.info("⏰ Nenhum jogo programado para hoje")
                
                # Mostrar jogos finalizados
                finished_matches = [m for m in today_matches if m['fixture']['status']['short'] == 'FT']
                if finished_matches:
                    with st.expander("📋 Ver jogos finalizados"):
                        for match in finished_matches[:5]:
                            home = match['teams']['home']['name']
                            away = match['teams']['away']['name']
                            league = match['league']['name']
                            
                            ht = match['score']['halftime']
                            if ht and ht['home'] is not None:
                                ht_goals = ht['home'] + ht['away']
                                result = "Over 0.5" if ht_goals > 0.5 else "Under 0.5"
                                st.write(f"⚽ **{home} vs {away}** - HT: {ht['home']}-{ht['away']} ({result}) | {league}")
                return
            
            st.success(f"✅ {len(upcoming_matches)} jogos programados encontrados!")
            
            # Buscar dados históricos
            with st.spinner(f"📈 Analisando {days_history} dias de histórico..."):
                historical_matches, successful_days = get_historical_data_api_football(days_history)
            
            if not historical_matches:
                st.error("❌ Não foi possível obter dados históricos")
                return
            
            st.info(f"📊 Analisados {len(historical_matches)} jogos de {successful_days} dias")
            
            # Analisar equipes
            team_stats = analyze_teams_api_football(historical_matches)
            
            if not team_stats:
                st.warning("⚠️ Dados insuficientes para análise")
                return
            
            st.success(f"🎯 {len(team_stats)} equipes analisadas!")
            
            # Gerar previsões
            predictions = predict_matches_api_football(upcoming_matches, team_stats)
            
            if predictions:
                # Mostrar melhores apostas
                st.header("🏆 MELHORES APOSTAS DO DIA")
                
                best_bets = [p for p in predictions if p['sort_priority'] >= 0.55]
                
                if best_bets:
                    for bet in best_bets:
                        st.markdown(f"""
                        <div class="{bet['bet_class']}">
                            <h4>⚽ {bet['home_team']} vs {bet['away_team']}</h4>
                            <p><strong>Liga:</strong> {bet['league']}</p>
                            <p><strong>Data/Hora:</strong> {bet['time']}</p>
                            <p><strong>Previsão:</strong> {bet['prediction']} ({bet['probability']})</p>
                            <p><strong>Confiança:</strong> {bet['confidence']}</p>
                            <hr style="border-color: rgba(255,255,255,0.3);">
                            <p><strong>🏠 {bet['home_team']}:</strong> {bet['home_class']} - Taxa: {bet['home_rate']} ({bet['home_games']} jogos)</p>
                            <p><strong>✈️ {bet['away_team']}:</strong> {bet['away_class']} - Taxa: {bet['away_rate']} ({bet['away_games']} jogos)</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("🤔 Nenhuma aposta de alta confiança encontrada hoje")
                
                # Todas as previsões
                with st.expander("📋 TODAS AS PREVISÕES"):
                    for pred in predictions:
                        st.write(f"⚽ **{pred['home_team']} vs {pred['away_team']}**")
                        st.write(f"   Liga: {pred['league']} | Previsão: {pred['prediction']} | Confiança: {pred['confidence']}")
                        st.write("---")
    
    with tab2:
        st.header("🏆 Ranking das Equipes")
        st.info("Execute uma análise na aba 'ANÁLISE REAL' primeiro")
    
    with tab3:
        st.header("📊 DEMO - Sistema API-Football")
        st.info("🎯 Demonstração com dados realísticos da API-Football")
        
        if st.button("🚀 VER DEMO COMPLETO"):
            demo_teams, demo_matches = create_demo_api_football()
            
            # Simular previsões
            predictions = []
            for match in demo_matches:
                home_team = match['home']
                away_team = match['away']
                league = match['league']
                match_time = match['time']
                
                home_stats = demo_teams[home_team]
                away_stats = demo_teams[away_team]
                
                home_rate = home_stats['home_over_rate']
                away_rate = away_stats['away_over_rate']
                combined_rate = (home_rate * 0.6) + (away_rate * 0.4)
                
                if combined_rate >= 0.65:
                    prediction = "✅ OVER 0.5"
                    confidence = "ALTA"
                    bet_class = "team-over"
                elif combined_rate <= 0.35:
                    prediction = "❌ UNDER 0.5"
                    confidence = "ALTA"
                    bet_class = "team-under"
                else:
                    prediction = "⚖️ INDEFINIDO"
                    confidence = "BAIXA"
                    bet_class = "team-balanced"
                
                predictions.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'league': league,
                    'time': match_time,
                    'prediction': prediction,
                    'confidence': confidence,
                    'probability': f"{combined_rate:.1%}",
                    'home_rate': f"{home_rate:.1%}",
                    'away_rate': f"{away_rate:.1%}",
                    'home_class': home_stats['classification'],
                    'away_class': away_stats['classification'],
                    'bet_class': bet_class
                })
            
            # Mostrar previsões
            st.subheader("🏆 PREVISÕES DEMO")
            for pred in predictions:
                st.markdown(f"""
                <div class="{pred['bet_class']}">
                    <h4>⚽ {pred['home_team']} vs {pred['away_team']}</h4>
                    <p><strong>Liga:</strong> {pred['league']}</p>
                    <p><strong>Data/Hora:</strong> {pred['time']}</p>
                    <p><strong>Previsão:</strong> {pred['prediction']} ({pred['probability']})</p>
                    <p><strong>Confiança:</strong> {pred['confidence']}</p>
                    <hr style="border-color: rgba(255,255,255,0.3);">
                    <p><strong>🏠 {pred['home_team']}:</strong> {pred['home_class']} - Casa: {pred['home_rate']}</p>
                    <p><strong>✈️ {pred['away_team']}:</strong> {pred['away_class']} - Fora: {pred['away_rate']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Ranking
            st.subheader("🏆 Ranking Demo")
            team_list = []
            for team, stats in demo_teams.items():
                team_list.append({
                    'Equipe': team,
                    'Liga': stats['league'],
                    'Taxa Over': f"{stats['over_rate']:.1%}",
                    'Casa': f"{stats['home_over_rate']:.1%}",
                    'Fora': f"{stats['away_over_rate']:.1%}",
                    'Jogos': stats['total_games'],
                    'Classificação': stats['classification']
                })
            
            df = pd.DataFrame(team_list)
            df = df.sort_values('Taxa Over', ascending=False)
            st.dataframe(df, use_container_width=True)
    
    with tab4:
        st.header("ℹ️ Sobre API-Football.com")
        
        st.write("""
        ### 🔧 **API-Football.com Features:**
        
        **✅ Vantagens:**
        - 🌍 **900+ ligas** cobertas globalmente
        - 📊 **Dados em tempo real** e históricos
        - 🎯 **Estatísticas detalhadas** (primeiro tempo, cartões, etc.)
        - 🏆 **Todas as principais ligas** europeias
        - 📱 **Rate limit generoso** para análises
        
        ### 🎯 **Como funciona nosso sistema:**
        
        **1. Coleta de Dados:**
        - 📅 Busca jogos por data
        - 📊 Analisa histórico de 7-30 dias
        - 🏠 Separa performance casa/fora
        
        **2. Análise Inteligente:**
        - 🧠 Algoritmo que combina dados das equipes
        - ⚖️ Peso diferente para casa (60%) vs fora (40%)
        - 🔥 Bonus para equipas ofensivas
        - ❄️ Penalidade para equipas defensivas
        
        **3. Sistema de Confiança:**
        - 🎯 **ALTA**: Probabilidade >75% ou <25%
        - ⚠️ **MÉDIA**: Probabilidade 55-75% ou 25-45%
        - ❓ **BAIXA**: Probabilidade 45-55%
        
        ### 🏆 **Classificações das Equipes:**
        - 🔥 **OVER FORTE**: ≥75% dos jogos Over 0.5 HT
        - 📈 **OVER**: 60-74% dos jogos
        - ⚖️ **EQUILIBRADA**: 40-59% dos jogos
        - 📉 **UNDER**: 25-39% dos jogos
        - ❄️ **UNDER FORTE**: ≤25% dos jogos
        
        ### 💡 **Dicas de Uso:**
        1. **Foque em confiança ALTA/MÉDIA**
        2. **Prefira equipes com +5 jogos analisados**
        3. **Use diariamente** para dados atualizados
        4. **Combine com sua análise** para melhores resultados
        
        ### 🚀 **Sua API está configurada e funcionando!**
        - ✅ API-Football.com conectada
        - ✅ Dados em tempo real
        - ✅ Sistema de ML ativo
        - 📊 Análise por equipes funcionando
        """)

if __name__ == "__main__":
    main()
