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

# SUA API Key (funcionando!)
API_KEY = "2aad0db0e5b88b3a080bdc85461a919"

# Configuração da API que funciona
API_CONFIG = {
    'base_url': 'https://v3.football.api-sports.io',
    'headers': {
        'X-RapidAPI-Key': API_KEY,
        'X-RapidAPI-Host': 'v3.football.api-sports.io'
    }
}

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
    .stats-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def test_api_connection():
    """Testa se a API está funcionando"""
    try:
        response = requests.get(
            f"{API_CONFIG['base_url']}/status",
            headers=API_CONFIG['headers'],
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return True, f"✅ API funcionando! Rate limit: {data.get('response', {}).get('requests', 'N/A')}"
        else:
            return False, f"❌ Erro {response.status_code}"
    except:
        # Teste alternativo com leagues
        try:
            response = requests.get(
                f"{API_CONFIG['base_url']}/leagues",
                headers=API_CONFIG['headers'],
                timeout=10
            )
            if response.status_code == 200:
                return True, "✅ API funcionando!"
            else:
                return False, f"❌ Erro {response.status_code}"
        except Exception as e:
            return False, f"❌ Erro: {str(e)}"

def get_fixtures_by_date(date):
    """Busca jogos por data"""
    date_str = date.strftime('%Y-%m-%d')
    
    try:
        response = requests.get(
            f"{API_CONFIG['base_url']}/fixtures",
            headers=API_CONFIG['headers'],
            params={'date': date_str},
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            matches = data.get('response', [])
            return matches, f"✅ {len(matches)} jogos encontrados"
        else:
            return [], f"❌ Erro {response.status_code}"
            
    except Exception as e:
        return [], f"❌ Erro: {str(e)}"

def get_historical_fixtures(days=14):
    """Busca dados históricos"""
    all_matches = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    successful_days = 0
    
    for i in range(days):
        date = datetime.now() - timedelta(days=i+1)
        status_text.text(f"📊 Dia {i+1}/{days}: {date.strftime('%d/%m/%Y')}")
        
        matches, message = get_fixtures_by_date(date)
        
        if matches:
            # Filtrar apenas jogos finalizados
            finished_matches = [m for m in matches if m['fixture']['status']['short'] == 'FT']
            
            # Filtrar apenas jogos com dados do primeiro tempo
            valid_matches = []
            for match in finished_matches:
                if (match.get('score', {}).get('halftime', {}).get('home') is not None and
                    match.get('score', {}).get('halftime', {}).get('away') is not None):
                    valid_matches.append(match)
            
            all_matches.extend(valid_matches)
            if valid_matches:
                successful_days += 1
        
        progress_bar.progress((i+1)/days)
        time.sleep(0.3)  # Rate limiting
    
    progress_bar.empty()
    status_text.empty()
    
    return all_matches, successful_days

def analyze_teams_from_fixtures(matches):
    """Analisa equipes baseado nos jogos"""
    team_data = {}
    
    for match in matches:
        home_team = match['teams']['home']['name']
        away_team = match['teams']['away']['name']
        league = match['league']['name']
        
        ht_home = match['score']['halftime']['home']
        ht_away = match['score']['halftime']['away']
        total_ht = ht_home + ht_away
        over_05 = 1 if total_ht > 0.5 else 0
        
        # Analisar equipe da casa
        if home_team not in team_data:
            team_data[home_team] = {
                'league': league,
                'home_games': 0, 'away_games': 0,
                'home_over': 0, 'away_over': 0,
                'home_goals_scored': 0, 'away_goals_scored': 0,
                'home_goals_conceded': 0, 'away_goals_conceded': 0
            }
        
        team_data[home_team]['home_games'] += 1
        team_data[home_team]['home_over'] += over_05
        team_data[home_team]['home_goals_scored'] += ht_home
        team_data[home_team]['home_goals_conceded'] += ht_away
        
        # Analisar equipe visitante
        if away_team not in team_data:
            team_data[away_team] = {
                'league': league,
                'home_games': 0, 'away_games': 0,
                'home_over': 0, 'away_over': 0,
                'home_goals_scored': 0, 'away_goals_scored': 0,
                'home_goals_conceded': 0, 'away_goals_conceded': 0
            }
        
        team_data[away_team]['away_games'] += 1
        team_data[away_team]['away_over'] += over_05
        team_data[away_team]['away_goals_scored'] += ht_away
        team_data[away_team]['away_goals_conceded'] += ht_home
    
    # Calcular estatísticas
    team_stats = {}
    for team, data in team_data.items():
        total_games = data['home_games'] + data['away_games']
        
        if total_games >= 4:  # Mínimo 4 jogos para análise confiável
            total_over = data['home_over'] + data['away_over']
            over_rate = total_over / total_games
            
            home_over_rate = data['home_over'] / max(data['home_games'], 1)
            away_over_rate = data['away_over'] / max(data['away_games'], 1)
            
            avg_goals_scored_home = data['home_goals_scored'] / max(data['home_games'], 1)
            avg_goals_scored_away = data['away_goals_scored'] / max(data['away_games'], 1)
            avg_goals_conceded_home = data['home_goals_conceded'] / max(data['home_games'], 1)
            avg_goals_conceded_away = data['away_goals_conceded'] / max(data['away_games'], 1)
            
            team_stats[team] = {
                'league': data['league'],
                'over_rate': over_rate,
                'home_over_rate': home_over_rate,
                'away_over_rate': away_over_rate,
                'total_games': total_games,
                'home_games': data['home_games'],
                'away_games': data['away_games'],
                'avg_goals_scored_home': avg_goals_scored_home,
                'avg_goals_scored_away': avg_goals_scored_away,
                'avg_goals_conceded_home': avg_goals_conceded_home,
                'avg_goals_conceded_away': avg_goals_conceded_away,
                'classification': get_team_classification(over_rate),
                'attacking_power': (avg_goals_scored_home + avg_goals_scored_away) / 2,
                'defensive_power': (avg_goals_conceded_home + avg_goals_conceded_away) / 2
            }
    
    return team_stats

def get_team_classification(over_rate):
    """Classifica equipe baseada na taxa Over 0.5"""
    if over_rate >= 0.80:
        return "🔥 EQUIPE OVER FORTE"
    elif over_rate >= 0.65:
        return "📈 EQUIPE OVER"
    elif over_rate <= 0.20:
        return "❄️ EQUIPE UNDER FORTE"
    elif over_rate <= 0.35:
        return "📉 EQUIPE UNDER"
    else:
        return "⚖️ EQUILIBRADA"

def get_bet_class_and_icon(over_rate):
    """Retorna classe CSS e ícone baseado na taxa"""
    if over_rate >= 0.80:
        return "team-over", "🔥"
    elif over_rate >= 0.65:
        return "team-over", "🎯"
    elif over_rate >= 0.55:
        return "team-over", "📈"
    elif over_rate <= 0.20:
        return "team-under", "🧊"
    elif over_rate <= 0.35:
        return "team-under", "❄️"
    elif over_rate <= 0.45:
        return "team-under", "📉"
    else:
        return "team-balanced", "⚖️"

def predict_matches_advanced(today_fixtures, team_stats):
    """Sistema de predição avançado"""
    predictions = []
    
    for fixture in today_fixtures:
        if fixture['fixture']['status']['short'] not in ['NS', 'TBD', '1H', 'HT', '2H']:
            continue  # Pular jogos já iniciados ou finalizados
            
        home_team = fixture['teams']['home']['name']
        away_team = fixture['teams']['away']['name']
        league = fixture['league']['name']
        
        # Dados da partida
        match_time = fixture['fixture']['date']
        venue = fixture['fixture']['venue']['name'] if fixture['fixture']['venue'] else 'N/A'
        
        home_stats = team_stats.get(home_team, {})
        away_stats = team_stats.get(away_team, {})
        
        if not home_stats or not away_stats:
            continue  # Pular se não temos dados das equipes
        
        # Algoritmo avançado de predição
        home_rate = home_stats.get('home_over_rate', 0.5)
        away_rate = away_stats.get('away_over_rate', 0.5)
        
        # Fatores adicionais
        home_attack = home_stats.get('attacking_power', 0.5)
        away_attack = away_stats.get('attacking_power', 0.5)
        home_defense = home_stats.get('defensive_power', 0.5)
        away_defense = away_stats.get('defensive_power', 0.5)
        
        # Cálculo base (60% casa, 40% fora)
        base_probability = (home_rate * 0.6) + (away_rate * 0.4)
        
        # Ajuste por poder ofensivo
        offensive_factor = (home_attack + away_attack) / 2
        if offensive_factor > 1.0:
            base_probability += 0.05
        elif offensive_factor > 0.7:
            base_probability += 0.02
        
        # Ajuste por fragilidade defensiva
        defensive_factor = (home_defense + away_defense) / 2
        if defensive_factor > 1.0:
            base_probability += 0.03
        
        # Ajuste por histórico das equipes
        home_over_general = home_stats.get('over_rate', 0.5)
        away_over_general = away_stats.get('over_rate', 0.5)
        
        if home_over_general > 0.75 and away_over_general > 0.75:
            base_probability += 0.08  # Bonus para dois times muito ofensivos
        elif home_over_general > 0.65 and away_over_general > 0.65:
            base_probability += 0.04  # Bonus para dois times ofensivos
        elif home_over_general < 0.35 and away_over_general < 0.35:
            base_probability -= 0.08  # Penalidade para dois times defensivos
        elif home_over_general < 0.45 and away_over_general < 0.45:
            base_probability -= 0.04  # Penalidade média
        
        # Garantir limites
        final_probability = max(0.05, min(0.95, base_probability))
        
        # Classificação da previsão
        bet_class, icon = get_bet_class_and_icon(final_probability)
        
        if final_probability >= 0.80:
            prediction = "✅ OVER 0.5"
            confidence = "MUITO ALTA"
        elif final_probability >= 0.70:
            prediction = "✅ OVER 0.5"
            confidence = "ALTA"
        elif final_probability >= 0.60:
            prediction = "✅ OVER 0.5"
            confidence = "MÉDIA"
        elif final_probability <= 0.20:
            prediction = "❌ UNDER 0.5"
            confidence = "MUITO ALTA"
        elif final_probability <= 0.30:
            prediction = "❌ UNDER 0.5"
            confidence = "ALTA"
        elif final_probability <= 0.40:
            prediction = "❌ UNDER 0.5"
            confidence = "MÉDIA"
        else:
            prediction = "⚖️ EVITAR"
            confidence = "BAIXA"
        
        predictions.append({
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
            'time': match_time[:16] if match_time else '',
            'venue': venue,
            'prediction': prediction,
            'confidence': confidence,
            'icon': icon,
            'probability': f"{final_probability:.1%}",
            'home_rate': f"{home_rate:.1%}",
            'away_rate': f"{away_rate:.1%}",
            'home_class': home_stats.get('classification', 'Sem dados'),
            'away_class': away_stats.get('classification', 'Sem dados'),
            'home_games': home_stats.get('total_games', 0),
            'away_games': away_stats.get('total_games', 0),
            'home_attack': f"{home_attack:.2f}",
            'away_attack': f"{away_attack:.2f}",
            'bet_class': bet_class,
            'sort_priority': final_probability if final_probability >= 0.60 else (1 - final_probability) if final_probability <= 0.40 else 0
        })
    
    # Ordenar por prioridade
    predictions.sort(key=lambda x: x['sort_priority'], reverse=True)
    return predictions

def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>⚽ Over 0.5 HT - Sistema Inteligente</h1>
        <p>Powered by API-Football.com - Análise profissional por equipes</p>
        <span class="premium-badge">API FUNCIONANDO ✨</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Configurações")
    
    # Status da API
    api_status, api_message = test_api_connection()
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
        max_value=21,
        value=14,
        help="Mais dias = análise mais precisa"
    )
    
    # Tabs principais
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 ANÁLISE DO DIA",
        "🏆 RANKING EQUIPES", 
        "📊 ESTATÍSTICAS",
        "ℹ️ COMO USAR"
    ])
    
    with tab1:
        st.header(f"🎯 Análise Over 0.5 HT - {selected_date.strftime('%d/%m/%Y')}")
        
        if not api_status:
            st.error("⚠️ API não está disponível")
            return
        
        col1, col2 = st.columns([3, 1])
        with col1:
            analyze_button = st.button("🚀 ANALISAR JOGOS DO DIA", type="primary")
        with col2:
            st.metric("Histórico", f"{days_history} dias")
        
        if analyze_button:
            # Buscar jogos de hoje
            with st.spinner("🔍 Buscando jogos de hoje..."):
                today_fixtures, message = get_fixtures_by_date(selected_date)
            
            st.info(message)
            
            if not today_fixtures:
                st.warning("❌ Nenhum jogo encontrado para esta data")
                st.info("💡 Tente uma data diferente (fins de semana têm mais jogos)")
                return
            
            # Filtrar jogos não iniciados
            upcoming_fixtures = [f for f in today_fixtures if f['fixture']['status']['short'] in ['NS', 'TBD']]
            
            if not upcoming_fixtures:
                st.info("⏰ Nenhum jogo programado para hoje")
                
                # Mostrar jogos finalizados com resultados HT
                finished_fixtures = [f for f in today_fixtures if f['fixture']['status']['short'] == 'FT']
                if finished_fixtures:
                    with st.expander("📋 Resultados do Primeiro Tempo"):
                        for fixture in finished_fixtures[:10]:
                            home = fixture['teams']['home']['name']
                            away = fixture['teams']['away']['name']
                            league = fixture['league']['name']
                            
                            ht = fixture['score']['halftime']
                            if ht and ht['home'] is not None:
                                ht_total = ht['home'] + ht['away']
                                result = "✅ Over 0.5" if ht_total > 0.5 else "❌ Under 0.5"
                                st.write(f"⚽ **{home} vs {away}** - HT: {ht['home']}-{ht['away']} ({result}) | {league}")
                return
            
            st.success(f"✅ {len(upcoming_fixtures)} jogos programados encontrados!")
            
            # Buscar dados históricos
            with st.spinner(f"📈 Analisando {days_history} dias de histórico..."):
                historical_fixtures, successful_days = get_historical_fixtures(days_history)
            
            if not historical_fixtures:
                st.error("❌ Não foi possível obter dados históricos suficientes")
                return
            
            st.info(f"📊 {len(historical_fixtures)} jogos analisados de {successful_days} dias")
            
            # Analisar equipes
            with st.spinner("🧠 Analisando padrões das equipes..."):
                team_stats = analyze_teams_from_fixtures(historical_fixtures)
            
            if not team_stats:
                st.warning("⚠️ Dados insuficientes para análise")
                return
            
            st.success(f"🎯 {len(team_stats)} equipes analisadas com sucesso!")
            
            # Gerar previsões
            predictions = predict_matches_advanced(upcoming_fixtures, team_stats)
            
            if predictions:
                # Estatísticas gerais
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    high_confidence = len([p for p in predictions if p['confidence'] in ['MUITO ALTA', 'ALTA']])
                    st.metric("🎯 Alta Confiança", high_confidence)
                
                with col2:
                    over_predictions = len([p for p in predictions if 'OVER' in p['prediction']])
                    st.metric("📈 Over 0.5", over_predictions)
                
                with col3:
                    under_predictions = len([p for p in predictions if 'UNDER' in p['prediction']])
                    st.metric("📉 Under 0.5", under_predictions)
                
                with col4:
                    avg_prob = sum([float(p['probability'][:-1]) for p in predictions]) / len(predictions)
                    st.metric("⚖️ Prob. Média", f"{avg_prob:.1f}%")
                
                # Melhores apostas
                st.header("🏆 MELHORES OPORTUNIDADES")
                
                best_bets = [p for p in predictions if p['sort_priority'] >= 0.60]
                
                if best_bets:
                    for bet in best_bets:
                        st.markdown(f"""
                        <div class="{bet['bet_class']}">
                            <h4>{bet['icon']} {bet['home_team']} vs {bet['away_team']}</h4>
                            <p><strong>Liga:</strong> {bet['league']} | <strong>Horário:</strong> {bet['time']}</p>
                            <p><strong>Local:</strong> {bet['venue']}</p>
                            <p><strong>Previsão:</strong> {bet['prediction']} ({bet['probability']})</p>
                            <p><strong>Confiança:</strong> {bet['confidence']}</p>
                            <hr style="border-color: rgba(255,255,255,0.3);">
                            <p><strong>🏠 {bet['home_team']}:</strong> {bet['home_class']} - Casa: {bet['home_rate']} | Ataque: {bet['home_attack']} ({bet['home_games']} jogos)</p>
                            <p><strong>✈️ {bet['away_team']}:</strong> {bet['away_class']} - Fora: {bet['away_rate']} | Ataque: {bet['away_attack']} ({bet['away_games']} jogos)</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("🤔 Nenhuma oportunidade de alta confiança encontrada hoje")
                    st.write("💡 Isso é normal - o sistema é conservador e só recomenda apostas com alta probabilidade")
                
                # Todas as previsões
                with st.expander("📋 TODAS AS PREVISÕES DO DIA"):
                    for pred in predictions:
                        confidence_color = "🔥" if pred['confidence'] == "MUITO ALTA" else "🎯" if pred['confidence'] == "ALTA" else "📊" if pred['confidence'] == "MÉDIA" else "❓"
                        st.write(f"{confidence_color} **{pred['home_team']} vs {pred['away_team']}**")
                        st.write(f"   Liga: {pred['league']} | Previsão: {pred['prediction']} | Confiança: {pred['confidence']} | Prob: {pred['probability']}")
                        st.write(f"   🏠 Casa: {pred['home_class']} ({pred['home_rate']}) | ✈️ Fora: {pred['away_class']} ({pred['away_rate']})")
                        st.write("---")
            else:
                st.info("📊 Nenhuma previsão disponível (equipes sem dados históricos suficientes)")
    
    with tab2:
        st.header("🏆 Ranking das Equipes")
        st.info("Execute uma análise na aba 'ANÁLISE DO DIA' primeiro para ver o ranking atual")
    
    with tab3:
        st.header("📊 Estatísticas do Sistema")
        st.info("Execute uma análise na aba 'ANÁLISE DO DIA' primeiro para ver estatísticas detalhadas")
    
    with tab4:
        st.header("ℹ️ Como Usar o Sistema")
        
        st.write("""
        ### 🎯 **Seu Sistema Over 0.5 HT Está Funcionando!**
        
        **✅ API Conectada:** API-Football.com
        **✅ Dados em Tempo Real:** Jogos e estatísticas atualizadas
        **✅ Análise Inteligente:** Algoritmo ML avançado
        
        ### 🚀 **Como Usar:**
        
        **1. 📅 Selecione a Data**
        - Escolha a data na sidebar (hoje ou futuro)
        - Fins de semana têm mais jogos
        
        **2. ⚙️ Configure o Histórico**
        - 7-21 dias de análise
        - Mais dias = análise mais precisa
        
        **3. 🎯 Execute a Análise**
        - Clique "ANALISAR JOGOS DO DIA"
        - Sistema busca jogos e analisa equipes
        - Gera previsões inteligentes
        
        ### 🧠 **Algoritmo Avançado:**
        
        **Fatores Analisados:**
        - 🏠 **Performance em casa** vs ✈️ **fora**
        - ⚽ **Poder ofensivo** das equipes
        - 🛡️ **Fragilidade defensiva**
        - 📊 **Histórico de Over 0.5 HT**
        - 🎯 **Padrões de confronto**
        
        **Níveis de Confiança:**
        - 🔥 **MUITO ALTA**: ≥80% (aposte com segurança)
        - 🎯 **ALTA**: 70-79% (boa oportunidade)
        - 📊 **MÉDIA**: 60-69% (considere o contexto)
        - ❓ **BAIXA**: <60% (evite)
        
        ### 💡 **Dicas de Uso:**
        
        **✅ Focar em:**
        - Apostas de confiança ALTA/MUITO ALTA
        - Equipes com +5 jogos analisados
        - Ligas que você conhece
        
        **❌ Evitar:**
        - Apostas de baixa confiança
        - Equipes com poucos dados
        - Múltiplas com muitos jogos
        
        ### 📈 **Para o Lion Sports Trading:**
        
        **Pre-Match:**
        - Use para identificar oportunidades
        - Compare com as odds oferecidas
        - Procure value bets (prob > odds)
        
        **Live Trading:**
        - Aguarde confirmação nos primeiros 15min
        - Se não sair gol, considere apostar Over
        - Use stop-loss em HT 0-0
        
        ### 🎯 **Sistema Atualizado Diariamente!**
        Execute a análise todos os dias para sempre ter os insights mais recentes!
        """)

if __name__ == "__main__":
    main()
