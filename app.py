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

# SUA API Key
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
</style>
""", unsafe_allow_html=True)

def get_real_fixtures(date):
    """Busca jogos REAIS da API-Football"""
    headers = {
        'X-RapidAPI-Key': API_KEY,
        'X-RapidAPI-Host': 'v3.football.api-sports.io'
    }
    
    date_str = date.strftime('%Y-%m-%d')
    
    try:
        response = requests.get(
            'https://v3.football.api-sports.io/fixtures',
            headers=headers,
            params={'date': date_str},
            timeout=20
        )
        
        if response.status_code == 200:
            data = response.json()
            matches = data.get('response', [])
            return matches, f"✅ {len(matches)} jogos reais encontrados"
        
        elif response.status_code == 429:
            return [], "⚠️ Rate limit - aguarde alguns segundos"
        
        elif response.status_code == 403:
            return [], "❌ API Key sem permissões"
        
        else:
            return [], f"❌ Erro API: {response.status_code}"
            
    except Exception as e:
        return [], f"❌ Erro conexão: {str(e)}"

def get_real_historical_data(days=14):
    """Busca dados históricos REAIS"""
    all_matches = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    successful_days = 0
    
    for i in range(days):
        date = datetime.now() - timedelta(days=i+1)
        status_text.text(f"📊 Buscando dados reais: {date.strftime('%d/%m/%Y')} ({i+1}/{days})")
        
        matches, message = get_real_fixtures(date)
        
        if matches:
            # Apenas jogos finalizados com dados do primeiro tempo
            finished_matches = []
            for match in matches:
                if (match['fixture']['status']['short'] == 'FT' and 
                    match.get('score', {}).get('halftime', {}).get('home') is not None):
                    finished_matches.append(match)
            
            if finished_matches:
                all_matches.extend(finished_matches)
                successful_days += 1
        
        progress_bar.progress((i+1)/days)
        time.sleep(0.5)  # Rate limiting
    
    progress_bar.empty()
    status_text.empty()
    
    return all_matches, successful_days

def analyze_real_teams(matches):
    """Analisa equipes REAIS baseado em dados da API"""
    team_data = {}
    
    for match in matches:
        home_team = match['teams']['home']['name']
        away_team = match['teams']['away']['name']
        league = match['league']['name']
        country = match['league']['country']
        
        ht_home = match['score']['halftime']['home']
        ht_away = match['score']['halftime']['away']
        total_ht = ht_home + ht_away
        over_05 = 1 if total_ht > 0.5 else 0
        
        # Dados da equipe da casa
        if home_team not in team_data:
            team_data[home_team] = {
                'league': league,
                'country': country,
                'home_games': 0, 'away_games': 0,
                'home_over': 0, 'away_over': 0,
                'home_goals_scored': 0, 'away_goals_scored': 0,
                'total_ht_goals': 0
            }
        
        team_data[home_team]['home_games'] += 1
        team_data[home_team]['home_over'] += over_05
        team_data[home_team]['home_goals_scored'] += ht_home
        team_data[home_team]['total_ht_goals'] += ht_home
        
        # Dados da equipe visitante
        if away_team not in team_data:
            team_data[away_team] = {
                'league': league,
                'country': country,
                'home_games': 0, 'away_games': 0,
                'home_over': 0, 'away_over': 0,
                'home_goals_scored': 0, 'away_goals_scored': 0,
                'total_ht_goals': 0
            }
        
        team_data[away_team]['away_games'] += 1
        team_data[away_team]['away_over'] += over_05
        team_data[away_team]['away_goals_scored'] += ht_away
        team_data[away_team]['total_ht_goals'] += ht_away
    
    # Calcular estatísticas finais
    team_stats = {}
    for team, data in team_data.items():
        total_games = data['home_games'] + data['away_games']
        
        if total_games >= 5:  # Mínimo 5 jogos para análise confiável
            total_over = data['home_over'] + data['away_over']
            over_rate = total_over / total_games
            
            home_over_rate = data['home_over'] / max(data['home_games'], 1)
            away_over_rate = data['away_over'] / max(data['away_games'], 1)
            
            avg_ht_goals = data['total_ht_goals'] / total_games
            
            # Classificação baseada em dados reais
            if over_rate >= 0.75:
                classification = "🔥 EQUIPE OVER FORTE"
            elif over_rate >= 0.60:
                classification = "📈 EQUIPE OVER"
            elif over_rate <= 0.25:
                classification = "❄️ EQUIPE UNDER FORTE"
            elif over_rate <= 0.40:
                classification = "📉 EQUIPE UNDER"
            else:
                classification = "⚖️ EQUILIBRADA"
            
            team_stats[team] = {
                'league': data['league'],
                'country': data['country'],
                'over_rate': over_rate,
                'home_over_rate': home_over_rate,
                'away_over_rate': away_over_rate,
                'total_games': total_games,
                'home_games': data['home_games'],
                'away_games': data['away_games'],
                'avg_ht_goals': avg_ht_goals,
                'classification': classification
            }
    
    return team_stats

def predict_real_matches(today_fixtures, team_stats):
    """Previsões baseadas em dados REAIS"""
    predictions = []
    
    for fixture in today_fixtures:
        # Apenas jogos não iniciados
        if fixture['fixture']['status']['short'] not in ['NS', 'TBD']:
            continue
            
        home_team = fixture['teams']['home']['name']
        away_team = fixture['teams']['away']['name']
        league = fixture['league']['name']
        country = fixture['league']['country']
        
        home_stats = team_stats.get(home_team)
        away_stats = team_stats.get(away_team)
        
        if not home_stats or not away_stats:
            continue  # Pular se não temos dados históricos
        
        # Algoritmo baseado em dados reais
        home_rate = home_stats['home_over_rate']
        away_rate = away_stats['away_over_rate']
        
        # Cálculo da probabilidade
        base_probability = (home_rate * 0.6) + (away_rate * 0.4)
        
        # Ajustes baseados no histórico geral
        home_general = home_stats['over_rate']
        away_general = away_stats['over_rate']
        
        if home_general > 0.7 and away_general > 0.7:
            base_probability += 0.05
        elif home_general < 0.3 and away_general < 0.3:
            base_probability -= 0.05
        
        final_probability = max(0.05, min(0.95, base_probability))
        
        # Classificação da previsão
        if final_probability >= 0.70:
            prediction = "✅ OVER 0.5"
            confidence = "ALTA"
            bet_class = "team-over"
        elif final_probability >= 0.55:
            prediction = "✅ OVER 0.5"
            confidence = "MÉDIA"
            bet_class = "team-over"
        elif final_probability <= 0.30:
            prediction = "❌ UNDER 0.5"
            confidence = "ALTA"
            bet_class = "team-under"
        elif final_probability <= 0.45:
            prediction = "❌ UNDER 0.5"
            confidence = "MÉDIA"
            bet_class = "team-under"
        else:
            prediction = "⚖️ EVITAR"
            confidence = "BAIXA"
            bet_class = "team-balanced"
        
        predictions.append({
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
            'country': country,
            'time': fixture['fixture']['date'][:16],
            'prediction': prediction,
            'confidence': confidence,
            'probability': f"{final_probability:.1%}",
            'home_rate': f"{home_rate:.1%}",
            'away_rate': f"{away_rate:.1%}",
            'home_class': home_stats['classification'],
            'away_class': away_stats['classification'],
            'home_games': home_stats['total_games'],
            'away_games': away_stats['total_games'],
            'bet_class': bet_class,
            'sort_priority': final_probability if final_probability >= 0.55 else 0
        })
    
    predictions.sort(key=lambda x: x['sort_priority'], reverse=True)
    return predictions

def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>⚽ Over 0.5 HT - DADOS REAIS API-Football</h1>
        <p>Sistema com dados 100% reais de 900+ ligas mundiais</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Configurações")
    st.sidebar.success("🔑 API-Football conectada")
    
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
        value=14
    )
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🎯 ANÁLISE REAL",
        "🏆 RANKING REAL", 
        "📊 LIGAS REAIS"
    ])
    
    with tab1:
        st.header(f"🎯 Análise Over 0.5 HT - {selected_date.strftime('%d/%m/%Y')}")
        st.info("🌍 Buscando jogos reais de todas as ligas mundiais via API-Football")
        
        if st.button("🚀 BUSCAR JOGOS REAIS", type="primary"):
            # Buscar jogos de hoje
            with st.spinner("🔍 Buscando jogos reais de hoje..."):
                today_fixtures, message = get_real_fixtures(selected_date)
            
            st.info(message)
            
            if not today_fixtures:
                st.warning("❌ Nenhum jogo real encontrado para esta data")
                st.info("💡 Tente uma data diferente ou verifique se é dia de jogos")
                return
            
            # Mostrar jogos encontrados por liga/país
            st.subheader("🌍 Jogos Encontrados por País/Liga")
            
            league_summary = {}
            upcoming_count = 0
            
            for fixture in today_fixtures:
                country = fixture['league']['country']
                league = fixture['league']['name']
                status = fixture['fixture']['status']['short']
                
                if country not in league_summary:
                    league_summary[country] = {}
                if league not in league_summary[country]:
                    league_summary[country][league] = {'total': 0, 'upcoming': 0}
                
                league_summary[country][league]['total'] += 1
                if status in ['NS', 'TBD']:
                    league_summary[country][league]['upcoming'] += 1
                    upcoming_count += 1
            
            # Mostrar resumo por país
            for country, leagues in league_summary.items():
                with st.expander(f"🏴 {country} ({sum([l['total'] for l in leagues.values()])} jogos)"):
                    for league, counts in leagues.items():
                        st.write(f"🏆 **{league}**: {counts['total']} jogos ({counts['upcoming']} programados)")
            
            st.success(f"✅ Total: {len(today_fixtures)} jogos | 📅 Programados: {upcoming_count}")
            
            if upcoming_count == 0:
                st.info("⏰ Nenhum jogo programado - apenas jogos finalizados")
                return
            
            # Buscar dados históricos
            with st.spinner(f"📈 Analisando {days_history} dias de dados históricos REAIS..."):
                historical_fixtures, successful_days = get_real_historical_data(days_history)
            
            if not historical_fixtures:
                st.error("❌ Não foi possível obter dados históricos")
                return
            
            st.info(f"📊 {len(historical_fixtures)} jogos históricos analisados de {successful_days} dias")
            
            # Analisar equipes
            with st.spinner("🧠 Analisando padrões REAIS das equipes..."):
                team_stats = analyze_real_teams(historical_fixtures)
            
            if not team_stats:
                st.warning("⚠️ Dados insuficientes para análise")
                return
            
            st.success(f"🎯 {len(team_stats)} equipes reais analisadas!")
            
            # Gerar previsões
            predictions = predict_real_matches(today_fixtures, team_stats)
            
            if predictions:
                # Estatísticas
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🎯 Previsões", len(predictions))
                
                with col2:
                    high_conf = len([p for p in predictions if p['confidence'] == 'ALTA'])
                    st.metric("🔥 Alta Confiança", high_conf)
                
                with col3:
                    over_pred = len([p for p in predictions if 'OVER' in p['prediction']])
                    st.metric("📈 Over 0.5", over_pred)
                
                with col4:
                    under_pred = len([p for p in predictions if 'UNDER' in p['prediction']])
                    st.metric("📉 Under 0.5", under_pred)
                
                # Melhores apostas
                st.header("🏆 MELHORES APOSTAS REAIS")
                
                best_bets = [p for p in predictions if p['confidence'] in ['ALTA', 'MÉDIA'] and p['sort_priority'] > 0]
                
                if best_bets:
                    for bet in best_bets[:10]:  # Top 10 apostas
                        st.markdown(f"""
                        <div class="{bet['bet_class']}">
                            <h4>⚽ {bet['home_team']} vs {bet['away_team']}</h4>
                            <p><strong>🏆 Liga:</strong> {bet['league']} ({bet['country']})</p>
                            <p><strong>🕐 Horário:</strong> {bet['time']}</p>
                            <p><strong>🎯 Previsão:</strong> {bet['prediction']} ({bet['probability']})</p>
                            <p><strong>📊 Confiança:</strong> {bet['confidence']}</p>
                            <hr style="border-color: rgba(255,255,255,0.3);">
                            <p><strong>🏠 {bet['home_team']}:</strong> {bet['home_class']} - Casa: {bet['home_rate']} ({bet['home_games']} jogos)</p>
                            <p><strong>✈️ {bet['away_team']}:</strong> {bet['away_class']} - Fora: {bet['away_rate']} ({bet['away_games']} jogos)</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("🤔 Nenhuma aposta de boa confiança encontrada hoje")
                
                # Todas as previsões
                with st.expander("📋 TODAS AS PREVISÕES REAIS"):
                    for pred in predictions:
                        st.write(f"⚽ **{pred['home_team']} vs {pred['away_team']}**")
                        st.write(f"   🏆 {pred['league']} ({pred['country']}) | 🎯 {pred['prediction']} ({pred['probability']}) | 📊 {pred['confidence']}")
                        st.write("---")
            
            else:
                st.info("📊 Nenhuma previsão disponível (equipes sem dados históricos suficientes)")
    
    with tab2:
        st.header("🏆 Ranking das Equipes REAIS")
        st.info("Execute uma análise na aba 'ANÁLISE REAL' primeiro")
    
    with tab3:
        st.header("📊 Estatísticas das Ligas REAIS")
        st.info("Execute uma análise na aba 'ANÁLISE REAL' primeiro")

if __name__ == "__main__":
    main()
