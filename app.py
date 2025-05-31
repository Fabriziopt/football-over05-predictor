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

# SUA API Key correta
API_KEY = "0e43ab2d4fe34abcca071e6783fad72d"

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
    .api-test {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .success-test {
        background: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

def get_matches_api(date, api_key=API_KEY):
    """Busca jogos da API com sua chave"""
    endpoints = [
        {
            'url': 'https://api.football-data.org/v4/matches',
            'params': {'dateFrom': date.strftime('%Y-%m-%d'), 'dateTo': date.strftime('%Y-%m-%d')}
        },
        {
            'url': 'https://api.football-data.org/v2/matches',
            'params': {'dateFrom': date.strftime('%Y-%m-%d'), 'dateTo': date.strftime('%Y-%m-%d')}
        }
    ]
    
    headers = {'X-Auth-Token': api_key}
    
    for endpoint in endpoints:
        try:
            response = requests.get(
                endpoint['url'],
                headers=headers,
                params=endpoint['params'],
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                matches = data.get('matches', [])
                api_version = 'v4' if 'v4' in endpoint['url'] else 'v2'
                return matches, f"✅ {len(matches)} jogos encontrados (API {api_version})"
            
            elif response.status_code == 403:
                return [], "❌ API Key sem permissões para este endpoint"
            
            elif response.status_code == 429:
                time.sleep(2)
                continue
                
        except Exception as e:
            continue
    
    return [], "❌ Erro em todos os endpoints"

def test_api_connection():
    """Testa se a API está funcionando"""
    test_endpoints = [
        'https://api.football-data.org/v4/competitions',
        'https://api.football-data.org/v2/competitions'
    ]
    
    headers = {'X-Auth-Token': API_KEY}
    
    for url in test_endpoints:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                competitions = data.get('competitions', [])
                version = 'v4' if 'v4' in url else 'v2'
                return True, f"✅ API {version} funcionando ({len(competitions)} competições)"
            
        except:
            continue
    
    return False, "❌ API não disponível"

def get_historical_matches(days=14):
    """Busca dados históricos"""
    all_matches = []
    end_date = datetime.now()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    successful_days = 0
    
    for i in range(days):
        date = end_date - timedelta(days=i+1)
        status_text.text(f"📊 Buscando: {date.strftime('%d/%m/%Y')} ({i+1}/{days})")
        
        matches, message = get_matches_api(date)
        if matches:
            all_matches.extend(matches)
            successful_days += 1
        
        progress_bar.progress((i+1)/days)
        time.sleep(0.2)  # Rate limiting
    
    progress_bar.empty()
    status_text.empty()
    
    return all_matches, successful_days

def analyze_teams_from_matches(matches):
    """Analisa equipes baseado nos jogos reais"""
    team_data = {}
    
    for match in matches:
        if (match.get('status') == 'FINISHED' and 
            match.get('score', {}).get('halfTime')):
            
            league = match['competition']['name']
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            
            ht_score = match['score']['halfTime']
            if ht_score.get('home') is not None and ht_score.get('away') is not None:
                total_ht_goals = ht_score['home'] + ht_score['away']
                over_05 = 1 if total_ht_goals > 0.5 else 0
                
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
                team_data[home_team]['home_goals'] += ht_score['home']
                
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
                team_data[away_team]['away_goals'] += ht_score['away']
    
    # Calcular estatísticas
    team_stats = {}
    for team, data in team_data.items():
        total_games = data['home_games'] + data['away_games']
        
        if total_games >= 3:  # Mínimo 3 jogos
            total_over = data['home_over'] + data['away_over']
            over_rate = total_over / total_games
            
            # Taxas específicas
            home_over_rate = data['home_over'] / max(data['home_games'], 1)
            away_over_rate = data['away_over'] / max(data['away_games'], 1)
            
            # Média de gols
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

def predict_matches_real(today_matches, team_stats):
    """Faz previsões reais baseadas nos dados"""
    predictions = []
    
    for match in today_matches:
        if match.get('status') in ['SCHEDULED', 'TIMED']:
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            league = match['competition']['name']
            
            home_stats = team_stats.get(home_team, {})
            away_stats = team_stats.get(away_team, {})
            
            # Usar taxas específicas casa/fora quando disponível
            home_rate = home_stats.get('home_over_rate', home_stats.get('over_rate', 0.5))
            away_rate = away_stats.get('away_over_rate', away_stats.get('over_rate', 0.5))
            
            # Algoritmo de predição
            combined_rate = (home_rate * 0.6) + (away_rate * 0.4)
            
            # Ajustes baseados no histórico
            if home_stats.get('over_rate', 0) > 0.7 and away_stats.get('over_rate', 0) > 0.7:
                combined_rate += 0.05  # Bonus dois times ofensivos
            elif home_stats.get('over_rate', 0) < 0.3 and away_stats.get('over_rate', 0) < 0.3:
                combined_rate -= 0.05  # Penalidade dois times defensivos
            
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

def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>⚽ Over 0.5 HT - Análise por EQUIPES</h1>
        <p>Sistema inteligente que analisa equipes individuais dentro de cada liga</p>
        <span class="premium-badge">API PRO ATIVA ✨</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Configurações")
    st.sidebar.success("🔑 API Key: Configurada e Ativa")
    
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
    
    # Teste de conexão
    api_status, api_message = test_api_connection()
    if api_status:
        st.sidebar.success(api_message)
    else:
        st.sidebar.error(api_message)
    
    # Tabs principais
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 PREVISÕES INTELIGENTES", 
        "🏆 RANKING DE EQUIPES", 
        "📊 ANÁLISE POR LIGA",
        "ℹ️ COMO FUNCIONA"
    ])
    
    with tab1:
        st.header(f"🎯 Previsões Inteligentes - {selected_date.strftime('%d/%m/%Y')}")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            analyze_button = st.button(
                "🚀 ANALISAR JOGOS DO DIA",
                type="primary",
                use_container_width=True
            )
        with col2:
            st.metric("Histórico", f"{days_history} dias")
        
        if analyze_button:
            # Buscar jogos de hoje
            with st.spinner("🔍 Buscando jogos de hoje..."):
                today_matches, message = get_matches_api(selected_date)
            
            st.info(message)
            
            if not today_matches:
                st.warning("❌ Nenhum jogo encontrado para esta data")
                st.info("💡 Tente uma data diferente ou verifique se há jogos programados")
                return
            
            # Filtrar jogos programados
            upcoming_matches = [m for m in today_matches if m.get('status') in ['SCHEDULED', 'TIMED']]
            
            if not upcoming_matches:
                st.info("⏰ Nenhum jogo programado para hoje (apenas jogos finalizados)")
                with st.expander("📋 Ver jogos finalizados"):
                    finished_matches = [m for m in today_matches if m.get('status') == 'FINISHED']
                    for match in finished_matches[:5]:
                        ht_score = match.get('score', {}).get('halfTime', {})
                        if ht_score:
                            ht_goals = (ht_score.get('home', 0) or 0) + (ht_score.get('away', 0) or 0)
                            result = "✅ Over 0.5" if ht_goals > 0.5 else "❌ Under 0.5"
                            st.write(f"**{match['homeTeam']['name']} vs {match['awayTeam']['name']}** - HT: {ht_score.get('home', 0)}-{ht_score.get('away', 0)} ({result})")
                return
            
            st.success(f"✅ {len(upcoming_matches)} jogos programados encontrados!")
            
            # Buscar dados históricos
            with st.spinner(f"📈 Analisando {days_history} dias de histórico..."):
                historical_matches, successful_days = get_historical_matches(days_history)
            
            if not historical_matches:
                st.error("❌ Não foi possível obter dados históricos")
                return
            
            st.info(f"📊 Analisados {len(historical_matches)} jogos de {successful_days} dias")
            
            # Analisar equipes
            team_stats = analyze_teams_from_matches(historical_matches)
            
            if not team_stats:
                st.warning("⚠️ Dados insuficientes para análise de equipes")
                return
            
            st.success(f"🎯 {len(team_stats)} equipes analisadas com sucesso!")
            
            # Gerar previsões
            predictions = predict_matches_real(upcoming_matches, team_stats)
            
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
                            <p><strong>Previsão:</strong> {bet['prediction']} ({bet['probability']})</p>
                            <p><strong>Confiança:</strong> {bet['confidence']}</p>
                            <hr style="border-color: rgba(255,255,255,0.3);">
                            <p><strong>🏠 {bet['home_team']}:</strong> {bet['home_class']} - Taxa: {bet['home_rate']} ({bet['home_games']} jogos)</p>
                            <p><strong>✈️ {bet['away_team']}:</strong> {bet['away_class']} - Taxa: {bet['away_rate']} ({bet['away_games']} jogos)</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("🤔 Nenhuma aposta de alta confiança identificada hoje")
                
                # Mostrar todas as previsões
                with st.expander("📋 TODAS AS PREVISÕES DO DIA"):
                    for pred in predictions:
                        confidence_emoji = "🔥" if pred['confidence'] == "ALTA" else "⚠️" if pred['confidence'] == "MÉDIA" else "❓"
                        st.write(f"{confidence_emoji} **{pred['home_team']} vs {pred['away_team']}**")
                        st.write(f"   Liga: {pred['league']} | Previsão: {pred['prediction']} | Confiança: {pred['confidence']} | Prob: {pred['probability']}")
                        st.write(f"   Casa: {pred['home_class']} ({pred['home_rate']}) | Fora: {pred['away_class']} ({pred['away_rate']})")
                        st.write("---")
            else:
                st.info("📊 Nenhuma previsão disponível")
    
    with tab2:
        st.header("🏆 Ranking das Equipes")
        
        if st.button("📊 Gerar Ranking Completo"):
            with st.spinner("Analisando todas as equipes..."):
                historical_matches, successful_days = get_historical_matches(days_history)
                team_stats = analyze_teams_from_matches(historical_matches)
            
            if team_stats:
                # Separar por categoria
                over_teams = []
                under_teams = []
                balanced_teams = []
                
                for team, stats in team_stats.items():
                    team_info = {
                        'Equipe': team,
                        'Liga': stats['league'],
                        'Taxa Over 0.5': f"{stats['over_rate']:.1%}",
                        'Casa': f"{stats['home_over_rate']:.1%}",
                        'Fora': f"{stats['away_over_rate']:.1%}",
                        'Total Jogos': stats['total_games'],
                        'Jogos Casa': stats['home_games'],
                        'Jogos Fora': stats['away_games'],
                        'Classificação': stats['classification']
                    }
                    
                    if stats['over_rate'] >= 0.60:
                        over_teams.append(team_info)
                    elif stats['over_rate'] <= 0.40:
                        under_teams.append(team_info)
                    else:
                        balanced_teams.append(team_info)
                
                # Mostrar rankings
                if over_teams:
                    st.subheader("🔥 EQUIPES OVER (Recomendadas para Over 0.5)")
                    over_df = pd.DataFrame(over_teams)
                    over_df = over_df.sort_values('Taxa Over 0.5', ascending=False)
                    st.dataframe(over_df, use_container_width=True)
                
                if under_teams:
                    st.subheader("❄️ EQUIPES UNDER (Evitar Over 0.5)")
                    under_df = pd.DataFrame(under_teams)
                    under_df = under_df.sort_values('Taxa Over 0.5', ascending=True)
                    st.dataframe(under_df, use_container_width=True)
                
                if balanced_teams:
                    st.subheader("⚖️ EQUIPES EQUILIBRADAS")
                    balanced_df = pd.DataFrame(balanced_teams)
                    st.dataframe(balanced_df, use_container_width=True)
                
                # Estatísticas gerais
                total_teams = len(team_stats)
                over_count = len(over_teams)
                under_count = len(under_teams)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Equipes", total_teams)
                with col2:
                    st.metric("Equipes OVER", over_count)
                with col3:
                    st.metric("Equipes UNDER", under_count)
                with col4:
                    st.metric("Taxa OVER", f"{(over_count/total_teams*100):.1f}%")
            
            else:
                st.error("❌ Erro ao analisar equipes")
    
    with tab3:
        st.header("📊 Análise por Liga")
        st.info("Esta função será implementada na próxima versão")
        st.write("🔜 Em breve: análise detalhada de cada liga com suas equipes OVER e UNDER")
    
    with tab4:
        st.header("ℹ️ Como Funciona o Sistema")
        
        st.write("""
        ### 🎯 **O que faz o sistema:**
        
        **1. Coleta dados reais** dos últimos 7-30 dias via API
        **2. Analisa cada equipe individualmente:**
        - 🏠 Performance em casa vs ✈️ fora
        - 📊 Taxa de Over 0.5 no primeiro tempo
        - 🎯 Classificação: OVER, UNDER ou Equilibrada
        
        **3. Faz previsões inteligentes:**
        - 🧠 Algoritmo que combina dados das duas equipes
        - ⚖️ Peso maior para equipe da casa (60% vs 40%)
        - 🔥 Bonus para dois times ofensivos
        - ❄️ Penalidade para dois times defensivos
        
        ### 🏆 **Classificações:**
        - 🔥 **EQUIPE OVER FORTE**: ≥75% dos jogos têm Over 0.5 HT
        - 📈 **EQUIPE OVER**: 60-74% dos jogos
        - ⚖️ **EQUILIBRADA**: 40-59% dos jogos  
        - 📉 **EQUIPE UNDER**: 25-39% dos jogos
        - ❄️ **EQUIPE UNDER FORTE**: ≤25% dos jogos
        
        ### 🎯 **Confiança das Previsões:**
        - 🔥 **ALTA**: Probabilidade ≥75% ou ≤25%
        - ⚠️ **MÉDIA**: Probabilidade 55-74% ou 25-45%
        - ❓ **BAIXA**: Probabilidade 45-55%
        
        ### 💡 **Dicas de uso:**
        1. **Foque em confiança ALTA e MÉDIA**
        2. **Prefira equipes com +10 jogos analisados**
        3. **Use diariamente** para dados sempre atualizados
        4. **Combine com sua análise** para melhores resultados
        
        ### 🚀 **Vantagens da sua API PRO:**
        - ✅ Dados em tempo real
        - ✅ Histórico estendido (até 30 dias)
        - ✅ Todas as principais ligas
        - ✅ Sem limitações de uso
        """)

if __name__ == "__main__":
    main()
