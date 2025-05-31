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

# CSS simples
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
    .bet-card {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        margin: 1rem 0;
    }
    .success-bet {
        background: #f0fff4;
        border-left: 4px solid #28a745;
    }
    .warning-bet {
        background: #fffbf0;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

def get_matches_from_api(api_key, date):
    """Busca jogos da API"""
    url = "https://api.football-data.org/v4/matches"
    headers = {'X-Auth-Token': api_key}
    date_str = date.strftime('%Y-%m-%d')
    params = {'dateFrom': date_str, 'dateTo': date_str}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json().get('matches', [])
        else:
            st.error(f"Erro na API: Status {response.status_code}")
            if response.status_code == 403:
                st.error("❌ API Key inválida ou sem permissões")
            return []
    except Exception as e:
        st.error(f"Erro de conexão: {str(e)}")
        return []

def get_historical_matches(api_key, days=7):
    """Busca jogos históricos"""
    all_matches = []
    end_date = datetime.now()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(days):
        date = end_date - timedelta(days=i+1)
        status_text.text(f"Buscando dados: {date.strftime('%Y-%m-%d')} ({i+1}/{days})")
        
        matches = get_matches_from_api(api_key, date)
        all_matches.extend(matches)
        
        progress_bar.progress((i+1)/days)
        time.sleep(0.2)  # Respeitar rate limit
    
    progress_bar.empty()
    status_text.empty()
    
    return all_matches

def analyze_leagues(matches):
    """Analisa padrões das ligas"""
    league_data = {}
    
    for match in matches:
        if (match.get('status') == 'FINISHED' and 
            match.get('score', {}).get('halfTime')):
            
            league = match['competition']['name']
            ht_score = match['score']['halfTime']
            
            if ht_score['home'] is not None and ht_score['away'] is not None:
                total_ht_goals = ht_score['home'] + ht_score['away']
                
                if league not in league_data:
                    league_data[league] = {'games': 0, 'over_05': 0, 'total_goals': 0}
                
                league_data[league]['games'] += 1
                league_data[league]['total_goals'] += total_ht_goals
                
                if total_ht_goals > 0.5:
                    league_data[league]['over_05'] += 1
    
    # Calcular estatísticas
    league_stats = {}
    for league, data in league_data.items():
        if data['games'] >= 3:  # Mínimo 3 jogos
            over_rate = data['over_05'] / data['games']
            avg_goals = data['total_goals'] / data['games']
            
            league_stats[league] = {
                'over_rate': over_rate,
                'avg_goals_ht': avg_goals,
                'total_games': data['games'],
                'classification': get_league_classification(over_rate)
            }
    
    return league_stats

def get_league_classification(over_rate):
    """Classifica liga baseada na taxa de Over 0.5"""
    if over_rate >= 0.70:
        return "🔥 OVER Liga"
    elif over_rate >= 0.55:
        return "📈 OVER Moderada"
    elif over_rate <= 0.30:
        return "❄️ UNDER Liga"
    elif over_rate <= 0.45:
        return "📉 UNDER Moderada"
    else:
        return "⚖️ Equilibrada"

def predict_matches(today_matches, league_stats):
    """Faz previsões para jogos de hoje"""
    predictions = []
    
    for match in today_matches:
        if match.get('status') in ['SCHEDULED', 'TIMED']:
            league = match['competition']['name']
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            
            if league in league_stats:
                stats = league_stats[league]
                over_rate = stats['over_rate']
                classification = stats['classification']
                
                # Determinar previsão
                if over_rate >= 0.65:
                    prediction = "✅ OVER 0.5"
                    confidence = "ALTA" if over_rate >= 0.75 else "MÉDIA"
                    bet_class = "success-bet"
                elif over_rate >= 0.55:
                    prediction = "📊 OVER 0.5"
                    confidence = "MÉDIA"
                    bet_class = "warning-bet"
                else:
                    prediction = "❌ UNDER 0.5"
                    confidence = "BAIXA"
                    bet_class = "bet-card"
                
                predictions.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'league': league,
                    'prediction': prediction,
                    'confidence': confidence,
                    'over_rate': f"{over_rate:.1%}",
                    'classification': classification,
                    'total_games': stats['total_games'],
                    'bet_class': bet_class,
                    'sort_priority': over_rate if over_rate >= 0.55 else 0
                })
            else:
                predictions.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'league': league,
                    'prediction': "❓ SEM DADOS",
                    'confidence': "N/A",
                    'over_rate': "N/A",
                    'classification': "Sem histórico",
                    'total_games': 0,
                    'bet_class': "bet-card",
                    'sort_priority': 0
                })
    
    # Ordenar por prioridade (melhores apostas primeiro)
    predictions.sort(key=lambda x: x['sort_priority'], reverse=True)
    
    return predictions

def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>⚽ Over 0.5 HT - Análise Inteligente por Liga</h1>
        <p>Sistema que identifica automaticamente ligas OVER e UNDER</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Configurações")
    
    # API Key
    api_key = st.sidebar.text_input(
        "🔑 Sua API Key:",
        type="password",
        placeholder="Cole sua API key aqui"
    )
    
    # Data
    selected_date = st.sidebar.date_input(
        "📅 Data para análise:",
        value=datetime.now().date()
    )
    
    # Histórico
    days_history = st.sidebar.slider(
        "📊 Dias de histórico:",
        min_value=3,
        max_value=14,
        value=7,
        help="Mais dias = análise mais precisa, mas demora mais"
    )
    
    if not api_key:
        st.info("👆 **Insira sua API Key na sidebar para começar!**")
        st.write("### 🔑 Como obter sua API Key:")
        st.write("1. Acesse: https://www.football-data.org/")
        st.write("2. Crie uma conta gratuita")
        st.write("3. Copie sua API Key e cole na sidebar")
        return
    
    # Tabs principais
    tab1, tab2, tab3 = st.tabs(["🎯 PREVISÕES DO DIA", "📊 ANÁLISE DAS LIGAS", "ℹ️ COMO USAR"])
    
    with tab1:
        st.header(f"🎯 Análise para {selected_date.strftime('%d/%m/%Y')}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            analyze_button = st.button(
                "🚀 ANALISAR JOGOS DO DIA",
                type="primary",
                use_container_width=True
            )
        
        with col2:
            st.write(f"📊 Usando {days_history} dias de histórico")
        
        if analyze_button:
            with st.spinner("🔍 Buscando jogos de hoje..."):
                today_matches = get_matches_from_api(api_key, selected_date)
            
            if not today_matches:
                st.warning("❌ Nenhum jogo encontrado para esta data")
                return
            
            upcoming_matches = [m for m in today_matches if m.get('status') in ['SCHEDULED', 'TIMED']]
            
            if not upcoming_matches:
                st.info("⏰ Nenhum jogo programado para hoje (apenas jogos já finalizados)")
                return
            
            st.success(f"✅ {len(upcoming_matches)} jogos encontrados!")
            
            with st.spinner(f"📈 Analisando padrões dos últimos {days_history} dias..."):
                historical_matches = get_historical_matches(api_key, days_history)
            
            if not historical_matches:
                st.error("❌ Erro ao buscar dados históricos")
                return
            
            # Analisar ligas
            league_stats = analyze_leagues(historical_matches)
            
            if not league_stats:
                st.warning("⚠️ Dados insuficientes para análise")
                return
            
            # Gerar previsões
            predictions = predict_matches(upcoming_matches, league_stats)
            
            # Mostrar resultados
            st.header("🏆 MELHORES APOSTAS DO DIA")
            
            best_bets = [p for p in predictions if p['sort_priority'] >= 0.55]
            
            if best_bets:
                for bet in best_bets:
                    st.markdown(f"""
                    <div class="{bet['bet_class']}">
                        <h4>⚽ {bet['home_team']} vs {bet['away_team']}</h4>
                        <p><strong>Liga:</strong> {bet['league']} ({bet['classification']})</p>
                        <p><strong>Previsão:</strong> {bet['prediction']}</p>
                        <p><strong>Confiança:</strong> {bet['confidence']}</p>
                        <p><strong>Taxa Over 0.5 da Liga:</strong> {bet['over_rate']} ({bet['total_games']} jogos)</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("🤔 Nenhuma aposta de boa confiança encontrada hoje")
            
            # Todas as previsões
            with st.expander("📋 TODAS AS PREVISÕES"):
                for pred in predictions:
                    st.write(f"**{pred['home_team']} vs {pred['away_team']}**")
                    st.write(f"Liga: {pred['league']} | Previsão: {pred['prediction']} | Taxa: {pred['over_rate']}")
                    st.write("---")
    
    with tab2:
        st.header("📊 Classificação das Ligas")
        
        if st.button("🔍 Analisar Todas as Ligas"):
            with st.spinner("Buscando dados..."):
                historical_matches = get_historical_matches(api_key, days_history)
                league_stats = analyze_leagues(historical_matches)
            
            if league_stats:
                # Separar por tipo
                over_leagues = []
                under_leagues = []
                balanced_leagues = []
                
                for league, stats in league_stats.items():
                    league_info = {
                        'Liga': league,
                        'Taxa Over 0.5': f"{stats['over_rate']:.1%}",
                        'Média Gols HT': f"{stats['avg_goals_ht']:.2f}",
                        'Total Jogos': stats['total_games'],
                        'Classificação': stats['classification']
                    }
                    
                    if stats['over_rate'] >= 0.55:
                        over_leagues.append(league_info)
                    elif stats['over_rate'] <= 0.45:
                        under_leagues.append(league_info)
                    else:
                        balanced_leagues.append(league_info)
                
                # Mostrar resultados
                if over_leagues:
                    st.subheader("🔥 LIGAS OVER (Recomendadas)")
                    st.dataframe(pd.DataFrame(over_leagues), use_container_width=True)
                
                if under_leagues:
                    st.subheader("❄️ LIGAS UNDER (Evitar)")
                    st.dataframe(pd.DataFrame(under_leagues), use_container_width=True)
                
                if balanced_leagues:
                    st.subheader("⚖️ LIGAS EQUILIBRADAS")
                    st.dataframe(pd.DataFrame(balanced_leagues), use_container_width=True)
    
    with tab3:
        st.header("ℹ️ Como Usar o Sistema")
        
        st.write("""
        ### 🎯 **Objetivo:**
        Identificar automaticamente as melhores apostas **Over 0.5 no 1º Tempo** baseado em padrões das ligas.
        
        ### 📊 **Como funciona:**
        1. **Análise por Liga**: Cada liga é analisada separadamente
        2. **Padrões Históricos**: Últimos 7-14 dias de dados
        3. **Classificação Automática**:
           - 🔥 **OVER Liga** (≥70% dos jogos têm Over 0.5 HT)
           - 📈 **OVER Moderada** (55-69%)
           - ⚖️ **Equilibrada** (45-54%)
           - 📉 **UNDER Moderada** (30-44%)
           - ❄️ **UNDER Liga** (≤30%)
        
        ### 🚀 **Passo a passo:**
        1. Cole sua API Key na sidebar
        2. Escolha a data e dias de histórico
        3. Clique em "ANALISAR JOGOS DO DIA"
        4. Veja as melhores apostas identificadas
        
        ### 💡 **Dicas:**
        - Foque em apostas de **CONFIANÇA ALTA**
        - Ligas **🔥 OVER** são as melhores para Over 0.5
        - Evite ligas **❄️ UNDER**
        - Mais dias de histórico = análise mais precisa
        
        ### 🔄 **Atualize diariamente** para sempre ter os melhores insights!
        """)

if __name__ == "__main__":
    main()
