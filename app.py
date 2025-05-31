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
    .debug-info {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def test_all_api_formats():
    """Testa TODOS os formatos possíveis da API"""
    headers = {'X-Auth-Token': API_KEY}
    
    # Lista completa de endpoints e formatos para testar
    test_configs = [
        # Teste básico de conexão
        {
            'name': '🔗 Teste Básico - Competições v4',
            'url': 'https://api.football-data.org/v4/competitions',
            'params': {},
            'description': 'Testa se a API Key funciona'
        },
        {
            'name': '🔗 Teste Básico - Competições v2',
            'url': 'https://api.football-data.org/v2/competitions',
            'params': {},
            'description': 'Fallback para API v2'
        },
        
        # Busca de jogos - formato atual
        {
            'name': '📅 Jogos v4 - dateFrom/dateTo',
            'url': 'https://api.football-data.org/v4/matches',
            'params': {'dateFrom': '2025-05-31', 'dateTo': '2025-05-31'},
            'description': 'Formato padrão com intervalo de datas'
        },
        {
            'name': '📅 Jogos v2 - dateFrom/dateTo',
            'url': 'https://api.football-data.org/v2/matches',
            'params': {'dateFrom': '2025-05-31', 'dateTo': '2025-05-31'},
            'description': 'Mesmo formato na API v2'
        },
        
        # Formatos alternativos de data
        {
            'name': '📅 Jogos v4 - date única',
            'url': 'https://api.football-data.org/v4/matches',
            'params': {'date': '2025-05-31'},
            'description': 'Parâmetro date único'
        },
        {
            'name': '📅 Jogos v4 - sem filtro',
            'url': 'https://api.football-data.org/v4/matches',
            'params': {},
            'description': 'Buscar todos os jogos recentes'
        },
        
        # Datas alternativas (fim de semana)
        {
            'name': '📅 Jogos v4 - Sábado anterior',
            'url': 'https://api.football-data.org/v4/matches',
            'params': {'dateFrom': '2025-05-24', 'dateTo': '2025-05-24'},
            'description': 'Sábado anterior (pode ter jogos)'
        },
        {
            'name': '📅 Jogos v4 - Domingo anterior',
            'url': 'https://api.football-data.org/v4/matches',
            'params': {'dateFrom': '2025-05-25', 'dateTo': '2025-05-25'},
            'description': 'Domingo anterior (pode ter jogos)'
        },
        
        # Busca por competições específicas
        {
            'name': '🏆 Premier League - Jogos recentes',
            'url': 'https://api.football-data.org/v4/competitions/PL/matches',
            'params': {},
            'description': 'Jogos da Premier League'
        },
        {
            'name': '🏆 Champions League - Jogos recentes',
            'url': 'https://api.football-data.org/v4/competitions/CL/matches',
            'params': {},
            'description': 'Jogos da Champions League'
        },
        
        # Busca por período maior
        {
            'name': '📊 Última semana',
            'url': 'https://api.football-data.org/v4/matches',
            'params': {'dateFrom': '2025-05-24', 'dateTo': '2025-05-31'},
            'description': 'Buscar jogos da última semana'
        }
    ]
    
    results = []
    working_endpoints = []
    
    for config in test_configs:
        result = {
            'name': config['name'],
            'description': config['description'],
            'url': config['url'],
            'params': str(config['params']),
            'status': 'TESTANDO...',
            'response': '',
            'data_count': 0,
            'error': '',
            'working': False
        }
        
        try:
            response = requests.get(
                config['url'],
                headers=headers,
                params=config['params'],
                timeout=15
            )
            
            result['status'] = f"HTTP {response.status_code}"
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Contar dados retornados
                    if 'competitions' in data:
                        result['data_count'] = len(data['competitions'])
                        result['response'] = f"✅ {result['data_count']} competições"
                    elif 'matches' in data:
                        result['data_count'] = len(data['matches'])
                        result['response'] = f"✅ {result['data_count']} jogos"
                    else:
                        result['response'] = f"✅ Dados recebidos: {list(data.keys())}"
                    
                    result['working'] = True
                    working_endpoints.append({
                        'config': config,
                        'data': data,
                        'count': result['data_count']
                    })
                    
                except Exception as e:
                    result['response'] = f"⚠️ Erro ao processar JSON: {str(e)}"
                    
            elif response.status_code == 400:
                try:
                    error_data = response.json()
                    result['error'] = error_data.get('message', 'Erro 400 sem detalhes')
                except:
                    result['error'] = 'Bad Request - Parâmetros inválidos'
                result['response'] = f"❌ {result['error']}"
                
            elif response.status_code == 403:
                result['response'] = "❌ API Key sem permissão para este endpoint"
                
            elif response.status_code == 404:
                result['response'] = "❌ Endpoint não encontrado"
                
            elif response.status_code == 429:
                result['response'] = "⚠️ Rate limit atingido"
                
            else:
                result['response'] = f"❌ Erro HTTP {response.status_code}"
                
        except requests.exceptions.Timeout:
            result['response'] = "⏰ Timeout - Conexão muito lenta"
            
        except requests.exceptions.ConnectionError:
            result['response'] = "🌐 Erro de conexão"
            
        except Exception as e:
            result['response'] = f"❌ Erro: {str(e)[:100]}"
        
        results.append(result)
        time.sleep(0.5)  # Rate limiting
    
    return results, working_endpoints

def create_advanced_demo():
    """Demo avançado com mais equipes e dados realistas"""
    
    # Dados baseados em temporadas reais
    demo_teams = {
        # Premier League
        'Manchester City': {'league': 'Premier League', 'over_rate': 0.76, 'home_over_rate': 0.83, 'away_over_rate': 0.69, 'total_games': 29, 'classification': '🔥 EQUIPE OVER FORTE'},
        'Liverpool': {'league': 'Premier League', 'over_rate': 0.71, 'home_over_rate': 0.79, 'away_over_rate': 0.63, 'total_games': 28, 'classification': '📈 EQUIPE OVER'},
        'Arsenal': {'league': 'Premier League', 'over_rate': 0.67, 'home_over_rate': 0.73, 'away_over_rate': 0.61, 'total_games': 27, 'classification': '📈 EQUIPE OVER'},
        'Tottenham': {'league': 'Premier League', 'over_rate': 0.69, 'home_over_rate': 0.75, 'away_over_rate': 0.63, 'total_games': 26, 'classification': '📈 EQUIPE OVER'},
        'Chelsea': {'league': 'Premier League', 'over_rate': 0.58, 'home_over_rate': 0.65, 'away_over_rate': 0.51, 'total_games': 25, 'classification': '⚖️ EQUILIBRADA'},
        'Brighton': {'league': 'Premier League', 'over_rate': 0.72, 'home_over_rate': 0.78, 'away_over_rate': 0.66, 'total_games': 24, 'classification': '📈 EQUIPE OVER'},
        'Burnley': {'league': 'Premier League', 'over_rate': 0.31, 'home_over_rate': 0.38, 'away_over_rate': 0.24, 'total_games': 23, 'classification': '📉 EQUIPE UNDER'},
        'Sheffield United': {'league': 'Premier League', 'over_rate': 0.28, 'home_over_rate': 0.35, 'away_over_rate': 0.21, 'total_games': 22, 'classification': '❄️ EQUIPE UNDER FORTE'},
        
        # La Liga
        'Real Madrid': {'league': 'La Liga', 'over_rate': 0.78, 'home_over_rate': 0.85, 'away_over_rate': 0.71, 'total_games': 32, 'classification': '🔥 EQUIPE OVER FORTE'},
        'Barcelona': {'league': 'La Liga', 'over_rate': 0.74, 'home_over_rate': 0.81, 'away_over_rate': 0.67, 'total_games': 31, 'classification': '📈 EQUIPE OVER'},
        'Atletico Madrid': {'league': 'La Liga', 'over_rate': 0.35, 'home_over_rate': 0.42, 'away_over_rate': 0.28, 'total_games': 30, 'classification': '📉 EQUIPE UNDER'},
        'Sevilla': {'league': 'La Liga', 'over_rate': 0.59, 'home_over_rate': 0.66, 'away_over_rate': 0.52, 'total_games': 29, 'classification': '⚖️ EQUILIBRADA'},
        'Real Sociedad': {'league': 'La Liga', 'over_rate': 0.63, 'home_over_rate': 0.70, 'away_over_rate': 0.56, 'total_games': 28, 'classification': '📈 EQUIPE OVER'},
        'Getafe': {'league': 'La Liga', 'over_rate': 0.29, 'home_over_rate': 0.36, 'away_over_rate': 0.22, 'total_games': 27, 'classification': '❄️ EQUIPE UNDER FORTE'},
        
        # Bundesliga
        'Bayern Munich': {'league': 'Bundesliga', 'over_rate': 0.84, 'home_over_rate': 0.91, 'away_over_rate': 0.77, 'total_games': 26, 'classification': '🔥 EQUIPE OVER FORTE'},
        'Borussia Dortmund': {'league': 'Bundesliga', 'over_rate': 0.73, 'home_over_rate': 0.80, 'away_over_rate': 0.66, 'total_games': 25, 'classification': '📈 EQUIPE OVER'},
        'RB Leipzig': {'league': 'Bundesliga', 'over_rate': 0.68, 'home_over_rate': 0.75, 'away_over_rate': 0.61, 'total_games': 24, 'classification': '📈 EQUIPE OVER'},
        'Bayer Leverkusen': {'league': 'Bundesliga', 'over_rate': 0.71, 'home_over_rate': 0.78, 'away_over_rate': 0.64, 'total_games': 23, 'classification': '📈 EQUIPE OVER'},
        
        # Serie A
        'Inter Milan': {'league': 'Serie A', 'over_rate': 0.65, 'home_over_rate': 0.72, 'away_over_rate': 0.58, 'total_games': 30, 'classification': '📈 EQUIPE OVER'},
        'AC Milan': {'league': 'Serie A', 'over_rate': 0.61, 'home_over_rate': 0.68, 'away_over_rate': 0.54, 'total_games': 29, 'classification': '📈 EQUIPE OVER'},
        'Juventus': {'league': 'Serie A', 'over_rate': 0.48, 'home_over_rate': 0.55, 'away_over_rate': 0.41, 'total_games': 28, 'classification': '⚖️ EQUILIBRADA'},
        'Napoli': {'league': 'Serie A', 'over_rate': 0.67, 'home_over_rate': 0.74, 'away_over_rate': 0.60, 'total_games': 27, 'classification': '📈 EQUIPE OVER'},
        
        # Ligue 1
        'PSG': {'league': 'Ligue 1', 'over_rate': 0.79, 'home_over_rate': 0.86, 'away_over_rate': 0.72, 'total_games': 24, 'classification': '🔥 EQUIPE OVER FORTE'},
        'Monaco': {'league': 'Ligue 1', 'over_rate': 0.70, 'home_over_rate': 0.77, 'away_over_rate': 0.63, 'total_games': 23, 'classification': '📈 EQUIPE OVER'},
        'Marseille': {'league': 'Ligue 1', 'over_rate': 0.66, 'home_over_rate': 0.73, 'away_over_rate': 0.59, 'total_games': 22, 'classification': '📈 EQUIPE OVER'},
    }
    
    # Jogos de exemplo para diferentes cenários
    demo_matches = [
        # Jogos de alta probabilidade Over
        {'home': 'Manchester City', 'away': 'Liverpool', 'league': 'Premier League', 'time': '15:30'},
        {'home': 'Real Madrid', 'away': 'Barcelona', 'league': 'La Liga', 'time': '21:00'},
        {'home': 'Bayern Munich', 'away': 'Borussia Dortmund', 'league': 'Bundesliga', 'time': '18:30'},
        {'home': 'PSG', 'away': 'Monaco', 'league': 'Ligue 1', 'time': '20:45'},
        
        # Jogos equilibrados
        {'home': 'Arsenal', 'away': 'Chelsea', 'league': 'Premier League', 'time': '17:30'},
        {'home': 'Inter Milan', 'away': 'AC Milan', 'league': 'Serie A', 'time': '20:45'},
        
        # Jogos com tendência Under
        {'home': 'Atletico Madrid', 'away': 'Getafe', 'league': 'La Liga', 'time': '16:15'},
        {'home': 'Burnley', 'away': 'Sheffield United', 'league': 'Premier League', 'time': '15:00'},
    ]
    
    return demo_teams, demo_matches

def predict_advanced_matches(demo_teams, demo_matches):
    """Sistema de predição avançado"""
    predictions = []
    
    for match in demo_matches:
        home_team = match['home']
        away_team = match['away']
        league = match['league']
        match_time = match['time']
        
        home_stats = demo_teams[home_team]
        away_stats = demo_teams[away_team]
        
        # Algoritmo avançado de predição
        home_rate = home_stats['home_over_rate']
        away_rate = away_stats['away_over_rate']
        
        # Peso diferente para casa/fora
        base_probability = (home_rate * 0.65) + (away_rate * 0.35)
        
        # Ajustes por contexto
        if home_stats['over_rate'] > 0.75 and away_stats['over_rate'] > 0.75:
            base_probability += 0.08  # Bonus grande para dois times ofensivos
        elif home_stats['over_rate'] > 0.65 and away_stats['over_rate'] > 0.65:
            base_probability += 0.04  # Bonus médio
        elif home_stats['over_rate'] < 0.35 and away_stats['over_rate'] < 0.35:
            base_probability -= 0.08  # Penalidade dois times defensivos
        elif home_stats['over_rate'] < 0.45 and away_stats['over_rate'] < 0.45:
            base_probability -= 0.04  # Penalidade média
        
        # Garantir que está entre 0 e 1
        final_probability = max(0.05, min(0.95, base_probability))
        
        # Classificação da previsão
        if final_probability >= 0.75:
            prediction = "✅ OVER 0.5"
            confidence = "MUITO ALTA"
            bet_class = "team-over"
            confidence_icon = "🔥"
        elif final_probability >= 0.65:
            prediction = "✅ OVER 0.5"
            confidence = "ALTA"
            bet_class = "team-over"
            confidence_icon = "🎯"
        elif final_probability >= 0.55:
            prediction = "✅ OVER 0.5"
            confidence = "MÉDIA"
            bet_class = "team-over"
            confidence_icon = "📈"
        elif final_probability <= 0.25:
            prediction = "❌ UNDER 0.5"
            confidence = "MUITO ALTA"
            bet_class = "team-under"
            confidence_icon = "🧊"
        elif final_probability <= 0.35:
            prediction = "❌ UNDER 0.5"
            confidence = "ALTA"
            bet_class = "team-under"
            confidence_icon = "❄️"
        elif final_probability <= 0.45:
            prediction = "❌ UNDER 0.5"
            confidence = "MÉDIA"
            bet_class = "team-under"
            confidence_icon = "📉"
        else:
            prediction = "⚖️ INDEFINIDO"
            confidence = "BAIXA"
            bet_class = "team-balanced"
            confidence_icon = "❓"
        
        predictions.append({
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
            'time': match_time,
            'prediction': prediction,
            'confidence': confidence,
            'confidence_icon': confidence_icon,
            'probability': f"{final_probability:.1%}",
            'home_rate': f"{home_rate:.1%}",
            'away_rate': f"{away_rate:.1%}",
            'home_class': home_stats['classification'],
            'away_class': away_stats['classification'],
            'home_games': home_stats['total_games'],
            'away_games': away_stats['total_games'],
            'bet_class': bet_class,
            'sort_priority': final_probability if final_probability >= 0.55 else (1 - final_probability) if final_probability <= 0.45 else 0
        })
    
    # Ordenar por prioridade (melhores apostas primeiro)
    predictions.sort(key=lambda x: x['sort_priority'], reverse=True)
    return predictions

def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>⚽ Over 0.5 HT - Sistema Inteligente</h1>
        <p>Análise por equipes com diagnóstico avançado</p>
        <span class="premium-badge">API PRO ✨</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Configurações")
    st.sidebar.info("🔑 API Key configurada automaticamente")
    
    # Data
    selected_date = st.sidebar.date_input(
        "📅 Data para análise:",
        value=datetime.now().date()
    )
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔧 DIAGNÓSTICO COMPLETO", 
        "📊 DEMO AVANÇADO", 
        "🎯 TESTE REAL",
        "💡 SOLUÇÕES"
    ])
    
    with tab1:
        st.header("🔧 Diagnóstico Completo da API")
        st.info("Este teste verificará TODOS os formatos possíveis da API")
        
        if st.button("🚀 EXECUTAR DIAGNÓSTICO COMPLETO", type="primary"):
            with st.spinner("Testando todos os formatos da API..."):
                results, working_endpoints = test_all_api_formats()
            
            st.subheader("📊 Resultados dos Testes")
            
            # Mostrar resultados
            success_count = sum(1 for r in results if r['working'])
            total_count = len(results)
            
            st.metric("🎯 Endpoints Funcionando", f"{success_count}/{total_count}")
            
            # Resultados detalhados
            for result in results:
                if result['working']:
                    st.markdown(f'<div class="debug-success"><strong>{result["name"]}</strong><br>{result["description"]}<br><strong>Status:</strong> {result["status"]}<br><strong>Resultado:</strong> {result["response"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="debug-error"><strong>{result["name"]}</strong><br>{result["description"]}<br><strong>Status:</strong> {result["status"]}<br><strong>Erro:</strong> {result["response"]}</div>', unsafe_allow_html=True)
            
            # Se encontrou dados
            if working_endpoints:
                st.subheader("✅ Dados Encontrados")
                
                for endpoint in working_endpoints:
                    if endpoint['count'] > 0:
                        st.success(f"🎯 {endpoint['config']['name']}: {endpoint['count']} itens encontrados")
                        
                        # Mostrar sample dos dados
                        if 'matches' in endpoint['data']:
                            matches = endpoint['data']['matches'][:3]
                            for match in matches:
                                status = match.get('status', 'N/A')
                                home = match.get('homeTeam', {}).get('name', 'N/A')
                                away = match.get('awayTeam', {}).get('name', 'N/A')
                                competition = match.get('competition', {}).get('name', 'N/A')
                                st.write(f"⚽ {home} vs {away} ({competition}) - Status: {status}")
            
            else:
                st.error("❌ Nenhum endpoint retornou dados úteis")
    
    with tab2:
        st.header("📊 DEMO Avançado - Sistema Completo")
        st.info("🎯 Demonstração com dados realísticos de múltiplas ligas")
        
        if st.button("🚀 VER SISTEMA AVANÇADO EM AÇÃO"):
            demo_teams, demo_matches = create_advanced_demo()
            predictions = predict_advanced_matches(demo_teams, demo_matches)
            
            # Mostrar previsões por categoria
            st.subheader("🏆 MELHORES APOSTAS DO DIA")
            
            # Apostas de alta confiança
            high_confidence = [p for p in predictions if p['confidence'] in ['MUITO ALTA', 'ALTA']]
            
            if high_confidence:
                for bet in high_confidence:
                    st.markdown(f"""
                    <div class="{bet['bet_class']}">
                        <h4>{bet['confidence_icon']} {bet['home_team']} vs {bet['away_team']} - {bet['time']}</h4>
                        <p><strong>Liga:</strong> {bet['league']}</p>
                        <p><strong>Previsão:</strong> {bet['prediction']} ({bet['probability']})</p>
                        <p><strong>Confiança:</strong> {bet['confidence']}</p>
                        <hr style="border-color: rgba(255,255,255,0.3);">
                        <p><strong>🏠 {bet['home_team']}:</strong> {bet['home_class']} - Casa: {bet['home_rate']} ({bet['home_games']} jogos)</p>
                        <p><strong>✈️ {bet['away_team']}:</strong> {bet['away_class']} - Fora: {bet['away_rate']} ({bet['away_games']} jogos)</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Resumo por liga
            st.subheader("📈 Análise por Liga")
            
            leagues = {}
            for team, stats in demo_teams.items():
                league = stats['league']
                if league not in leagues:
                    leagues[league] = {'teams': [], 'avg_over': 0}
                leagues[league]['teams'].append(stats['over_rate'])
            
            for league, data in leagues.items():
                avg_over = sum(data['teams']) / len(data['teams'])
                team_count = len(data['teams'])
                
                league_type = "🔥 LIGA OVER" if avg_over > 0.65 else "❄️ LIGA UNDER" if avg_over < 0.45 else "⚖️ LIGA EQUILIBRADA"
                
                st.write(f"🏆 **{league}**: {league_type} - Média: {avg_over:.1%} ({team_count} equipes)")
            
            # Ranking completo
            st.subheader("🏆 Ranking Completo das Equipes")
            
            team_list = []
            for team, stats in demo_teams.items():
                team_list.append({
                    'Equipe': team,
                    'Liga': stats['league'],
                    'Taxa Over Geral': f"{stats['over_rate']:.1%}",
                    'Taxa Casa': f"{stats['home_over_rate']:.1%}",
                    'Taxa Fora': f"{stats['away_over_rate']:.1%}",
                    'Total Jogos': stats['total_games'],
                    'Classificação': stats['classification']
                })
            
            df = pd.DataFrame(team_list)
            df = df.sort_values('Taxa Over Geral', ascending=False)
            st.dataframe(df, use_container_width=True, height=400)
    
    with tab3:
        st.header("🎯 Teste com Dados Reais")
        
        st.info("Use esta aba quando o diagnóstico encontrar endpoints funcionando")
        
        if st.button("🔍 BUSCAR DADOS REAIS"):
            st.warning("💡 Execute primeiro o DIAGNÓSTICO COMPLETO para identificar endpoints funcionando")
    
    with tab4:
        st.header("💡 Análise e Soluções")
        
        st.write("""
        ### 🔍 **Diagnóstico do Problema:**
        
        **Erro HTTP 400** geralmente significa:
        1. 🗓️ **Formato de data incorreto** para a API
        2. 📅 **Parâmetros não suportados** pelo endpoint
        3. 🔧 **Versão da API** incompatível
        4. ⚡ **Rate limiting** muito agressivo
        
        ### 🛠️ **O que o diagnóstico vai mostrar:**
        
        - ✅ **Endpoints funcionando** (se houver)
        - ❌ **Endpoints com erro** e detalhes
        - 📊 **Dados disponíveis** (competições, jogos)
        - 🎯 **Formato correto** para sua API
        
        ### 📋 **Próximos passos:**
        
        **1. Execute o DIAGNÓSTICO COMPLETO**
        - Testa 11 formatos diferentes
        - Identifica o que funciona
        - Mostra dados disponíveis
        
        **2. Se nenhum endpoint funcionar:**
        - 🔑 Problema com a API Key
        - 📞 Contatar suporte da football-data.org
        - 🔄 Verificar status da conta
        
        **3. Se alguns endpoints funcionarem:**
        - ✅ Usar o formato que funciona
        - 🎯 Adaptar o sistema
        - 📊 Analisar dados disponíveis
        
        ### 🎯 **O DEMO mostra exatamente como será quando funcionar:**
        - 🏆 Previsões inteligentes
        - 📈 Análise por equipes
        - 🎲 Sistema de confiança
        - 📊 Rankings detalhados
        
        **🚀 Execute o diagnóstico para descobrir o que está acontecendo!**
        """)

if __name__ == "__main__":
    main()
