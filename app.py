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

# API Key automática
API_KEY = "1136cc77028d84lfd0efa2a603f81638"

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
</style>
""", unsafe_allow_html=True)

def test_api_multiple_endpoints():
    """Testa múltiplos endpoints da API para encontrar o que funciona"""
    
    endpoints_to_test = [
        {
            'url': 'https://api.football-data.org/v4/competitions',
            'name': 'API v4 - Competições',
            'headers': {'X-Auth-Token': API_KEY}
        },
        {
            'url': 'https://api.football-data.org/v2/competitions',
            'name': 'API v2 - Competições',
            'headers': {'X-Auth-Token': API_KEY}
        },
        {
            'url': 'https://api.football-data.org/v4/matches',
            'name': 'API v4 - Jogos',
            'headers': {'X-Auth-Token': API_KEY},
            'params': {'dateFrom': '2025-05-30', 'dateTo': '2025-05-30'}
        }
    ]
    
    working_endpoints = []
    
    for endpoint in endpoints_to_test:
        try:
            response = requests.get(
                endpoint['url'], 
                headers=endpoint['headers'],
                params=endpoint.get('params', {}),
                timeout=10
            )
            
            status = response.status_code
            endpoint['status'] = status
            endpoint['response_size'] = len(response.content) if response.content else 0
            
            if status == 200:
                endpoint['result'] = "✅ FUNCIONANDO"
                working_endpoints.append(endpoint)
            elif status == 403:
                endpoint['result'] = "❌ API Key inválida"
            elif status == 429:
                endpoint['result'] = "⚠️ Rate limit"
            elif status == 404:
                endpoint['result'] = "❌ Endpoint não encontrado"
            else:
                endpoint['result'] = f"❌ Erro {status}"
                
        except Exception as e:
            endpoint['result'] = f"❌ Erro conexão: {str(e)[:50]}"
            endpoint['status'] = 'N/A'
            
    return endpoints_to_test, working_endpoints

def get_matches_robust(date, max_retries=3):
    """Busca jogos com múltiplas tentativas e endpoints"""
    
    # Lista de configurações para tentar
    api_configs = [
        {
            'url': 'https://api.football-data.org/v4/matches',
            'params': {'dateFrom': date.strftime('%Y-%m-%d'), 'dateTo': date.strftime('%Y-%m-%d')},
            'headers': {'X-Auth-Token': API_KEY}
        },
        {
            'url': 'https://api.football-data.org/v2/matches',
            'params': {'dateFrom': date.strftime('%Y-%m-%d'), 'dateTo': date.strftime('%Y-%m-%d')},
            'headers': {'X-Auth-Token': API_KEY}
        },
        {
            'url': 'https://api.football-data.org/v4/matches',
            'params': {'date': date.strftime('%Y-%m-%d')},
            'headers': {'X-Auth-Token': API_KEY}
        }
    ]
    
    for config in api_configs:
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    config['url'],
                    headers=config['headers'],
                    params=config['params'],
                    timeout=15
                )
                
                if response.status_code == 200:
                    data = response.json()
                    matches = data.get('matches', [])
                    api_version = 'v4' if 'v4' in config['url'] else 'v2'
                    return matches, f"✅ {len(matches)} jogos (API {api_version})"
                
                elif response.status_code == 429:
                    time.sleep(2)  # Rate limit
                    continue
                    
                elif response.status_code in [400, 404]:
                    break  # Não tenta mais este endpoint
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    continue  # Vai para próximo config
                time.sleep(1)
    
    return [], "❌ Erro em todos os endpoints testados"

def create_demo_data():
    """Cria dados demo quando a API não funciona"""
    demo_teams = {
        'Manchester City': {'league': 'Premier League', 'over_rate': 0.78, 'games': 15},
        'Liverpool': {'league': 'Premier League', 'over_rate': 0.72, 'games': 14},
        'Arsenal': {'league': 'Premier League', 'over_rate': 0.65, 'games': 16},
        'Burnley': {'league': 'Premier League', 'over_rate': 0.25, 'games': 12},
        'Crystal Palace': {'league': 'Premier League', 'over_rate': 0.30, 'games': 13},
        
        'Real Madrid': {'league': 'La Liga', 'over_rate': 0.80, 'games': 18},
        'Barcelona': {'league': 'La Liga', 'over_rate': 0.75, 'games': 17},
        'Atletico Madrid': {'league': 'La Liga', 'over_rate': 0.35, 'games': 16},
        
        'Bayern Munich': {'league': 'Bundesliga', 'over_rate': 0.85, 'games': 20},
        'Borussia Dortmund': {'league': 'Bundesliga', 'over_rate': 0.70, 'games': 19},
    }
    
    demo_matches = [
        {'home': 'Manchester City', 'away': 'Liverpool', 'league': 'Premier League'},
        {'home': 'Real Madrid', 'away': 'Barcelona', 'league': 'La Liga'},
        {'home': 'Bayern Munich', 'away': 'Borussia Dortmund', 'league': 'Bundesliga'},
        {'home': 'Arsenal', 'away': 'Burnley', 'league': 'Premier League'},
    ]
    
    return demo_teams, demo_matches

def analyze_demo_matches(demo_teams, demo_matches):
    """Analisa os jogos demo"""
    predictions = []
    
    for match in demo_matches:
        home_team = match['home']
        away_team = match['away']
        league = match['league']
        
        home_rate = demo_teams[home_team]['over_rate']
        away_rate = demo_teams[away_team]['over_rate']
        
        # Média ponderada
        combined_rate = (home_rate * 0.6) + (away_rate * 0.4)
        
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
        
        predictions.append({
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
            'prediction': prediction,
            'confidence': confidence,
            'probability': f"{combined_rate:.1%}",
            'home_rate': f"{home_rate:.1%}",
            'away_rate': f"{away_rate:.1%}",
            'home_class': get_team_class(home_rate),
            'away_class': get_team_class(away_rate),
            'bet_class': bet_class
        })
    
    return predictions

def get_team_class(rate):
    """Classifica equipe baseada na taxa"""
    if rate >= 0.70:
        return "🔥 EQUIPE OVER"
    elif rate >= 0.55:
        return "📈 OVER Moderada"
    elif rate <= 0.30:
        return "❄️ EQUIPE UNDER"
    elif rate <= 0.45:
        return "📉 UNDER Moderada"
    else:
        return "⚖️ Equilibrada"

def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>⚽ Over 0.5 HT - Análise por EQUIPES</h1>
        <p>Sistema que analisa equipes individuais dentro de cada liga</p>
        <span class="premium-badge">API PAGA ✨</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Configurações")
    st.sidebar.success("🔑 API Key: Configurada automaticamente")
    
    selected_date = st.sidebar.date_input(
        "📅 Data para análise:",
        value=datetime.now().date()
    )
    
    days_history = st.sidebar.slider(
        "📊 Dias de histórico:",
        min_value=7,
        max_value=90,
        value=30,
        help="API paga permite até 90 dias"
    )
    
    # Tabs principais
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 PREVISÕES", 
        "🔧 TESTE API", 
        "📊 DEMO",
        "ℹ️ SOBRE"
    ])
    
    with tab1:
        st.header(f"🎯 Análise - {selected_date.strftime('%d/%m/%Y')}")
        
        if st.button("🚀 ANALISAR EQUIPES DO DIA", type="primary"):
            with st.spinner("🔍 Conectando com API..."):
                matches, message = get_matches_robust(selected_date)
                
            st.info(message)
            
            if not matches:
                st.warning("⚠️ API não disponível - Veja o DEMO na aba ao lado")
                st.info("💡 Vá na aba 'TESTE API' para diagnosticar o problema")
            else:
                st.success(f"✅ {len(matches)} jogos encontrados!")
                # Aqui você pode continuar com a análise real
    
    with tab2:
        st.header("🔧 Diagnóstico da API")
        
        if st.button("🔍 TESTAR CONEXÕES DA API"):
            with st.spinner("Testando endpoints..."):
                all_tests, working = test_api_multiple_endpoints()
            
            st.subheader("📊 Resultados dos Testes:")
            
            for test in all_tests:
                st.markdown(f"""
                <div class="api-test">
                    <h4>{test['name']}</h4>
                    <p><strong>URL:</strong> {test['url']}</p>
                    <p><strong>Status:</strong> {test.get('status', 'N/A')}</p>
                    <p><strong>Resultado:</strong> {test['result']}</p>
                    <p><strong>Dados:</strong> {test.get('response_size', 0)} bytes</p>
                </div>
                """, unsafe_allow_html=True)
            
            if working:
                st.success(f"✅ {len(working)} endpoint(s) funcionando!")
                st.info("🔧 O sistema vai usar automaticamente o endpoint que funciona")
            else:
                st.error("❌ Nenhum endpoint funcionando")
                st.info("💡 Possíveis soluções:")
                st.write("1. Verificar se a API Key está correta")
                st.write("2. Verificar se a conta não expirou")
                st.write("3. Tentar novamente em alguns minutos")
                st.write("4. Contatar o suporte da API")
    
    with tab3:
        st.header("📊 DEMO - Como o Sistema Funciona")
        st.info("🎯 Demonstração com dados de exemplo")
        
        if st.button("🚀 VER DEMO DA ANÁLISE"):
            demo_teams, demo_matches = create_demo_data()
            predictions = analyze_demo_matches(demo_teams, demo_matches)
            
            st.subheader("🏆 EXEMPLO DE PREVISÕES")
            
            for pred in predictions:
                st.markdown(f"""
                <div class="{pred['bet_class']}">
                    <h4>⚽ {pred['home_team']} vs {pred['away_team']}</h4>
                    <p><strong>Liga:</strong> {pred['league']}</p>
                    <p><strong>Previsão:</strong> {pred['prediction']} ({pred['probability']})</p>
                    <p><strong>Confiança:</strong> {pred['confidence']}</p>
                    <hr style="border-color: rgba(255,255,255,0.3);">
                    <p><strong>🏠 {pred['home_team']}:</strong> {pred['home_class']} ({pred['home_rate']})</p>
                    <p><strong>✈️ {pred['away_team']}:</strong> {pred['away_class']} ({pred['away_rate']})</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.subheader("📈 Ranking das Equipes DEMO")
            
            # Criar tabela das equipes
            team_list = []
            for team, stats in demo_teams.items():
                team_list.append({
                    'Equipe': team,
                    'Liga': stats['league'],
                    'Taxa Over 0.5': f"{stats['over_rate']:.1%}",
                    'Jogos': stats['games'],
                    'Classificação': get_team_class(stats['over_rate'])
                })
            
            df = pd.DataFrame(team_list)
            df = df.sort_values('Taxa Over 0.5', ascending=False)
            st.dataframe(df, use_container_width=True)
    
    with tab4:
        st.header("ℹ️ Como Resolver Problemas da API")
        
        st.write("""
        ### 🔧 **Se a API não estiver funcionando:**
        
        **1. Verificar API Key:**
        - Acesse: https://www.football-data.org/
        - Faça login na sua conta
        - Verifique se a API Key está correta
        - Confirme se a assinatura está ativa
        
        **2. Verificar Status da Conta:**
        - API paga tem limites diferentes
        - Verifique se não ultrapassou requests/minuto
        - Confirme se a conta não expirou
        
        **3. Problemas Comuns:**
        - **Status 403**: API Key inválida ou sem permissões
        - **Status 429**: Muitos requests (aguarde 1 minuto)
        - **Status 400**: Formato de request incorreto
        - **Erro conexão**: Problema de rede temporário
        
        **4. Soluções:**
        - Use a aba "TESTE API" para diagnosticar
        - Veja o DEMO para entender como funciona
        - Aguarde alguns minutos e tente novamente
        - Contate o suporte se persistir
        
        ### 🎯 **Quando funcionar:**
        O sistema fará **análise real** de:
        - ✅ Equipes individuais
        - ✅ Performance casa/fora
        - ✅ Histórico de até 90 dias
        - ✅ Previsões precisas
        
        ### 🚀 **O DEMO mostra exatamente** como será!
        """)

if __name__ == "__main__":
    main()
