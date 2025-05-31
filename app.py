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
    .api-key-input {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #007bff;
        margin: 1rem 0;
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
</style>
""", unsafe_allow_html=True)

def test_api_key(api_key):
    """Testa se a API key funciona"""
    if not api_key or len(api_key.strip()) < 10:
        return False, "API Key muito curta ou vazia"
    
    api_key = api_key.strip()
    headers = {'X-Auth-Token': api_key}
    
    try:
        # Teste básico
        response = requests.get(
            'https://api.football-data.org/v4/competitions',
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            competitions = data.get('competitions', [])
            return True, f"✅ API funcionando! {len(competitions)} competições disponíveis"
        
        elif response.status_code == 403:
            return False, "❌ API Key inválida ou sem permissões"
        
        elif response.status_code == 400:
            try:
                error_data = response.json()
                return False, f"❌ Erro 400: {error_data.get('message', 'Request inválido')}"
            except:
                return False, "❌ Erro 400: Formato de request inválido"
        
        elif response.status_code == 429:
            return False, "⚠️ Rate limit atingido - aguarde alguns minutos"
        
        else:
            return False, f"❌ Erro HTTP {response.status_code}"
            
    except requests.exceptions.Timeout:
        return False, "⏰ Timeout - conexão muito lenta"
    
    except requests.exceptions.ConnectionError:
        return False, "🌐 Erro de conexão - verifique sua internet"
    
    except Exception as e:
        return False, f"❌ Erro: {str(e)}"

def get_matches_with_key(api_key, date):
    """Busca jogos com a API key fornecida"""
    headers = {'X-Auth-Token': api_key.strip()}
    
    date_str = date.strftime('%Y-%m-%d')
    
    # Múltiplos formatos para testar
    endpoints = [
        {
            'url': 'https://api.football-data.org/v4/matches',
            'params': {'dateFrom': date_str, 'dateTo': date_str},
            'name': 'API v4 com dateFrom/dateTo'
        },
        {
            'url': 'https://api.football-data.org/v2/matches',
            'params': {'dateFrom': date_str, 'dateTo': date_str},
            'name': 'API v2 com dateFrom/dateTo'
        },
        {
            'url': 'https://api.football-data.org/v4/matches',
            'params': {},
            'name': 'API v4 sem filtros'
        }
    ]
    
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
                
                # Filtrar para a data específica se necessário
                if not endpoint['params']:
                    matches = [m for m in matches if m.get('utcDate', '')[:10] == date_str]
                
                return matches, f"✅ {len(matches)} jogos encontrados via {endpoint['name']}"
            
            elif response.status_code == 429:
                time.sleep(2)
                continue
                
        except Exception as e:
            continue
    
    return [], "❌ Nenhum endpoint funcionou"

def create_demo_data():
    """Dados demo para quando API não funciona"""
    demo_teams = {
        'Manchester City': {
            'league': 'Premier League',
            'over_rate': 0.76,
            'home_over_rate': 0.83,
            'away_over_rate': 0.69,
            'total_games': 29,
            'classification': '🔥 EQUIPE OVER FORTE'
        },
        'Liverpool': {
            'league': 'Premier League', 
            'over_rate': 0.71,
            'home_over_rate': 0.79,
            'away_over_rate': 0.63,
            'total_games': 28,
            'classification': '📈 EQUIPE OVER'
        },
        'Arsenal': {
            'league': 'Premier League',
            'over_rate': 0.67,
            'home_over_rate': 0.73,
            'away_over_rate': 0.61,
            'total_games': 27,
            'classification': '📈 EQUIPE OVER'
        },
        'Real Madrid': {
            'league': 'La Liga',
            'over_rate': 0.78,
            'home_over_rate': 0.85,
            'away_over_rate': 0.71,
            'total_games': 32,
            'classification': '🔥 EQUIPE OVER FORTE'
        },
        'Barcelona': {
            'league': 'La Liga',
            'over_rate': 0.74,
            'home_over_rate': 0.81,
            'away_over_rate': 0.67,
            'total_games': 31,
            'classification': '📈 EQUIPE OVER'
        },
        'Atletico Madrid': {
            'league': 'La Liga',
            'over_rate': 0.35,
            'home_over_rate': 0.42,
            'away_over_rate': 0.28,
            'total_games': 30,
            'classification': '📉 EQUIPE UNDER'
        },
        'Bayern Munich': {
            'league': 'Bundesliga',
            'over_rate': 0.84,
            'home_over_rate': 0.91,
            'away_over_rate': 0.77,
            'total_games': 26,
            'classification': '🔥 EQUIPE OVER FORTE'
        },
        'Borussia Dortmund': {
            'league': 'Bundesliga',
            'over_rate': 0.73,
            'home_over_rate': 0.80,
            'away_over_rate': 0.66,
            'total_games': 25,
            'classification': '📈 EQUIPE OVER'
        }
    }
    
    demo_matches = [
        {'home': 'Manchester City', 'away': 'Liverpool', 'league': 'Premier League', 'time': '15:30'},
        {'home': 'Real Madrid', 'away': 'Barcelona', 'league': 'La Liga', 'time': '21:00'},
        {'home': 'Bayern Munich', 'away': 'Borussia Dortmund', 'league': 'Bundesliga', 'time': '18:30'},
        {'home': 'Arsenal', 'away': 'Atletico Madrid', 'league': 'Amistoso', 'time': '20:00'},
    ]
    
    return demo_teams, demo_matches

def predict_demo_matches(demo_teams, demo_matches):
    """Faz previsões com dados demo"""
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
        
        # Algoritmo de predição
        combined_rate = (home_rate * 0.6) + (away_rate * 0.4)
        
        # Ajustes
        if home_stats['over_rate'] > 0.75 and away_stats['over_rate'] > 0.75:
            combined_rate += 0.05
        elif home_stats['over_rate'] < 0.35 and away_stats['over_rate'] < 0.35:
            combined_rate -= 0.05
        
        # Classificação
        if combined_rate >= 0.70:
            prediction = "✅ OVER 0.5"
            confidence = "ALTA"
            bet_class = "team-over"
        elif combined_rate >= 0.55:
            prediction = "✅ OVER 0.5"
            confidence = "MÉDIA"
            bet_class = "team-over"
        elif combined_rate <= 0.30:
            prediction = "❌ UNDER 0.5"
            confidence = "ALTA"
            bet_class = "team-under"
        elif combined_rate <= 0.45:
            prediction = "❌ UNDER 0.5"
            confidence = "MÉDIA"
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
    
    return predictions

def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>⚽ Over 0.5 HT - Sistema Inteligente</h1>
        <p>Configure sua API Key e teste o sistema</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input da API Key
    st.markdown('<div class="api-key-input">', unsafe_allow_html=True)
    st.subheader("🔑 Configuração da API Key")
    
    # Input manual da API Key com sua chave pré-preenchida
    user_api_key = st.text_input(
        "Cole sua API Key do football-data.org:",
        value="2aad0db0e5b88b3a080bdc85461a919",
        help="Sua API Key já está configurada - clique em Testar",
        type="password"
    )
    
    st.markdown("### 📋 Como obter sua API Key:")
    st.write("""
    1. **Acesse:** https://www.football-data.org
    2. **Faça login** na sua conta
    3. **Vá em:** Dashboard → My Account → API Key
    4. **Copie exatamente** como aparece (sem espaços)
    5. **Cole acima** e clique em "Testar"
    """)
    
    # Teste da API Key
    if user_api_key:
        if st.button("🚀 TESTAR API KEY", type="primary"):
            with st.spinner("Testando API Key..."):
                success, message = test_api_key(user_api_key)
            
            if success:
                st.success(message)
                st.session_state.api_key = user_api_key
                st.session_state.api_validated = True
            else:
                st.error(message)
                st.session_state.api_validated = False
                
                # Dicas de solução
                st.warning("""
                **💡 Possíveis soluções:**
                - Verifique se copiou a API Key completa
                - Confirme se sua conta Pro está ativa
                - Tente aguardar alguns minutos (rate limit)
                - Verifique status da conta no dashboard
                """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tabs principais
    tab1, tab2, tab3 = st.tabs([
        "🎯 ANÁLISE REAL", 
        "📊 DEMO COMPLETO",
        "ℹ️ AJUDA"
    ])
    
    with tab1:
        st.header("🎯 Análise com Dados Reais")
        
        if not user_api_key:
            st.info("👆 Configure sua API Key acima para usar dados reais")
        elif not st.session_state.get('api_validated'):
            st.warning("⚠️ Teste sua API Key primeiro")
        else:
            selected_date = st.date_input(
                "📅 Data para análise:",
                value=datetime.now().date()
            )
            
            if st.button("🔍 BUSCAR JOGOS REAIS"):
                with st.spinner("Buscando jogos..."):
                    matches, message = get_matches_with_key(user_api_key, selected_date)
                
                st.info(message)
                
                if matches:
                    # Mostrar jogos encontrados
                    upcoming = [m for m in matches if m.get('status') in ['SCHEDULED', 'TIMED']]
                    finished = [m for m in matches if m.get('status') == 'FINISHED']
                    
                    if upcoming:
                        st.subheader(f"📅 {len(upcoming)} Jogos Programados")
                        for match in upcoming[:5]:
                            home = match['homeTeam']['name']
                            away = match['awayTeam']['name']
                            comp = match['competition']['name']
                            st.write(f"⚽ **{home} vs {away}** ({comp})")
                    
                    if finished:
                        st.subheader(f"✅ {len(finished)} Jogos Finalizados")
                        for match in finished[:5]:
                            home = match['homeTeam']['name']
                            away = match['awayTeam']['name']
                            comp = match['competition']['name']
                            
                            ht = match.get('score', {}).get('halfTime', {})
                            if ht and ht.get('home') is not None:
                                ht_goals = ht['home'] + ht['away']
                                result = "Over 0.5" if ht_goals > 0.5 else "Under 0.5"
                                st.write(f"⚽ **{home} vs {away}** - HT: {ht['home']}-{ht['away']} ({result}) | {comp}")
                else:
                    st.warning("❌ Nenhum jogo encontrado para esta data")
                    st.info("💡 Tente uma data diferente ou veja o DEMO")
    
    with tab2:
        st.header("📊 DEMO - Sistema Funcionando")
        st.info("🎯 Veja como o sistema funciona com dados de exemplo")
        
        if st.button("🚀 VER DEMO COMPLETO"):
            demo_teams, demo_matches = create_demo_data()
            predictions = predict_demo_matches(demo_teams, demo_matches)
            
            st.subheader("🏆 PREVISÕES INTELIGENTES")
            
            for pred in predictions:
                st.markdown(f"""
                <div class="{pred['bet_class']}">
                    <h4>⚽ {pred['home_team']} vs {pred['away_team']} - {pred['time']}</h4>
                    <p><strong>Liga:</strong> {pred['league']}</p>
                    <p><strong>Previsão:</strong> {pred['prediction']} ({pred['probability']})</p>
                    <p><strong>Confiança:</strong> {pred['confidence']}</p>
                    <hr style="border-color: rgba(255,255,255,0.3);">
                    <p><strong>🏠 {pred['home_team']}:</strong> {pred['home_class']} - Casa: {pred['home_rate']}</p>
                    <p><strong>✈️ {pred['away_team']}:</strong> {pred['away_class']} - Fora: {pred['away_rate']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Ranking
            st.subheader("🏆 Ranking das Equipes")
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
    
    with tab3:
        st.header("ℹ️ Ajuda e Soluções")
        
        st.write("""
        ### 🔧 **Resolvendo Problemas de API:**
        
        **Erro "Your API token is invalid":**
        1. 🔑 **API Key incorreta** - Copie novamente do dashboard
        2. 💳 **Conta suspensa** - Verifique problemas de pagamento
        3. ⏰ **Chave expirada** - Gere uma nova no dashboard
        4. 📱 **Caracteres extras** - Cole sem espaços/quebras de linha
        
        ### 📋 **Checklist de Verificação:**
        
        - ✅ Sua conta Pro está ativa?
        - ✅ A API Key foi copiada completa?
        - ✅ Não há espaços antes/depois da chave?
        - ✅ Sua internet está funcionando?
        - ✅ Não há problemas de pagamento na conta?
        
        ### 🎯 **Testando Manualmente:**
        
        Você pode testar sua API Key diretamente:
        
        1. **Abra:** https://api.football-data.org/v4/competitions
        2. **Adicione header:** `X-Auth-Token: SUA_CHAVE`
        3. **Deve retornar:** Lista de competições
        
        ### 📞 **Contato com Suporte:**
        
        Se continuar com problemas:
        - **Email:** daniel@football-data.org
        - **Site:** https://www.football-data.org/support
        
        ### 🚀 **Enquanto isso:**
        
        Use o **DEMO** para ver exatamente como o sistema funciona quando a API estiver configurada corretamente!
        """)

if __name__ == "__main__":
    # Inicializar session state
    if 'api_validated' not in st.session_state:
        st.session_state.api_validated = False
    
    main()
