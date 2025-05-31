import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import json

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
    .debug-info {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .team-over {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def debug_api_call(endpoint, params=None, description=""):
    """Função de debug para chamadas da API"""
    headers = {
        'X-RapidAPI-Key': API_KEY,
        'X-RapidAPI-Host': 'v3.football.api-sports.io'
    }
    
    url = f"https://v3.football.api-sports.io/{endpoint}"
    
    debug_info = []
    debug_info.append(f"🔍 {description}")
    debug_info.append(f"📡 URL: {url}")
    debug_info.append(f"🔑 Headers: {headers}")
    debug_info.append(f"📊 Params: {params}")
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        
        debug_info.append(f"📈 Status Code: {response.status_code}")
        debug_info.append(f"📏 Response Size: {len(response.content)} bytes")
        
        if response.status_code == 200:
            try:
                data = response.json()
                debug_info.append(f"✅ JSON Success!")
                debug_info.append(f"🔢 Response Keys: {list(data.keys())}")
                
                if 'response' in data:
                    response_data = data['response']
                    debug_info.append(f"📊 Items Count: {len(response_data)}")
                    
                    if len(response_data) > 0:
                        debug_info.append(f"📋 First Item Keys: {list(response_data[0].keys())}")
                        debug_info.append(f"🎯 Sample Data: {str(response_data[0])[:200]}...")
                
                return True, data, debug_info
                
            except Exception as e:
                debug_info.append(f"❌ JSON Parse Error: {str(e)}")
                debug_info.append(f"📝 Raw Response: {response.text[:200]}...")
                return False, None, debug_info
        else:
            debug_info.append(f"❌ HTTP Error: {response.status_code}")
            debug_info.append(f"📝 Error Response: {response.text[:200]}...")
            return False, None, debug_info
            
    except Exception as e:
        debug_info.append(f"❌ Request Exception: {str(e)}")
        return False, None, debug_info

def test_multiple_dates():
    """Testa múltiplas datas para encontrar jogos"""
    test_dates = []
    today = datetime.now()
    
    # Teste últimos 7 dias e próximos 7 dias
    for i in range(-7, 8):
        date = today + timedelta(days=i)
        test_dates.append(date)
    
    results = []
    
    for date in test_dates:
        date_str = date.strftime('%Y-%m-%d')
        success, data, debug_info = debug_api_call(
            'fixtures', 
            {'date': date_str}, 
            f"Testando data: {date_str}"
        )
        
        match_count = 0
        if success and data:
            match_count = len(data.get('response', []))
        
        results.append({
            'date': date_str,
            'day_name': date.strftime('%A'),
            'matches': match_count,
            'success': success,
            'debug': debug_info
        })
    
    return results

def get_live_fixtures():
    """Busca jogos ao vivo"""
    success, data, debug_info = debug_api_call(
        'fixtures', 
        {'live': 'all'}, 
        "Buscando jogos AO VIVO"
    )
    
    return success, data, debug_info

def get_todays_fixtures_detailed():
    """Busca jogos de hoje com debug detalhado"""
    today = datetime.now()
    date_str = today.strftime('%Y-%m-%d')
    
    # Múltiplos formatos para testar
    test_params = [
        {'date': date_str},
        {'from': date_str, 'to': date_str},
        {'date': date_str, 'timezone': 'Europe/London'},
        {'date': date_str, 'status': 'NS-LIVE-FT'},
    ]
    
    results = []
    
    for i, params in enumerate(test_params):
        success, data, debug_info = debug_api_call(
            'fixtures', 
            params, 
            f"Teste {i+1}: Formato {params}"
        )
        
        match_count = 0
        if success and data:
            match_count = len(data.get('response', []))
        
        results.append({
            'test': f"Teste {i+1}",
            'params': params,
            'matches': match_count,
            'success': success,
            'debug': debug_info,
            'data': data if success else None
        })
    
    return results

def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>⚽ DEBUG API-Football - Buscar TODOS os Jogos</h1>
        <p>Sistema de diagnóstico para encontrar jogos da API-Football</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Debug API-Football")
    st.sidebar.info("🔑 API Key configurada")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 BUSCA HOJE", 
        "📅 TESTE MÚLTIPLAS DATAS",
        "📺 JOGOS AO VIVO", 
        "🎯 DEMO SISTEMA"
    ])
    
    with tab1:
        st.header("🔍 Debug - Busca Jogos de Hoje")
        st.write(f"📅 Data: {datetime.now().strftime('%Y-%m-%d (%A)')}")
        
        if st.button("🚀 BUSCAR JOGOS DE HOJE - DEBUG COMPLETO", type="primary"):
            with st.spinner("🔍 Testando múltiplos formatos..."):
                results = get_todays_fixtures_detailed()
            
            # Mostrar resultados
            total_matches = 0
            working_result = None
            
            for result in results:
                st.subheader(f"📊 {result['test']}")
                
                if result['success']:
                    st.success(f"✅ Sucesso: {result['matches']} jogos encontrados")
                    total_matches += result['matches']
                    if result['matches'] > 0 and not working_result:
                        working_result = result
                else:
                    st.error(f"❌ Falhou: 0 jogos")
                
                # Debug info
                with st.expander(f"🔍 Debug {result['test']}"):
                    for debug_line in result['debug']:
                        st.text(debug_line)
            
            # Resumo
            st.header("📈 RESUMO")
            if total_matches > 0:
                st.success(f"🎉 TOTAL: {total_matches} jogos encontrados!")
                
                if working_result and working_result['data']:
                    st.subheader("⚽ Exemplos de Jogos Encontrados:")
                    matches = working_result['data']['response'][:10]
                    
                    for match in matches:
                        home = match['teams']['home']['name']
                        away = match['teams']['away']['name']
                        league = match['league']['name']
                        status = match['fixture']['status']['short']
                        time_str = match['fixture']['date'][:16]
                        
                        st.write(f"🏆 **{league}** | ⚽ {home} vs {away} | 🕐 {time_str} | 📊 {status}")
                        
                        # Mostrar resultado HT se disponível
                        if status == 'FT' and match['score']['halftime']:
                            ht = match['score']['halftime']
                            if ht['home'] is not None and ht['away'] is not None:
                                ht_total = ht['home'] + ht['away']
                                ht_result = "Over 0.5" if ht_total > 0.5 else "Under 0.5"
                                st.write(f"   📊 HT: {ht['home']}-{ht['away']} ({ht_result})")
                        
                        st.write("---")
            else:
                st.error("❌ Nenhum jogo encontrado em nenhum formato")
                st.info("💡 Isso pode significar:")
                st.write("- Data sem jogos programados")
                st.write("- Problema com a API Key")
                st.write("- Formato de request incorreto")
                st.write("- Rate limit atingido")
    
    with tab2:
        st.header("📅 Teste Múltiplas Datas")
        st.info("Vamos testar 15 datas (7 passadas + hoje + 7 futuras) para encontrar jogos")
        
        if st.button("📊 TESTAR 15 DATAS", type="primary"):
            with st.spinner("📅 Testando múltiplas datas..."):
                date_results = test_multiple_dates()
            
            # Criar tabela de resultados
            df_data = []
            total_found = 0
            
            for result in date_results:
                df_data.append({
                    'Data': result['date'],
                    'Dia': result['day_name'],
                    'Jogos': result['matches'],
                    'Status': '✅ Sucesso' if result['success'] else '❌ Erro'
                })
                
                if result['success']:
                    total_found += result['matches']
            
            # Mostrar tabela
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            # Resumo
            st.metric("🎯 Total de Jogos Encontrados", total_found)
            
            # Melhores datas
            best_dates = [r for r in date_results if r['matches'] > 0]
            if best_dates:
                st.subheader("📅 Datas com Mais Jogos:")
                for result in sorted(best_dates, key=lambda x: x['matches'], reverse=True)[:5]:
                    st.write(f"📅 **{result['date']}** ({result['day_name']}): {result['matches']} jogos")
            else:
                st.warning("❌ Nenhuma data com jogos encontrada")
    
    with tab3:
        st.header("📺 Jogos Ao Vivo")
        st.info("Buscar jogos que estão acontecendo AGORA")
        
        if st.button("🔴 BUSCAR JOGOS AO VIVO"):
            with st.spinner("📺 Buscando jogos ao vivo..."):
                success, data, debug_info = get_live_fixtures()
            
            # Debug info
            with st.expander("🔍 Debug Jogos Ao Vivo"):
                for debug_line in debug_info:
                    st.text(debug_line)
            
            if success and data:
                live_matches = data.get('response', [])
                
                if live_matches:
                    st.success(f"🔴 {len(live_matches)} jogos ao vivo!")
                    
                    for match in live_matches:
                        home = match['teams']['home']['name']
                        away = match['teams']['away']['name']
                        league = match['league']['name']
                        minute = match['fixture']['status']['elapsed']
                        
                        ft_score = match['score']['fulltime']
                        ht_score = match['score']['halftime']
                        
                        st.write(f"🔴 **{league}** | ⚽ {home} vs {away}")
                        st.write(f"   ⏰ {minute}' | 📊 {ft_score['home']}-{ft_score['away']}")
                        
                        if ht_score and ht_score['home'] is not None:
                            ht_total = ht_score['home'] + ht_score['away']
                            ht_result = "Over 0.5" if ht_total > 0.5 else "Under 0.5"
                            st.write(f"   📊 HT: {ht_score['home']}-{ht_score['away']} ({ht_result})")
                        
                        st.write("---")
                else:
                    st.info("📺 Nenhum jogo ao vivo no momento")
            else:
                st.error("❌ Erro ao buscar jogos ao vivo")
    
    with tab4:
        st.header("🎯 Demo do Sistema Over 0.5 HT")
        st.info("Como o sistema funcionará quando encontrar jogos")
        
        # Demo com dados fictícios
        demo_matches = [
            {
                'home': 'Manchester City', 'away': 'Liverpool', 'league': 'Premier League',
                'home_rate': 0.85, 'away_rate': 0.71, 'prediction': '✅ OVER 0.5', 'confidence': 'ALTA', 'prob': '78%'
            },
            {
                'home': 'Real Madrid', 'away': 'Barcelona', 'league': 'La Liga',
                'home_rate': 0.82, 'away_rate': 0.74, 'prediction': '✅ OVER 0.5', 'confidence': 'ALTA', 'prob': '76%'
            },
            {
                'home': 'Atletico Madrid', 'away': 'Getafe', 'league': 'La Liga',
                'home_rate': 0.35, 'away_rate': 0.28, 'prediction': '❌ UNDER 0.5', 'confidence': 'MÉDIA', 'prob': '32%'
            }
        ]
        
        st.subheader("🏆 EXEMPLO DE PREVISÕES")
        
        for match in demo_matches:
            if 'OVER' in match['prediction']:
                css_class = 'team-over'
            else:
                css_class = 'team-under'
            
            st.markdown(f"""
            <div class="{css_class}">
                <h4>⚽ {match['home']} vs {match['away']}</h4>
                <p><strong>Liga:</strong> {match['league']}</p>
                <p><strong>Previsão:</strong> {match['prediction']} ({match['prob']})</p>
                <p><strong>Confiança:</strong> {match['confidence']}</p>
                <p><strong>🏠 Casa:</strong> {match['home_rate']:.0%} | <strong>✈️ Fora:</strong> {match['away_rate']:.0%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.write("### 🎯 Quando a API encontrar jogos reais, o sistema mostrará:")
        st.write("✅ Análise de 900+ ligas mundiais")
        st.write("✅ Previsões Over/Under 0.5 HT")
        st.write("✅ Sistema de confiança (ALTA/MÉDIA/BAIXA)")
        st.write("✅ Estatísticas detalhadas das equipes")
        st.write("✅ Ranking das melhores oportunidades")

if __name__ == "__main__":
    main()
