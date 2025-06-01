import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="⚽ HT Goals AI Engine",
    page_icon="🤖",
    layout="wide"
)

# Configuração da API Key
try:
    API_KEY = st.secrets["API_KEY"]
except:
    API_KEY = "474f15de5b22951077ccb71b8d75b95c"

# URL base da API-SPORTS
API_BASE_URL = "https://v3.football.api-sports.io"

# Diretório para salvar modelos
MODEL_DIR = "models"
try:
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
except:
    MODEL_DIR = "/tmp/models"
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

# Médias de Over 0.5 HT por Liga
LEAGUE_AVERAGES = {
    'Premier League': 65,
    'La Liga': 60,
    'Serie A': 55,
    'Bundesliga': 70,
    'Ligue 1': 58,
    'Eredivisie': 75,
    'Primeira Liga': 62,
    'Championship': 68,
    'Liga MX': 72,
    'MLS': 70,
    'Brasileirão': 68,
    'Brasileirão Série A': 68,
    'Brasileirão Série B': 65,
    'Champions League': 58,
    'Europa League': 62,
    'Copa Libertadores': 65,
    'Bundesliga 2': 72,
    'Serie B': 58,
    'La Liga 2': 62,
    'Liga Portugal 2': 65,
    'Ligue 2': 60,
    'Scottish Premiership': 68,
    'Belgian Pro League': 63,
    'Superliga Argentina': 66,
    'Liga Profesional': 66,
    'J1 League': 61,
    'K League 1': 64,
    'Chinese Super League': 59,
    'Indian Super League': 67,
    'A-League': 69,
    'USL League One': 71,
    'USL League Two': 73,
    'USL Championship': 69
}

def get_api_headers():
    """Retorna os headers corretos para API-SPORTS"""
    return {'x-apisports-key': API_KEY}

def test_api_connection():
    """Testa a conectividade com a API"""
    try:
        response = requests.get(f'{API_BASE_URL}/status', headers=get_api_headers(), timeout=10)
        return response.status_code == 200
    except:
        return False

def get_fixtures_for_date(date_str):
    """Busca jogos de uma data específica"""
    try:
        response = requests.get(
            f'{API_BASE_URL}/fixtures',
            headers=get_api_headers(),
            params={'date': date_str},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get('response', [])
    except:
        pass
    return []

def load_historical_data():
    """Carrega dados históricos do cache"""
    data_files = [
        "data/historical_matches.parquet",
        "data/historical_matches.csv",
        "historical_matches.parquet",
        "historical_matches.csv"
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            try:
                if file_path.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                else:
                    df = pd.read_csv(file_path)
                
                if 'ht_home' in df.columns and 'ht_away' in df.columns:
                    df['over_05'] = (df['ht_home'] + df['ht_away']) > 0
                return df
            except:
                continue
    return None

def collect_minimal_data(days=7):
    """Coleta dados mínimos para treinar o modelo"""
    all_data = []
    
    for i in range(days):
        date = datetime.now() - timedelta(days=i+1)
        date_str = date.strftime('%Y-%m-%d')
        
        fixtures = get_fixtures_for_date(date_str)
        
        for match in fixtures:
            try:
                if match['fixture']['status']['short'] == 'FT' and match.get('score', {}).get('halftime'):
                    ht_home = match['score']['halftime']['home']
                    ht_away = match['score']['halftime']['away']
                    
                    match_data = {
                        'date': match['fixture']['date'][:10],
                        'league_id': match['league']['id'],
                        'league_name': match['league']['name'],
                        'country': match['league']['country'],
                        'home_team': match['teams']['home']['name'],
                        'away_team': match['teams']['away']['name'],
                        'home_team_id': match['teams']['home']['id'],
                        'away_team_id': match['teams']['away']['id'],
                        'ht_home_goals': ht_home,
                        'ht_away_goals': ht_away,
                        'ht_total_goals': ht_home + ht_away,
                        'over_05': 1 if (ht_home + ht_away) > 0 else 0
                    }
                    all_data.append(match_data)
            except:
                continue
    
    return pd.DataFrame(all_data)

def prepare_simple_features(df):
    """Prepara features simples e eficientes"""
    team_stats = {}
    
    # Calcular estatísticas por time
    for idx, row in df.iterrows():
        home_id = row['home_team_id']
        away_id = row['away_team_id']
        
        if home_id not in team_stats:
            team_stats[home_id] = {'games': 0, 'over_05': 0, 'goals': 0}
        if away_id not in team_stats:
            team_stats[away_id] = {'games': 0, 'over_05': 0, 'goals': 0}
    
    features = []
    
    for idx, row in df.iterrows():
        home_id = row['home_team_id']
        away_id = row['away_team_id']
        
        home_stats = team_stats[home_id]
        away_stats = team_stats[away_id]
        
        # Features simples
        home_over_rate = home_stats['over_05'] / max(home_stats['games'], 1)
        away_over_rate = away_stats['over_05'] / max(away_stats['games'], 1)
        
        features.append({
            'home_over_rate': home_over_rate,
            'away_over_rate': away_over_rate,
            'combined_over_rate': (home_over_rate + away_over_rate) / 2,
            'target': row['over_05']
        })
        
        # Atualizar estatísticas
        team_stats[home_id]['games'] += 1
        team_stats[home_id]['over_05'] += row['over_05']
        team_stats[home_id]['goals'] += row['ht_home_goals']
        
        team_stats[away_id]['games'] += 1
        team_stats[away_id]['over_05'] += row['over_05']
        team_stats[away_id]['goals'] += row['ht_away_goals']
    
    return pd.DataFrame(features), team_stats

def train_simple_model(df):
    """Treina modelo simplificado"""
    features_df, team_stats = prepare_simple_features(df)
    
    X = features_df[['home_over_rate', 'away_over_rate', 'combined_over_rate']]
    y = features_df['target']
    
    # Split 70/15/15
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)
    
    # Treinar modelo
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Avaliar
    test_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, test_pred)
    
    return {
        'model': model,
        'team_stats': team_stats,
        'accuracy': accuracy,
        'feature_names': ['home_over_rate', 'away_over_rate', 'combined_over_rate']
    }

def predict_match(fixture, model_data):
    """Faz previsão para um jogo"""
    try:
        home_id = fixture['teams']['home']['id']
        away_id = fixture['teams']['away']['id']
        
        team_stats = model_data['team_stats']
        
        if home_id not in team_stats or away_id not in team_stats:
            return None
        
        home_stats = team_stats[home_id]
        away_stats = team_stats[away_id]
        
        home_over_rate = home_stats['over_05'] / max(home_stats['games'], 1)
        away_over_rate = away_stats['over_05'] / max(away_stats['games'], 1)
        combined_rate = (home_over_rate + away_over_rate) / 2
        
        X = pd.DataFrame([[home_over_rate, away_over_rate, combined_rate]], 
                        columns=model_data['feature_names'])
        
        pred_proba = model_data['model'].predict_proba(X)[0]
        pred_class = model_data['model'].predict(X)[0]
        
        return {
            'home_team': fixture['teams']['home']['name'],
            'away_team': fixture['teams']['away']['name'],
            'league': fixture['league']['name'],
            'country': fixture['league']['country'],
            'kickoff': fixture['fixture']['date'],
            'prediction': 'OVER 0.5' if pred_class == 1 else 'UNDER 0.5',
            'confidence': max(pred_proba) * 100,
            'probability_over': pred_proba[1] * 100
        }
    except:
        return None

def get_league_average(league_name):
    """Retorna a média da liga"""
    for key, value in LEAGUE_AVERAGES.items():
        if key.lower() in league_name.lower() or league_name.lower() in key.lower():
            return value
    return 60  # Média padrão

def display_prediction_card(pred):
    """Exibe card de previsão no estilo da imagem"""
    # Container principal
    with st.container():
        # Criar card com bordas
        card_html = f"""
        <div style="
            background-color: white;
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h3 style="margin: 0; color: #1a1a1a;">⚽ {pred['home_team']} vs {pred['away_team']}</h3>
                <div style="
                    background-color: #4CAF50;
                    color: white;
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-weight: bold;
                    font-size: 16px;
                ">{pred['confidence']:.1f}%</div>
            </div>
            
            <div style="display: flex; gap: 20px; margin-bottom: 15px; color: #666;">
                <span>🏆 <strong>Liga:</strong> {pred['league']} ({pred['country']})</span>
                <span>🕐 <strong>Horário PT:</strong> {pred['kickoff'][11:16]}</span>
            </div>
            
            <div style="
                background-color: #f5f5f5;
                padding: 15px;
                border-radius: 8px;
                margin-top: 10px;
            ">
                <div style="font-size: 16px; margin-bottom: 10px;">
                    🎯 <strong>Previsão ML:</strong> <span style="color: #4CAF50; font-weight: bold;">{pred['prediction']}</span>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 10px;">
                    <div>
                        <div style="color: #666; font-size: 14px;">📊 Média da Liga</div>
                        <div style="font-size: 18px; font-weight: bold; color: #1a1a1a;">{get_league_average(pred['league']):.0f}%</div>
                    </div>
                    <div>
                        <div style="color: #666; font-size: 14px;">🤖 Média do Sistema Over 0.5 HT</div>
                        <div style="font-size: 18px; font-weight: bold; color: #4CAF50;">{pred['probability_over']:.0f}%</div>
                    </div>
                </div>
            </div>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)

def main():
    # Header principal
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #1a1a1a; font-size: 2.5rem;">⚽ HT Goals AI Engine</h1>
        <p style="color: #666; font-size: 1.1rem;">Sistema Automático de Previsões Over 0.5 HT</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Container central
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Status do sistema
        if 'model_data' not in st.session_state:
            st.info("🤖 Sistema pronto para iniciar análise automática")
        else:
            st.success(f"✅ Modelo treinado - Precisão: {st.session_state.model_data['accuracy']:.1%}")
        
        # Botão único centralizado
        if st.button("🚀 ANALISAR JOGOS DE HOJE", use_container_width=True, type="primary"):
            
            # Passo 1: Verificar conexão
            with st.spinner("🔍 Verificando conexão com API..."):
                if not test_api_connection():
                    st.error("❌ Erro de conexão. Verifique sua internet.")
                    return
            
            # Passo 2: Carregar/Coletar dados
            with st.spinner("📊 Carregando dados históricos..."):
                df = load_historical_data()
                if df is None:
                    st.info("📥 Coletando dados recentes...")
                    df = collect_minimal_data(days=7)
                    if df.empty:
                        st.error("❌ Não foi possível coletar dados")
                        return
            
            # Passo 3: Treinar modelo
            with st.spinner("🧠 Treinando modelo ML (70% treino, 15% validação, 15% teste)..."):
                model_data = train_simple_model(df)
                st.session_state.model_data = model_data
                st.success(f"✅ Modelo treinado com sucesso! Precisão: {model_data['accuracy']:.1%}")
            
            # Passo 4: Buscar jogos de hoje
            today = datetime.now().strftime('%Y-%m-%d')
            with st.spinner("🔍 Buscando jogos de hoje..."):
                fixtures = get_fixtures_for_date(today)
                
                if not fixtures:
                    st.warning("📅 Nenhum jogo encontrado para hoje")
                    return
            
            # Passo 5: Fazer previsões
            with st.spinner("🎯 Gerando previsões..."):
                predictions = []
                for fixture in fixtures:
                    if fixture['fixture']['status']['short'] in ['NS', 'TBD']:
                        pred = predict_match(fixture, model_data)
                        if pred and pred['prediction'] == 'OVER 0.5' and pred['confidence'] > 65:
                            predictions.append(pred)
                
                # Ordenar por confiança
                predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Passo 6: Exibir resultados
            if predictions:
                st.markdown("---")
                st.markdown(f"### 🏆 Melhores Apostas Over 0.5 HT - {datetime.now().strftime('%d/%m/%Y')}")
                
                # Estatísticas resumo
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total de Jogos", len(predictions))
                with col2:
                    avg_conf = sum(p['confidence'] for p in predictions) / len(predictions)
                    st.metric("Confiança Média", f"{avg_conf:.1f}%")
                with col3:
                    high_conf = len([p for p in predictions if p['confidence'] > 70])
                    st.metric("Alta Confiança (>70%)", high_conf)
                
                st.markdown("---")
                
                # Exibir cards
                for pred in predictions[:10]:  # Top 10
                    display_prediction_card(pred)
            else:
                st.info("🤷 Nenhuma aposta com boa confiança encontrada hoje")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>Sistema automático que treina com dados históricos e prevê jogos do dia</p>
        <p>Médias das ligas baseadas em estatísticas históricas de Over 0.5 HT</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
