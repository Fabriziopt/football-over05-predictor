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

# Inicializar session state para o modelo
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

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
except Exception as e:
    MODEL_DIR = "/tmp/models"
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

# CSS COMPLETO - Neumorphism + Cards Limpos
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Configuração global neumorphism */
    .stApp {
        background: #e0e5ec;
        font-family: 'Inter', sans-serif;
        color: #2d3748;
    }
    
    /* Header principal neumorphism */
    .main-header {
        background: #e0e5ec;
        padding: 3rem 2rem;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 15px 15px 30px #c8d0e7, -15px -15px 30px #f8ffff;
        color: #2d3748;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.05), transparent);
        animation: shine 4s infinite;
    }
    
    @keyframes shine {
        0% { left: -100%; }
        50%, 100% { left: 100%; }
    }
    
    .main-header h1 {
        color: #667eea;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(102, 126, 234, 0.1);
    }
    
    .main-header p {
        color: #718096;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    /* Cards neumorphism modernos */
    .metric-card {
        background: #e0e5ec;
        padding: 2rem 1.5rem;
        border-radius: 20px;
        box-shadow: 9px 9px 18px #c8d0e7, -9px -9px 18px #f8ffff;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        color: #2d3748;
        border: none;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, transparent 0%, rgba(102, 126, 234, 0.02) 50%, transparent 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-card:hover {
        box-shadow: inset 9px 9px 18px #c8d0e7, inset -9px -9px 18px #f8ffff;
        transform: scale(0.98);
    }
    
    .metric-card h1 {
        color: #667eea;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0.5rem 0;
        text-shadow: 1px 1px 2px rgba(102, 126, 234, 0.1);
    }
    
    .metric-card h3 {
        color: #4a5568;
        font-size: 1rem;
        font-weight: 600;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Cards simples e limpos */
    .prediction-card-simple {
        background: #e0e5ec;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 8px 8px 16px #c8d0e7, -8px -8px 16px #f8ffff;
        transition: all 0.3s ease;
        border: none;
    }
    
    .prediction-card-simple:hover {
        transform: translateY(-2px);
        box-shadow: 10px 10px 20px #c8d0e7, -10px -10px 20px #f8ffff;
    }
    
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 1rem;
    }
    
    .match-info h3 {
        color: #2d3748;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }
    
    .league-time {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }
    
    .league {
        color: #667eea;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .time {
        color: #718096;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    .confidence-simple {
        color: white;
        font-weight: 800;
        font-size: 1.2rem;
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        text-align: center;
        min-width: 70px;
        box-shadow: inset 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .prediction-info {
        background: rgba(102, 126, 234, 0.02);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: inset 2px 2px 4px rgba(200, 208, 231, 0.3);
    }
    
    .main-prediction {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(200, 208, 231, 0.3);
    }
    
    .prediction-label {
        color: #4a5568;
        font-weight: 600;
        font-size: 1rem;
    }
    
    .prediction-value {
        color: #2d3748;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .context-info {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.8rem;
    }
    
    .context-item {
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
    }
    
    .context-label {
        color: #718096;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    
    .context-value {
        color: #2d3748;
        font-size: 0.95rem;
        font-weight: 600;
    }
    
    /* Lista de jogos simplificada */
    .games-list-simple {
        background: #e0e5ec;
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 2rem;
        box-shadow: 8px 8px 16px #c8d0e7, -8px -8px 16px #f8ffff;
    }
    
    .games-list-simple h3 {
        color: #667eea;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
        font-size: 1.2rem;
    }
    
    .game-item-simple {
        background: #e0e5ec;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        margin-bottom: 0.6rem;
        box-shadow: 3px 3px 6px #c8d0e7, -3px -3px 6px #f8ffff;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.2s ease;
    }
    
    .game-item-simple:hover {
        box-shadow: inset 3px 3px 6px #c8d0e7, inset -3px -3px 6px #f8ffff;
    }
    
    .game-teams-simple {
        font-weight: 600;
        color: #2d3748;
        font-size: 0.9rem;
    }
    
    .game-confidence-simple {
        background: #667eea;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    
    /* Sidebar neumorphism */
    .css-1d391kg {
        background: #e0e5ec;
        box-shadow: inset 3px 3px 6px #c8d0e7, inset -3px -3px 6px #f8ffff;
        border-right: none;
    }
    
    /* Botões neumorphism */
    .stButton > button {
        background: #e0e5ec;
        color: #667eea;
        border: none;
        border-radius: 15px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        box-shadow: 6px 6px 12px #c8d0e7, -6px -6px 12px #f8ffff;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        box-shadow: inset 6px 6px 12px #c8d0e7, inset -6px -6px 12px #f8ffff;
        transform: scale(0.98);
        color: #5a6fd8;
    }
    
    /* Tabs neumorphism */
    .stTabs [data-baseweb="tab-list"] {
        background: #e0e5ec;
        border-radius: 15px;
        box-shadow: inset 3px 3px 6px #c8d0e7, inset -3px -3px 6px #f8ffff;
        border: none;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #2d3748;
        border-radius: 10px;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: #e0e5ec;
        box-shadow: 3px 3px 6px #c8d0e7, -3px -3px 6px #f8ffff;
        color: #667eea;
    }
    
    /* Cards de liga */
    .league-card {
        background: #e0e5ec;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 6px 6px 12px #c8d0e7, -6px -6px 12px #f8ffff;
        border: none;
        transition: all 0.3s ease;
    }
    
    .league-card:hover {
        box-shadow: inset 6px 6px 12px #c8d0e7, inset -6px -6px 12px #f8ffff;
    }
    
    .league-card h4 {
        color: #667eea;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Classes de precisão */
    .accuracy-high {
        color: #38ef7d;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(56, 239, 125, 0.2);
    }
    
    .accuracy-medium {
        color: #f093fb;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(240, 147, 251, 0.2);
    }
    
    .accuracy-low {
        color: #ff6b6b;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(255, 107, 107, 0.2);
    }
    
    /* Status de conexão */
    .connection-status {
        background: #e0e5ec;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: inset 2px 2px 4px #c8d0e7, inset -2px -2px 4px #f8ffff;
        font-size: 0.85rem;
    }
    
    /* Responsividade */
    @media (max-width: 768px) {
        .main-header {
            padding: 2rem 1rem;
            border-radius: 15px;
        }
        
        .main-header h1 {
            font-size: 2rem;
        }
        
        .metric-card, .prediction-card-simple {
            border-radius: 15px;
            padding: 1.5rem;
        }
        
        .card-header {
            flex-direction: column;
            gap: 1rem;
            text-align: center;
        }
        
        .league-time {
            justify-content: center;
        }
        
        .context-info {
            grid-template-columns: 1fr;
        }
        
        .game-item-simple {
            flex-direction: column;
            gap: 0.5rem;
            text-align: center;
        }
    }
</style>
""", unsafe_allow_html=True)

def get_api_headers():
    """Retorna os headers corretos para API-SPORTS"""
    return {
        'x-apisports-key': API_KEY
    }

def test_api_connection():
    """Testa a conectividade com a API"""
    try:
        headers = get_api_headers()
        response = requests.get(
            f'{API_BASE_URL}/status',
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            return True, "Conexão OK"
        else:
            return False, f"Status HTTP: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return False, "Timeout - conexão lenta"
    except requests.exceptions.ConnectionError:
        return False, "Erro de conexão - verifique internet"
    except Exception as e:
        return False, f"Erro: {str(e)}"

def check_api_status():
    """Verifica o status e limites da API com tratamento robusto"""
    headers = get_api_headers()
    
    try:
        response = requests.get(
            f'{API_BASE_URL}/status',
            headers=headers,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'response' in data:
                status = data['response']
                account = status.get('account', {})
                subscription = status.get('subscription', {})
                requests_info = status.get('requests', {})
                
                requests_remaining = requests_info.get('limit_day', 0) - requests_info.get('current', 0)
                
                return True, requests_remaining, {
                    'account': account,
                    'subscription': subscription,
                    'requests': requests_info
                }
            elif 'errors' in data:
                if isinstance(data['errors'], list) and len(data['errors']) > 0:
                    error_msg = data['errors'][0]
                elif isinstance(data['errors'], dict):
                    error_msg = data['errors'].get('token', str(data['errors']))
                else:
                    error_msg = str(data['errors'])
                return False, 0, error_msg
        else:
            return False, 0, f"Status Code: {response.status_code}"
    except requests.exceptions.Timeout:
        return False, 0, "Timeout - conexão lenta"
    except requests.exceptions.ConnectionError:
        return False, 0, "Erro de conexão"
    except Exception as e:
        return False, 0, str(e)

def get_fixtures_with_retry(date_str, max_retries=3):
    """Busca jogos da API com retry automático para tratar erros de conexão"""
    headers = get_api_headers()
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                f'{API_BASE_URL}/fixtures',
                headers=headers,
                params={'date': date_str},
                timeout=30,
                stream=False
            )
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    if 'errors' in data and data['errors']:
                        if attempt == 0:  # Só mostrar erro na primeira tentativa
                            st.warning(f"Erro da API para {date_str}: {data['errors']}")
                        return []
                    
                    fixtures = data.get('response', [])
                    return fixtures
                    
                except Exception as json_error:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        return []
            else:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return []
                    
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                return []
                
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                return []
                
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                return []
    
    return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_fixtures_cached_robust(date_str):
    """Busca jogos com cache robusto"""
    try:
        return get_fixtures_with_retry(date_str)
    except Exception:
        return []

def load_historical_data():
    """Carrega dados históricos do arquivo local"""
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
                return df, f"✅ {len(df)} jogos carregados do arquivo local"
            except Exception:
                continue
    
    return None, "❌ Nenhum arquivo de dados históricos encontrado"

def collect_historical_data_robust(days=30, use_cached=True):
    """Coleta dados históricos com tratamento robusto de erros"""
    if use_cached:
        df, message = load_historical_data()
        if df is not None:
            st.info(message)
            if days < 730:
                cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                if 'date' in df.columns:
                    df = df[df['date'] >= cutoff_date]
                    st.info(f"📊 Filtrado para {len(df)} jogos dos últimos {days} dias")
            return df
    
    # Teste de conexão antes de começar
    conn_ok, conn_msg = test_api_connection()
    if not conn_ok:
        st.error(f"❌ Problema de conectividade: {conn_msg}")
        st.info("💡 Tente usar 'Usar dados em cache' ou verificar sua internet")
        return pd.DataFrame()
    
    all_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    consecutive_errors = 0
    max_consecutive_errors = 5
    total_requests = 0
    successful_requests = 0
    
    for i in range(days):
        date = datetime.now() - timedelta(days=i+1)
        date_str = date.strftime('%Y-%m-%d')
        status_text.text(f"📊 Coletando dados ML: {date.strftime('%d/%m/%Y')} ({i+1}/{days})")
        
        try:
            fixtures = get_fixtures_with_retry(date_str)
            total_requests += 1
            
            if fixtures:
                successful_requests += 1
                consecutive_errors = 0
                
                for match in fixtures:
                    try:
                        if match['fixture']['status']['short'] == 'FT' and match.get('score', {}).get('halftime'):
                            match_data = extract_match_features(match)
                            if match_data:
                                all_data.append(match_data)
                    except Exception:
                        continue
                        
            else:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    st.warning(f"⚠️ Muitos erros consecutivos ({consecutive_errors}). Parando coleta.")
                    break
        
        except Exception:
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                st.warning(f"⚠️ Muitos erros consecutivos. Parando coleta.")
                break
        
        progress_bar.progress((i+1)/days)
        
        if i < days - 1:
            time.sleep(0.5)
    
    progress_bar.empty()
    status_text.empty()
    
    if total_requests > 0:
        success_rate = (successful_requests / total_requests) * 100
        st.info(f"📊 Coleta concluída: {successful_requests}/{total_requests} dias ({success_rate:.1f}% sucesso)")
        st.info(f"🎯 Total de jogos coletados: {len(all_data)}")
    
    if len(all_data) < 100:
        st.warning("⚠️ Poucos dados coletados. Recomendo usar dados em cache ou tentar novamente.")
    
    return pd.DataFrame(all_data)

def extract_match_features(match):
    """Extrai features para ML com tratamento de erro"""
    try:
        home_team = match['teams']['home']['name']
        away_team = match['teams']['away']['name']
        league_id = match['league']['id']
        league_name = match['league']['name']
        country = match['league']['country']
        
        ht_home = match['score']['halftime']['home']
        ht_away = match['score']['halftime']['away']
        over_05 = 1 if (ht_home + ht_away) > 0 else 0
        
        features = {
            'date': match['fixture']['date'][:10],
            'timestamp': match['fixture']['timestamp'],
            'league_id': league_id,
            'league_name': league_name,
            'country': country,
            'home_team': home_team,
            'away_team': away_team,
            'home_team_id': match['teams']['home']['id'],
            'away_team_id': match['teams']['away']['id'],
            'ht_home_goals': ht_home,
            'ht_away_goals': ht_away,
            'ht_total_goals': ht_home + ht_away,
            'over_05': over_05,
            'venue': match['fixture']['venue']['name'] if match['fixture']['venue'] else 'Unknown',
            'referee': match['fixture']['referee'] if match['fixture']['referee'] else 'Unknown'
        }
        
        return features
    except Exception:
        return None

def prepare_ml_features(df):
    """Prepara features avançadas para o modelo ML incluindo coeficiente de variação e combined score"""
    if 'over_05' not in df.columns:
        if 'ht_home_goals' in df.columns and 'ht_away_goals' in df.columns:
            df['over_05'] = (df['ht_home_goals'] + df['ht_away_goals']) > 0
        elif 'ht_home' in df.columns and 'ht_away' in df.columns:
            df['over_05'] = (df['ht_home'] + df['ht_away']) > 0
    
    if 'ht_total_goals' not in df.columns:
        if 'ht_home_goals' in df.columns and 'ht_away_goals' in df.columns:
            df['ht_total_goals'] = df['ht_home_goals'] + df['ht_away_goals']
        elif 'ht_home' in df.columns and 'ht_away' in df.columns:
            df['ht_total_goals'] = df['ht_home'] + df['ht_away']
    
    team_stats = {}
    
    for idx, row in df.iterrows():
        home_team = row['home_team_id']
        away_team = row['away_team_id']
        
        for team_id in [home_team, away_team]:
            if team_id not in team_stats:
                team_stats[team_id] = {
                    'games': 0, 
                    'over_05': 0, 
                    'over_05_binary': 0,
                    'goals_scored': 0, 
                    'goals_conceded': 0,
                    'goals_capped': 0,
                    'home_games': 0, 
                    'home_over': 0, 
                    'home_over_binary': 0,
                    'home_goals': 0,
                    'home_goals_capped': 0,
                    'away_games': 0, 
                    'away_over': 0,
                    'away_over_binary': 0,
                    'away_goals': 0,
                    'away_goals_capped': 0,
                    'goals_list': [],
                    'over_list': [],
                    'extreme_games': 0
                }
    
    features = []
    
    for idx, row in df.iterrows():
        home_id = row['home_team_id']
        away_id = row['away_team_id']
        
        home_stats = team_stats[home_id]
        home_over_rate = home_stats['over_05'] / max(home_stats['games'], 1)
        home_over_rate_binary = home_stats['over_05_binary'] / max(home_stats['games'], 1)
        home_avg_goals = home_stats['goals_scored'] / max(home_stats['games'], 1)
        home_avg_goals_capped = home_stats['goals_capped'] / max(home_stats['games'], 1)
        home_home_over_rate = home_stats['home_over'] / max(home_stats['home_games'], 1)
        home_home_over_rate_binary = home_stats['home_over_binary'] / max(home_stats['home_games'], 1)
        
        away_stats = team_stats[away_id]
        away_over_rate = away_stats['over_05'] / max(away_stats['games'], 1)
        away_over_rate_binary = away_stats['over_05_binary'] / max(away_stats['games'], 1)
        away_avg_goals = away_stats['goals_scored'] / max(away_stats['games'], 1)
        away_avg_goals_capped = away_stats['goals_capped'] / max(away_stats['games'], 1)
        away_away_over_rate = away_stats['away_over'] / max(away_stats['away_games'], 1)
        away_away_over_rate_binary = away_stats['away_over_binary'] / max(away_stats['away_games'], 1)
        
        league_games = df[df['league_id'] == row['league_id']]
        league_over_rate = league_games['over_05'].mean() if len(league_games) > 0 else 0.5
        league_over_rate_binary = (league_games['over_05'] > 0).mean() if len(league_games) > 0 else 0.5
        
        if len(home_stats['goals_list']) > 1:
            home_goals_cv = np.std(home_stats['goals_list']) / (np.mean(home_stats['goals_list']) + 0.01)
            home_consistency = 1 / (1 + home_goals_cv)
        else:
            home_consistency = 0.5
            home_goals_cv = 1.0
            
        if len(away_stats['goals_list']) > 1:
            away_goals_cv = np.std(away_stats['goals_list']) / (np.mean(away_stats['goals_list']) + 0.01)
            away_consistency = 1 / (1 + away_goals_cv)
        else:
            away_consistency = 0.5
            away_goals_cv = 1.0
        
        home_strength_binary = home_over_rate_binary * home_avg_goals_capped * home_consistency
        away_strength_binary = away_over_rate_binary * away_avg_goals_capped * away_consistency
        combined_score_binary = home_strength_binary + away_strength_binary
        
        home_recent = home_stats['over_list'][-5:] if len(home_stats['over_list']) >= 5 else home_stats['over_list']
        away_recent = away_stats['over_list'][-5:] if len(away_stats['over_list']) >= 5 else away_stats['over_list']
        
        home_momentum = sum([1 if x > 0 else 0 for x in home_recent]) / len(home_recent) if home_recent else home_over_rate_binary
        away_momentum = sum([1 if x > 0 else 0 for x in away_recent]) / len(away_recent) if away_recent else away_over_rate_binary
        
        feature_row = {
            'home_over_rate': home_over_rate,
            'home_avg_goals': home_avg_goals,
            'home_home_over_rate': home_home_over_rate,
            'away_over_rate': away_over_rate,
            'away_avg_goals': away_avg_goals,
            'away_away_over_rate': away_away_over_rate,
            'league_over_rate': league_over_rate,
            'combined_over_rate': (home_over_rate + away_over_rate) / 2,
            'combined_goals': home_avg_goals + away_avg_goals,
            'home_over_rate_binary': home_over_rate_binary,
            'home_avg_goals_capped': home_avg_goals_capped,
            'home_home_over_rate_binary': home_home_over_rate_binary,
            'away_over_rate_binary': away_over_rate_binary,
            'away_avg_goals_capped': away_avg_goals_capped,
            'away_away_over_rate_binary': away_away_over_rate_binary,
            'league_over_rate_binary': league_over_rate_binary,
            'combined_over_rate_binary': (home_over_rate_binary + away_over_rate_binary) / 2,
            'combined_goals_capped': home_avg_goals_capped + away_avg_goals_capped,
            'home_goals_cv': home_goals_cv,
            'away_goals_cv': away_goals_cv,
            'home_consistency': home_consistency,
            'away_consistency': away_consistency,
            'consistency_avg': (home_consistency + away_consistency) / 2,
            'consistency_diff': abs(home_consistency - away_consistency),
            'combined_score_binary': combined_score_binary,
            'home_strength_binary': home_strength_binary,
            'away_strength_binary': away_strength_binary,
            'home_momentum': home_momentum,
            'away_momentum': away_momentum,
            'momentum_sum': home_momentum + away_momentum,
            'momentum_diff': abs(home_momentum - away_momentum),
            'momentum_avg': (home_momentum + away_momentum) / 2,
            'target': row['over_05']
        }
        
        features.append(feature_row)
        
        ht_home_goals = row.get('ht_home_goals', row.get('ht_home', 0))
        ht_away_goals = row.get('ht_away_goals', row.get('ht_away', 0))
        ht_total = ht_home_goals + ht_away_goals
        
        team_stats[home_id]['games'] += 1
        team_stats[home_id]['over_05'] += row['over_05']
        team_stats[home_id]['over_05_binary'] += 1 if row['over_05'] > 0 else 0
        team_stats[home_id]['goals_scored'] += ht_home_goals
        team_stats[home_id]['goals_capped'] += min(ht_home_goals, 1)
        team_stats[home_id]['goals_conceded'] += ht_away_goals
        team_stats[home_id]['home_games'] += 1
        team_stats[home_id]['home_over'] += row['over_05']
        team_stats[home_id]['home_over_binary'] += 1 if row['over_05'] > 0 else 0
        team_stats[home_id]['home_goals'] += ht_home_goals
        team_stats[home_id]['home_goals_capped'] += min(ht_home_goals, 1)
        team_stats[home_id]['goals_list'].append(ht_home_goals)
        team_stats[home_id]['over_list'].append(row['over_05'])
        if ht_total > 2:
            team_stats[home_id]['extreme_games'] += 1
        
        team_stats[away_id]['games'] += 1
        team_stats[away_id]['over_05'] += row['over_05']
        team_stats[away_id]['over_05_binary'] += 1 if row['over_05'] > 0 else 0
        team_stats[away_id]['goals_scored'] += ht_away_goals
        team_stats[away_id]['goals_capped'] += min(ht_away_goals, 1)
        team_stats[away_id]['goals_conceded'] += ht_home_goals
        team_stats[away_id]['away_games'] += 1
        team_stats[away_id]['away_over'] += row['over_05']
        team_stats[away_id]['away_over_binary'] += 1 if row['over_05'] > 0 else 0
        team_stats[away_id]['away_goals'] += ht_away_goals
        team_stats[away_id]['away_goals_capped'] += min(ht_away_goals, 1)
        team_stats[away_id]['goals_list'].append(ht_away_goals)
        team_stats[away_id]['over_list'].append(row['over_05'])
        if ht_total > 2:
            team_stats[away_id]['extreme_games'] += 1
    
    return pd.DataFrame(features), team_stats

def train_ml_model_robust(df):
    """Versão robusta do treinamento com verificações"""
    
    # Verificar se temos dados suficientes
    if len(df) < 100:
        st.error("❌ Dados insuficientes para treinamento (mínimo 100 jogos)")
        st.info("💡 Tente coletar mais dias ou use dados em cache")
        return None, None
    
    try:
        st.info("🧠 Preparando features avançadas (coeficiente de variação, combined score, momentum)...")
        features_df, team_stats = prepare_ml_features(df)
        
        league_analysis = analyze_leagues(df)
        
        feature_cols = [col for col in features_df.columns if col != 'target']
        X = features_df[feature_cols]
        y = features_df['target']
        
        st.info(f"📊 Total de features: {len(feature_cols)}")
        st.info(f"🎯 Features incluem: Coeficiente de Variação, Combined Score, Momentum, Outliers")
        
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42, 
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.08,
                max_depth=6,
                min_samples_split=3,
                random_state=42
            )
        }
        
        best_model = None
        best_score = 0
        results = {}
        
        st.info("🧠 Treinando modelos avançados...")
        for name, model in models.items():
            with st.spinner(f"Treinando {name}..."):
                model.fit(X_train_scaled, y_train)
                
                val_pred = model.predict(X_val_scaled)
                val_acc = accuracy_score(y_val, val_pred)
                
                test_pred = model.predict(X_test_scaled)
                test_acc = accuracy_score(y_test, test_pred)
                test_prec = precision_score(y_test, test_pred)
                test_rec = recall_score(y_test, test_pred)
                test_f1 = f1_score(y_test, test_pred)
                
                results[name] = {
                    'val_accuracy': val_acc,
                    'test_accuracy': test_acc,
                    'precision': test_prec,
                    'recall': test_rec,
                    'f1_score': test_f1
                }
                
                st.success(f"✅ {name}: F1-Score = {test_f1:.1%} | Acurácia = {test_acc:.1%}")
                
                if test_f1 > best_score:
                    best_score = test_f1
                    best_model = model
        
        model_data = {
            'model': best_model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'team_stats': team_stats,
            'league_analysis': league_analysis,
            'results': results,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': len(df),
            'training_days': len(df['date'].unique()) if 'date' in df.columns else 0,
            'features_count': len(feature_cols),
            'advanced_features': True
        }
        
        st.session_state.trained_model = model_data
        st.session_state.model_trained = True
        
        try:
            for directory in [MODEL_DIR, "/tmp/models"]:
                try:
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    model_path = os.path.join(directory, f"model_advanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
                    joblib.dump(model_data, model_path)
                    st.success(f"💾 Modelo salvo: {model_path}")
                    break
                except Exception:
                    pass
        except Exception:
            pass
        
        return model_data, results
        
    except Exception as e:
        st.error(f"❌ Erro durante o treinamento: {str(e)}")
        st.info("💡 Tente com menos dias de treinamento ou use dados em cache")
        return None, None

def load_latest_model():
    """Carrega o modelo mais recente"""
    try:
        for directory in [MODEL_DIR, "/tmp/models"]:
            if os.path.exists(directory):
                model_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
                if model_files:
                    latest_model = sorted(model_files)[-1]
                    model_path = os.path.join(directory, latest_model)
                    return joblib.load(model_path)
    except Exception:
        pass
    return None

def get_league_context(league_name):
    """Retorna contexto da liga para comparação"""
    league_averages = {
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
    
    default_avg = 60
    
    league_avg = default_avg
    for key, value in league_averages.items():
        if key.lower() in league_name.lower() or league_name.lower() in key.lower():
            league_avg = value
            break
    
    return {
        'league_avg': league_avg,
        'comparison_color': '#38ef7d' if league_avg > 65 else '#667eea' if league_avg > 55 else '#f093fb',
        'comparison_text': 'Liga Over' if league_avg > 65 else 'Liga Equilibrada' if league_avg > 55 else 'Liga Under'
    }

def get_prediction_context(pred, league_context):
    """Gera contexto inteligente da previsão"""
    confidence = pred['confidence']
    league_avg = league_context['league_avg']
    
    if pred['prediction'] == 'OVER 0.5':
        if confidence > league_avg + 15:
            return "🔥 Excelente para Over"
        elif confidence > league_avg + 8:
            return "✅ Muito bom para Over"
        elif confidence > league_avg + 3:
            return "📈 Acima da média da liga"
        elif confidence > league_avg:
            return "📊 Ligeiramente acima"
        else:
            return "⚠️ Abaixo do esperado"
    else:
        if confidence > 75:
            return "❄️ Forte indicação Under"
        elif confidence > 65:
            return "📉 Boa tendência Under"
        else:
            return "🤔 Under com reservas"

def display_prediction_card_clean(pred, index):
    """Exibe card de previsão no estilo simples e limpo"""
    try:
        utc_time = datetime.strptime(pred['kickoff'][:16], '%Y-%m-%dT%H:%M')
        hora_portugal = utc_time.strftime('%H:%M')
    except Exception:
        hora_portugal = pred['kickoff'][11:16]
    
    league_context = get_league_context(pred['league'])
    
    if pred['confidence'] > 80:
        confidence_color = "#38ef7d"
    elif pred['confidence'] > 65:
        confidence_color = "#667eea"
    else:
        confidence_color = "#f093fb"
    
    prediction_context = get_prediction_context(pred, league_context)
    
    diff = pred['confidence'] - league_context['league_avg']
    diff_text = f"+{diff:.0f}%" if diff > 0 else f"{diff:.0f}%"
    diff_color = "#38ef7d" if diff > 5 else "#f093fb" if diff < -5 else "#667eea"
    
    st.markdown(f"""
    <div class="prediction-card-simple">
        <div class="card-header">
            <div class="match-info">
                <h3>⚽ {pred['home_team']} vs {pred['away_team']}</h3>
                <div class="league-time">
                    <span class="league">🏆 {pred['league']} ({pred['country']})</span>
                    <span class="time">🕐 {hora_portugal}</span>
                </div>
            </div>
            <div class="confidence-simple" style="background-color: {confidence_color};">
                {pred['confidence']:.0f}%
            </div>
        </div>
        
        <div class="prediction-info">
            <div class="main-prediction">
                <span class="prediction-label">🎯 Previsão ML:</span>
                <span class="prediction-value">{pred['prediction']}</span>
            </div>
            
            <div class="context-info">
                <div class="context-item">
                    <span class="context-label">📊 Liga (média)</span>
                    <span class="context-value">{league_context['league_avg']:.0f}%</span>
                </div>
                <div class="context-item">
                    <span class="context-label">📈 Vs Liga</span>
                    <span class="context-value" style="color: {diff_color};">
                        {diff_text}
                    </span>
                </div>
                <div class="context-item">
                    <span class="context-label">💡 Análise</span>
                    <span class="context-value">{prediction_context}</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_all_games_list_simple(predictions):
    """Exibe lista simplificada de todos os jogos"""
    if not predictions:
        return
    
    st.markdown("""
    <div class="games-list-simple">
        <h3>📋 Todos os Jogos do Dia</h3>
    """, unsafe_allow_html=True)
    
    for pred in predictions:
        try:
            utc_time = datetime.strptime(pred['kickoff'][:16], '%Y-%m-%dT%H:%M')
            hora = utc_time.strftime('%H:%M')
        except Exception:
            hora = pred['kickoff'][11:16]
        
        prediction_icon = "🟢" if pred['prediction'] == 'OVER 0.5' else "🔴"
        
        st.markdown(f"""
        <div class="game-item-simple">
            <div class="game-teams-simple">
                {prediction_icon} {pred['home_team']} vs {pred['away_team']} ({hora})
            </div>
            <div class="game-confidence-simple">
                {pred['confidence']:.0f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def predict_matches(fixtures, model_data):
    """Faz previsões para os jogos do dia usando features avançadas"""
    predictions = []
    
    if not model_data:
        return predictions
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = model_data['feature_cols']
    team_stats = model_data['team_stats']
    league_analysis = model_data.get('league_analysis', {})
    
    for fixture in fixtures:
        if fixture['fixture']['status']['short'] not in ['NS', 'TBD']:
            continue
        
        try:
            home_id = fixture['teams']['home']['id']
            away_id = fixture['teams']['away']['id']
            league_name = fixture['league']['name']
            
            if home_id not in team_stats or away_id not in team_stats:
                continue
            
            home_stats = team_stats[home_id]
            away_stats = team_stats[away_id]
            
            league_info = league_analysis.get(league_name, {})
            league_over_rate = league_info.get('over_rate', 0.5)
            
            home_over_rate = home_stats['over_05'] / max(home_stats['games'], 1)
            home_over_rate_binary = home_stats['over_05_binary'] / max(home_stats['games'], 1)
            home_avg_goals = home_stats['goals_scored'] / max(home_stats['games'], 1)
            home_avg_goals_capped = home_stats['goals_capped'] / max(home_stats['games'], 1)
            
            away_over_rate = away_stats['over_05'] / max(away_stats['games'], 1)
            away_over_rate_binary = away_stats['over_05_binary'] / max(away_stats['games'], 1)
            away_avg_goals = away_stats['goals_scored'] / max(away_stats['games'], 1)
            away_avg_goals_capped = away_stats['goals_capped'] / max(away_stats['games'], 1)
            
            if len(home_stats['goals_list']) > 1:
                home_goals_cv = np.std(home_stats['goals_list']) / (np.mean(home_stats['goals_list']) + 0.01)
                home_consistency = 1 / (1 + home_goals_cv)
            else:
                home_consistency = 0.5
                home_goals_cv = 1.0
                
            if len(away_stats['goals_list']) > 1:
                away_goals_cv = np.std(away_stats['goals_list']) / (np.mean(away_stats['goals_list']) + 0.01)
                away_consistency = 1 / (1 + away_goals_cv)
            else:
                away_consistency = 0.5
                away_goals_cv = 1.0
            
            home_strength_binary = home_over_rate_binary * home_avg_goals_capped * home_consistency
            away_strength_binary = away_over_rate_binary * away_avg_goals_capped * away_consistency
            combined_score_binary = home_strength_binary + away_strength_binary
            
            home_recent = home_stats['over_list'][-5:] if len(home_stats['over_list']) >= 5 else home_stats['over_list']
            away_recent = away_stats['over_list'][-5:] if len(away_stats['over_list']) >= 5 else away_stats['over_list']
            
            home_momentum = sum([1 if x > 0 else 0 for x in home_recent]) / len(home_recent) if home_recent else home_over_rate_binary
            away_momentum = sum([1 if x > 0 else 0 for x in away_recent]) / len(away_recent) if away_recent else away_over_rate_binary
            
            features = {}
            for col in feature_cols:
                if col == 'home_over_rate':
                    features[col] = home_over_rate
                elif col == 'home_avg_goals':
                    features[col] = home_avg_goals
                elif col == 'home_home_over_rate':
                    features[col] = home_stats['home_over'] / max(home_stats['home_games'], 1)
                elif col == 'away_over_rate':
                    features[col] = away_over_rate
                elif col == 'away_avg_goals':
                    features[col] = away_avg_goals
                elif col == 'away_away_over_rate':
                    features[col] = away_stats['away_over'] / max(away_stats['away_games'], 1)
                elif col == 'league_over_rate':
                    features[col] = league_over_rate
                elif col == 'combined_over_rate':
                    features[col] = (home_over_rate + away_over_rate) / 2
                elif col == 'combined_goals':
                    features[col] = home_avg_goals + away_avg_goals
                elif col == 'home_consistency':
                    features[col] = home_consistency
                elif col == 'away_consistency':
                    features[col] = away_consistency
                elif col == 'combined_score_binary':
                    features[col] = combined_score_binary
                elif col == 'home_momentum':
                    features[col] = home_momentum
                elif col == 'away_momentum':
                    features[col] = away_momentum
                else:
                    features[col] = 0.5
            
            X = pd.DataFrame([features])[feature_cols]
            X_scaled = scaler.transform(X)
            
            pred_proba = model.predict_proba(X_scaled)[0]
            pred_class = model.predict(X_scaled)[0]
            confidence = max(pred_proba) * 100
            
            league_trend = league_info.get('trend', 'BALANCED')
            if league_trend == 'OVER' and pred_class == 1:
                confidence = min(confidence * 1.05, 95)
            elif league_trend == 'UNDER' and pred_class == 1:
                confidence = confidence * 0.95
            
            prediction = {
                'fixture': fixture,
                'home_team': fixture['teams']['home']['name'],
                'away_team': fixture['teams']['away']['name'],
                'league': league_name,
                'league_trend': league_trend,
                'country': fixture['league']['country'],
                'kickoff': fixture['fixture']['date'],
                'prediction': 'OVER 0.5' if pred_class == 1 else 'UNDER 0.5',
                'confidence': confidence,
                'probability_over': pred_proba[1] * 100,
                'probability_under': pred_proba[0] * 100,
                'home_stats': home_stats,
                'away_stats': away_stats,
                'advanced_features': {
                    'home_consistency': home_consistency,
                    'away_consistency': away_consistency,
                    'combined_score': combined_score_binary,
                    'home_momentum': home_momentum,
                    'away_momentum': away_momentum
                }
            }
            
            predictions.append(prediction)
            
        except Exception:
            continue
    
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return predictions

def analyze_leagues(df):
    """Analisa tendências por liga"""
    league_analysis = {}
    
    for league_id in df['league_id'].unique():
        league_data = df[df['league_id'] == league_id]
        
        if len(league_data) >= 10:
            over_rate = league_data['over_05'].mean()
            avg_goals = league_data['ht_total_goals'].mean()
            
            if over_rate >= 0.70:
                classification = "🔥 LIGA OVER FORTE"
            elif over_rate >= 0.55:
                classification = "📈 LIGA OVER"
            elif over_rate <= 0.30:
                classification = "❄️ LIGA UNDER FORTE"
            elif over_rate <= 0.45:
                classification = "📉 LIGA UNDER"
            else:
                classification = "⚖️ LIGA EQUILIBRADA"
            
            league_analysis[league_data.iloc[0]['league_name']] = {
                'country': league_data.iloc[0]['country'],
                'total_games': len(league_data),
                'over_rate': over_rate,
                'avg_goals_ht': avg_goals,
                'classification': classification,
                'trend': 'OVER' if over_rate > 0.5 else 'UNDER'
            }
    
    return league_analysis

def main():
    st.markdown("""
    <div class="main-header">
        <h1>⚽ HT Goals AI Engine</h1>
        <p>🚀 Powered by Predictive Modeling & Advanced Metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Teste de conectividade inicial
    conn_ok, conn_msg = test_api_connection()
    
    with st.sidebar:
        st.title("⚙️ Configurações")
        
        # Status da API com indicador visual
        if conn_ok:
            api_ok, requests_left, api_status = check_api_status()
            
            if not api_ok:
                st.markdown(f'<div class="connection-status" style="color: #dc3545;">❌ {api_status}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="connection-status" style="color: #28a745;">✅ API conectada</div>', unsafe_allow_html=True)
                if requests_left > 0:
                    st.info(f"📊 Requests restantes hoje: {requests_left}")
                else:
                    st.warning(f"⚠️ Sem requests restantes hoje!")
        else:
            st.markdown(f'<div class="connection-status" style="color: #dc3545;">❌ {conn_msg}</div>', unsafe_allow_html=True)
        
        selected_date = st.date_input(
            "📅 Data para análise:",
            value=datetime.now().date()
        )
        
        st.subheader("🤖 Machine Learning Avançado")
        
        days_training = st.slider(
            "📊 Dias para treinamento:",
            min_value=15,
            max_value=730,
            value=150  # Valor padrão menor para evitar erros
        )
        
        use_cache = st.checkbox(
            "💾 Usar dados em cache",
            value=True,
            help="Recomendado: Usar dados históricos salvos localmente"
        )
        
        st.subheader("🧠 Features Avançadas")
        st.info("""
        ✅ **Coeficiente de Variação**
        ✅ **Combined Score**
        ✅ **Momentum Analysis**
        ✅ **Outlier Detection**
        ✅ **League Consistency**
        """)
        
        # Status do modelo
        if st.session_state.model_trained and st.session_state.trained_model:
            model_data = st.session_state.trained_model
            st.success("✅ Modelo carregado")
            st.info(f"📅 Treinado em: {model_data['training_date']}")
            st.info(f"📊 Amostras: {model_data['total_samples']}")
            
            if model_data.get('advanced_features', False):
                st.success("🧠 Modelo com features avançadas")
                st.info(f"🎯 Total features: {model_data.get('features_count', 'N/A')}")
            
            if 'results' in model_data:
                results = model_data['results']
                best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
                st.info(f"🏆 Melhor modelo: {best_model[0]}")
                st.info(f"📈 F1-Score: {best_model[1]['f1_score']:.1%}")
        else:
            model_data = load_latest_model()
            if model_data:
                st.session_state.trained_model = model_data
                st.session_state.model_trained = True
                st.success("✅ Modelo carregado do arquivo")
                if model_data.get('advanced_features', False):
                    st.success("🧠 Modelo com features avançadas")
            else:
                st.warning("⚠️ Nenhum modelo encontrado")
    
    # Tabs principais
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Previsões do Dia",
        "📊 Análise por Liga", 
        "🤖 Treinar Modelo Avançado",
        "📈 Performance ML"
    ])
    
    with tab1:
        st.header(f"🎯 Previsões para {selected_date.strftime('%d/%m/%Y')}")
        
        model_data = None
        
        if st.session_state.get('model_trained', False) and st.session_state.get('trained_model'):
            model_data = st.session_state.trained_model
            st.success("✅ Modelo carregado da sessão")
            if model_data.get('advanced_features', False):
                st.info("🧠 Usando modelo com features avançadas")
        else:
            model_data = load_latest_model()
            if model_data:
                st.session_state.trained_model = model_data
                st.session_state.model_trained = True
                st.success("✅ Modelo carregado do arquivo")
        
        if not model_data:
            st.warning("⚠️ Treine um modelo primeiro na aba 'Treinar Modelo Avançado'")
            
            if st.button("🔄 Tentar carregar modelo novamente"):
                st.rerun()
        else:
            st.info(f"🤖 Modelo: {model_data.get('training_date', 'Unknown')}")
            st.info(f"📊 Times no banco: {len(model_data.get('team_stats', {}))}")
            
            date_str = selected_date.strftime('%Y-%m-%d')
            
            with st.spinner("🔍 Buscando jogos do dia..."):
                fixtures = get_fixtures_cached_robust(date_str)
            
            if not fixtures:
                st.info("📅 Nenhum jogo encontrado para esta data")
                if not conn_ok:
                    st.error("❌ Verifique sua conexão com a internet")
            else:
                with st.spinner("🤖 Aplicando Machine Learning Avançado..."):
                    predictions = predict_matches(fixtures, model_data)
                
                if predictions:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_games = len(predictions)
                    high_confidence = len([p for p in predictions if p['confidence'] > 70])
                    over_predictions = len([p for p in predictions if p['prediction'] == 'OVER 0.5'])
                    avg_confidence = sum([p['confidence'] for p in predictions]) / len(predictions)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>🎮 Total de Jogos</h3>
                            <h1>{total_games}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>🎯 Alta Confiança</h3>
                            <h1>{high_confidence}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📈 Over 0.5</h3>
                            <h1>{over_predictions}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>💯 Confiança Média</h3>
                            <h1>{avg_confidence:.1f}%</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.subheader("🏆 Melhores Apostas (Análise Avançada)")
                    
                    best_bets = [p for p in predictions if p['prediction'] == 'OVER 0.5' and p['confidence'] > 65]
                    best_bets.sort(key=lambda x: x['confidence'], reverse=True)
                    
                    if best_bets:
                        for i, pred in enumerate(best_bets[:5]):
                            display_prediction_card_clean(pred, i)
                    else:
                        st.info("🤷 Nenhuma aposta OVER 0.5 com boa confiança encontrada hoje")
                    
                    display_all_games_list_simple(predictions)
                
                else:
                    st.info("🤷 Nenhuma previsão disponível (times sem dados históricos)")
    
    with tab2:
        st.header("📊 Análise de Ligas")
        
        if 'trained_model' in st.session_state:
            model_data = st.session_state.trained_model
        else:
            model_data = load_latest_model()
        
        if model_data:
            df = collect_historical_data_robust(days=15, use_cached=True)
            
            if not df.empty:
                league_analysis = analyze_leagues(df)
                
                over_leagues = {k: v for k, v in league_analysis.items() if v['trend'] == 'OVER'}
                under_leagues = {k: v for k, v in league_analysis.items() if v['trend'] == 'UNDER'}
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔥 Ligas OVER (> 50%)")
                    for league, stats in sorted(over_leagues.items(), key=lambda x: x[1]['over_rate'], reverse=True):
                        st.markdown(f"""
                        <div class="league-card">
                            <h4>{league}</h4>
                            <p>{stats['classification']}</p>
                            <p>📊 Taxa Over: {stats['over_rate']:.1%}</p>
                            <p>⚽ Média gols HT: {stats['avg_goals_ht']:.2f}</p>
                            <p>🎮 Jogos analisados: {stats['total_games']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.subheader("❄️ Ligas UNDER (< 50%)")
                    for league, stats in sorted(under_leagues.items(), key=lambda x: x[1]['over_rate']):
                        st.markdown(f"""
                        <div class="league-card">
                            <h4>{league}</h4>
                            <p>{stats['classification']}</p>
                            <p>📊 Taxa Over: {stats['over_rate']:.1%}</p>
                            <p>⚽ Média gols HT: {stats['avg_goals_ht']:.2f}</p>
                            <p>🎮 Jogos analisados: {stats['total_games']}</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("🤖 Treine um modelo primeiro")
    
    with tab3:
        st.header("🤖 Treinar Modelo ML Avançado")
        
        # Aviso sobre conectividade
        if not conn_ok:
            st.error(f"❌ {conn_msg}")
            st.info("💡 **Recomendação**: Marque 'Usar dados em cache' para treinar com dados locais")
        
        st.success("""
        🧠 **FEATURES AVANÇADAS INCLUÍDAS:**
        
        ✅ **Coeficiente de Variação**: Mede consistência dos times  
        ✅ **Combined Score**: Score combinado com múltiplos fatores  
        ✅ **Momentum Analysis**: Análise dos últimos 5 jogos  
        ✅ **Outlier Detection**: Detecção de jogos extremos  
        ✅ **League Consistency**: Consistência por liga  
        ✅ **Efficiency Metrics**: Eficiência de conversão em Over  
        """)
        
        st.info(f"""
        O modelo será treinado com **25+ features avançadas**:
        - **70%** dos dados para treinamento
        - **15%** para validação  
        - **15%** para teste final
        - **{days_training} dias** de dados históricos
        """)
        
        # Aviso sobre quantidade de dados
        if days_training > 365:
            st.warning("⚠️ Muitos dias podem causar timeout. Recomendado: 150-365 dias")
        
        if st.button("🚀 Iniciar Treinamento Avançado", type="primary"):
            with st.spinner(f"📊 Coletando {days_training} dias de dados históricos..."):
                df = collect_historical_data_robust(days=days_training, use_cached=use_cache)
            
            if df.empty:
                st.error("❌ Não foi possível coletar dados")
                st.info("💡 **Soluções:**")
                st.info("- Marque 'Usar dados em cache'")
                st.info("- Reduza os dias de treinamento")
                st.info("- Verifique sua conexão com internet")
            else:
                st.success(f"✅ {len(df)} jogos coletados")
                
                with st.spinner("🧠 Treinando modelos avançados..."):
                    model_data, results = train_ml_model_robust(df)
                
                if model_data and results:
                    st.success("✅ Modelo avançado treinado com sucesso!")
                    
                    st.subheader("📊 Resultados do Treinamento Avançado")
                    
                    for model_name, metrics in results.items():
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.metric(f"{model_name}", "")
                        with col2:
                            st.metric("Validação", f"{metrics['val_accuracy']:.1%}")
                        with col3:
                            st.metric("Teste", f"{metrics['test_accuracy']:.1%}")
                        with col4:
                            st.metric("Precisão", f"{metrics['precision']:.1%}")
                        with col5:
                            st.metric("F1-Score", f"{metrics['f1_score']:.1%}")
                    
                    if hasattr(model_data['model'], 'feature_importances_'):
                        st.subheader("🎯 Features Mais Importantes")
                        
                        feature_importance = pd.DataFrame({
                            'feature': model_data['feature_cols'],
                            'importance': model_data['model'].feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        top_features = feature_importance.head(20)
                        st.bar_chart(top_features.set_index('feature')['importance'])
                        
                        advanced_features = top_features[top_features['feature'].str.contains('consistency|combined|momentum|cv|efficiency')]
                        if not advanced_features.empty:
                            st.success("🧠 Features avançadas entre as mais importantes:")
                            for _, row in advanced_features.iterrows():
                                st.write(f"• **{row['feature']}**: {row['importance']:.3f}")
                else:
                    st.error("❌ Falha no treinamento")
    
    with tab4:
        st.header("📈 Performance do Modelo Avançado")
        
        if 'trained_model' in st.session_state:
            model_data = st.session_state.trained_model
        else:
            model_data = load_latest_model()
        
        if model_data and 'results' in model_data:
            if model_data.get('advanced_features', False):
                st.success("🧠 Modelo com Features Avançadas Ativo")
                st.info(f"📊 Total de features: {model_data.get('features_count', 'N/A')}")
            else:
                st.warning("⚠️ Modelo básico (sem features avançadas)")
            
            results = model_data['results']
            
            best_model_name = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
            best_metrics = results[best_model_name]
            
            st.subheader(f"🏆 Melhor Modelo: {best_model_name}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                accuracy = best_metrics['test_accuracy'] * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎯 Acurácia</h3>
                    <h1 class="{'accuracy-high' if accuracy > 65 else 'accuracy-medium'}">{accuracy:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                precision = best_metrics['precision'] * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💎 Precisão</h3>
                    <h1 class="{'accuracy-high' if precision > 65 else 'accuracy-medium'}">{precision:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                recall = best_metrics['recall'] * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 Recall</h3>
                    <h1 class="{'accuracy-high' if recall > 65 else 'accuracy-medium'}">{recall:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                f1 = best_metrics['f1_score'] * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🏅 F1-Score</h3>
                    <h1 class="{'accuracy-high' if f1 > 65 else 'accuracy-medium'}">{f1:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            st.subheader("📊 Performance Histórica do Modelo")
            
            if 'total_samples' in model_data:
                total_analyzed = model_data['total_samples']
                accuracy_rate = best_metrics['test_accuracy'] * 100
                correct_predictions = int(total_analyzed * best_metrics['test_accuracy'])
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📅 Jogos Analisados</h3>
                        <h1>{total_analyzed:,}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>✅ Acertos</h3>
                        <h1>{correct_predictions:,}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📈 Taxa de Acerto</h3>
                        <h1>{accuracy_rate:.1f}%</h1>
                    </div>
                    """, unsafe_allow_html=True)
            
            if model_data.get('advanced_features', False):
                st.subheader("🧠 Análise de Features Avançadas")
                
                if hasattr(model_data['model'], 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': model_data['feature_cols'],
                        'importance': model_data['model'].feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    basic_features = feature_importance[feature_importance['feature'].str.contains('home_over_rate|away_over_rate|league_over_rate|combined_over_rate|combined_goals')]
                    consistency_features = feature_importance[feature_importance['feature'].str.contains('consistency|cv')]
                    combined_score_features = feature_importance[feature_importance['feature'].str.contains('combined_score|strength|efficiency')]
                    momentum_features = feature_importance[feature_importance['feature'].str.contains('momentum')]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🎯 Top Features Coeficiente de Variação:**")
                        if not consistency_features.empty:
                            for _, row in consistency_features.head(5).iterrows():
                                st.write(f"• {row['feature']}: {row['importance']:.3f}")
                        else:
                            st.write("Nenhuma feature de consistência encontrada")
                        
                        st.write("**📈 Top Features Combined Score:**")
                        if not combined_score_features.empty:
                            for _, row in combined_score_features.head(5).iterrows():
                                st.write(f"• {row['feature']}: {row['importance']:.3f}")
                        else:
                            st.write("Nenhuma feature de combined score encontrada")
                    
                    with col2:
                        st.write("**🔥 Top Features Momentum:**")
                        if not momentum_features.empty:
                            for _, row in momentum_features.head(5).iterrows():
                                st.write(f"• {row['feature']}: {row['importance']:.3f}")
                        else:
                            st.write("Nenhuma feature de momentum encontrada")
                        
                        st.write("**⚡ Top Features Básicas:**")
                        if not basic_features.empty:
                            for _, row in basic_features.head(5).iterrows():
                                st.write(f"• {row['feature']}: {row['importance']:.3f}")
            
            with st.expander("📚 Entenda as Métricas"):
                st.write("""
                **Métricas de Performance:**
                - **Acurácia**: Percentual total de acertos do modelo
                - **Precisão**: Quando o modelo prevê OVER 0.5, quantas vezes acerta
                - **Recall**: Dos jogos que foram OVER 0.5, quantos o modelo identificou
                - **F1-Score**: Média harmônica entre Precisão e Recall (métrica principal)
                
                **Features Avançadas:**
                - **Coeficiente de Variação**: Mede a consistência dos times (menor variação = mais consistente)
                - **Combined Score**: Score que combina taxa Over, média de gols e consistência
                - **Momentum**: Análise dos últimos 5 jogos para detectar tendências
                - **Efficiency**: Relação entre gols marcados e taxa de Over
                """)
            
            st.subheader("ℹ️ Informações do Modelo")
            advanced_status = "🧠 Avançado" if model_data.get('advanced_features', False) else "📊 Básico"
            st.info(f"""
            - **Tipo**: {advanced_status}
            - **Data de Treinamento**: {model_data['training_date']}
            - **Total de Jogos Analisados**: {model_data['total_samples']:,}
            - **Times no Banco de Dados**: {len(model_data['team_stats']):,}
            - **Algoritmo**: {best_model_name}
            - **Total de Features**: {model_data.get('features_count', len(model_data.get('feature_cols', [])))}
            """)
            
            if model_data.get('advanced_features', False):
                st.success("""
                🎯 **Features Avançadas Ativas:**
                ✅ Coeficiente de Variação para consistência  
                ✅ Combined Score com múltiplos fatores  
                ✅ Análise de momentum dos últimos jogos  
                ✅ Detecção de outliers e jogos extremos  
                ✅ Métricas de eficiência de conversão  
                """)
        else:
            st.info("🤖 Nenhum modelo treinado ainda")
            st.write("Para começar:")
            st.write("1. Vá para a aba 'Treinar Modelo Avançado'")
            st.write("2. Marque 'Usar dados em cache' (recomendado)")
            st.write("3. Clique em 'Iniciar Treinamento Avançado'")
            st.write("4. Aguarde o modelo ser treinado com todas as features avançadas")

if __name__ == "__main__":
    main()
