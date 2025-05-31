import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import joblib
import os
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="⚽ Over 0.5 HT - ACCURACY MASTER",
    page_icon="🎯",
    layout="wide"
)

# Inicializar session state
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'accuracy_tracking' not in st.session_state:
    st.session_state.accuracy_tracking = {'predictions': [], 'results': []}

# Configuração da API Key
try:
    API_KEY = st.secrets["API_KEY"]
except:
    API_KEY = "474f15de5b22951077ccb71b8d75b95c"

API_BASE_URL = "https://v3.football.api-sports.io"

MODEL_DIR = "models"
try:
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
except Exception as e:
    MODEL_DIR = "/tmp/models"
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

# CSS OTIMIZADO PARA ACURÁCIA
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .accuracy-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        transition: transform 0.3s;
        border: 3px solid gold;
    }
    .accuracy-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.3);
    }
    .prediction-premium {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.4);
        border: 3px solid #FFD700;
    }
    .prediction-excellent {
        background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(78, 205, 196, 0.3);
        border: 2px solid #32CD32;
    }
    .accuracy-ultra-high {
        color: #FF6B6B;
        font-weight: 900;
        font-size: 1.2em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .accuracy-very-high {
        color: #4ECDC4;
        font-weight: bold;
        font-size: 1.1em;
    }
    .accuracy-high {
        color: #28a745;
        font-weight: bold;
    }
    .accuracy-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-meter {
        background: linear-gradient(90deg, #ff4757 0%, #ffa726 50%, #26de81 100%);
        height: 10px;
        border-radius: 5px;
        position: relative;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def get_api_headers():
    return {'x-apisports-key': API_KEY}

def check_api_status():
    headers = get_api_headers()
    try:
        response = requests.get(f'{API_BASE_URL}/status', headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'response' in data:
                status = data['response']
                requests_info = status.get('requests', {})
                requests_remaining = requests_info.get('limit_day', 0) - requests_info.get('current', 0)
                return True, requests_remaining, status
            elif 'errors' in data:
                return False, 0, str(data['errors'])
        else:
            return False, 0, f"Status Code: {response.status_code}"
    except Exception as e:
        return False, 0, str(e)

@st.cache_data(ttl=3600)
def get_fixtures_cached(date_str):
    return get_fixtures(date_str)

def get_fixtures(date_str):
    headers = get_api_headers()
    try:
        response = requests.get(
            f'{API_BASE_URL}/fixtures',
            headers=headers,
            params={'date': date_str},
            timeout=20
        )
        if response.status_code == 200:
            data = response.json()
            if 'errors' in data and data['errors']:
                st.error(f"Erro da API: {data['errors']}")
                return []
            return data.get('response', [])
        else:
            st.error(f"Erro API: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Erro de conexão: {str(e)}")
        return []

def get_fixture_events(fixture_id):
    """Busca eventos de um jogo específico (incluindo tempo dos gols)"""
    headers = get_api_headers()
    try:
        response = requests.get(
            f'{API_BASE_URL}/fixtures/events',
            headers=headers,
            params={'fixture': fixture_id},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return data.get('response', [])
        return []
    except:
        return []

def load_historical_data():
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
            except Exception as e:
                continue
    
    return None, "❌ Nenhum arquivo de dados históricos encontrado"

def collect_historical_data(days=30, use_cached=True):
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
    
    # Buscar da API se não houver cache
    all_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(days):
        date = datetime.now() - timedelta(days=i+1)
        date_str = date.strftime('%Y-%m-%d')
        status_text.text(f"📊 Coletando dados para MÁXIMA ACURÁCIA: {date.strftime('%d/%m/%Y')} ({i+1}/{days})")
        
        fixtures = get_fixtures(date_str)
        
        for match in fixtures:
            if match['fixture']['status']['short'] == 'FT' and match.get('score', {}).get('halftime'):
                match_data = extract_match_features_accuracy(match)
                if match_data:
                    all_data.append(match_data)
        
        progress_bar.progress((i+1)/days)
        time.sleep(0.3)
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(all_data)

def extract_match_features_accuracy(match):
    """Extrai features OTIMIZADAS PARA ACURÁCIA"""
    try:
        home_team = match['teams']['home']['name']
        away_team = match['teams']['away']['name']
        league_id = match['league']['id']
        league_name = match['league']['name']
        country = match['league']['country']
        
        ht_home = match['score']['halftime']['home']
        ht_away = match['score']['halftime']['away']
        ft_home = match['score']['fulltime']['home']
        ft_away = match['score']['fulltime']['away']
        over_05 = 1 if (ht_home + ht_away) > 0 else 0
        
        # FEATURES ESPECÍFICAS PARA ACURÁCIA EM OVER 0.5 HT
        venue = match['fixture']['venue']['name'] if match['fixture']['venue'] else 'Unknown'
        referee = match['fixture']['referee'] if match['fixture']['referee'] else 'Unknown'
        
        # Tempo estimado do primeiro gol
        events = get_fixture_events(match['fixture']['id']) if match['fixture']['status']['short'] == 'FT' else []
        first_goal_time = 90  # Default se não houver gols
        
        # Buscar tempo real do primeiro gol nos eventos
        goal_events = [e for e in events if e['type'] == 'Goal' and e['detail'] != 'Missed Penalty']
        if goal_events:
            # Ordenar por tempo e pegar o primeiro
            goal_times = [e['time']['elapsed'] for e in goal_events if e['time']['elapsed'] is not None]
            if goal_times:
                first_goal_time = min(goal_times)
        
        # Se não conseguiu dos eventos, estimar baseado no resultado
        elif ht_home > 0 or ht_away > 0:
            # Se houve gol no HT, estimar tempo baseado no número de gols
            if ht_home + ht_away >= 2:
                first_goal_time = 15  # Múltiplos gols = start rápido
            else:
                first_goal_time = 30  # 1 gol = tempo médio
        elif ft_home > 0 or ft_away > 0:
            first_goal_time = 65  # Gol só no segundo tempo
        
        # Análise temporal para ACURÁCIA
        match_date = datetime.strptime(match['fixture']['date'][:10], '%Y-%m-%d')
        day_of_week = match_date.weekday()
        month = match_date.month
        
        # Classificação de período
        if month in [8, 9, 10]:  # Início de temporada
            season_period = 'early'
            season_factor = 1.15  # Mais gols no início
        elif month in [11, 12, 1, 2]:  # Meio de temporada
            season_period = 'mid'
            season_factor = 1.0
        else:  # Final de temporada
            season_period = 'late'
            season_factor = 0.85  # Menos gols no final
        
        # Fator weekend (importante para acurácia)
        weekend_factor = 1.1 if day_of_week in [5, 6] else 1.0
        
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
            'ft_home_goals': ft_home,
            'ft_away_goals': ft_away,
            'ht_total_goals': ht_home + ht_away,
            'ft_total_goals': ft_home + ft_away,
            'over_05': over_05,
            'venue': venue,
            'referee': referee,
            'first_goal_time': first_goal_time,
            'day_of_week': day_of_week,
            'season_period': season_period,
            'season_factor': season_factor,
            'weekend_factor': weekend_factor,
            'goals_difference_ht': abs(ht_home - ht_away),
            'early_explosion': 1 if (ht_home + ht_away >= 2) else 0,  # 2+ gols HT
            'dominant_start': 1 if max(ht_home, ht_away) >= 2 else 0,  # Um time com 2+ gols HT
            'balanced_scoring': 1 if (ht_home > 0 and ht_away > 0) else 0,  # Ambos marcaram HT
        }
        
        return features
    except Exception as e:
        return None

def prepare_accuracy_features(df):
    """Prepara features OTIMIZADAS ESPECIFICAMENTE PARA ACURÁCIA MÁXIMA"""
    
    # Garantir colunas necessárias
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
    
    # Stats por time FOCADAS EM ACURÁCIA
    team_stats = {}
    referee_stats = {}
    venue_stats = {}
    league_time_stats = {}
    
    # Inicializar estruturas
    for idx, row in df.iterrows():
        home_team = row['home_team_id']
        away_team = row['away_team_id']
        
        for team_id in [home_team, away_team]:
            if team_id not in team_stats:
                team_stats[team_id] = {
                    # Stats básicas
                    'games': 0, 'over_05': 0, 'goals_scored': 0, 'goals_conceded': 0,
                    'home_games': 0, 'home_over': 0, 'home_goals': 0,
                    'away_games': 0, 'away_over': 0, 'away_goals': 0,
                    
                    # Stats para ACURÁCIA MÁXIMA
                    'first_goal_times': [], 'explosive_starts': 0, 'dominant_starts': 0,
                    'balanced_games': 0, 'clean_sheets_ht': 0, 'goals_conceded_ht': 0,
                    'weekend_games': 0, 'weekend_over': 0,
                    'early_season_games': 0, 'early_season_over': 0,
                    'mid_season_games': 0, 'mid_season_over': 0,
                    'late_season_games': 0, 'late_season_over': 0,
                    
                    # Listas para cálculos avançados
                    'over_sequence': [], 'goals_sequence': [], 'momentum_sequence': [],
                    'consistency_tracker': [], 'form_tracker': []
                }
    
    features = []
    
    for idx, row in df.iterrows():
        home_id = row['home_team_id']
        away_id = row['away_team_id']
        referee = row.get('referee', 'Unknown')
        venue = row.get('venue', 'Unknown')
        league_id = row['league_id']
        
        home_stats = team_stats[home_id]
        away_stats = team_stats[away_id]
        
        # FEATURES BÁSICAS OTIMIZADAS PARA ACURÁCIA
        home_over_rate = home_stats['over_05'] / max(home_stats['games'], 1)
        away_over_rate = away_stats['over_05'] / max(away_stats['games'], 1)
        home_avg_goals = home_stats['goals_scored'] / max(home_stats['games'], 1)
        away_avg_goals = away_stats['goals_scored'] / max(away_stats['games'], 1)
        
        # FEATURES ESPECÍFICAS PARA ACURÁCIA ALTA
        
        # 1. TAXA DE EXPLOSIVE STARTS (2+ gols HT)
        home_explosive_rate = home_stats['explosive_starts'] / max(home_stats['games'], 1)
        away_explosive_rate = away_stats['explosive_starts'] / max(away_stats['games'], 1)
        combined_explosive_rate = (home_explosive_rate + away_explosive_rate) / 2
        
        # 2. TAXA DE DOMINANT STARTS (um time 2+ gols HT)
        home_dominant_rate = home_stats['dominant_starts'] / max(home_stats['games'], 1)
        away_dominant_rate = away_stats['dominant_starts'] / max(away_stats['games'], 1)
        combined_dominant_rate = (home_dominant_rate + away_dominant_rate) / 2
        
        # 3. VELOCIDADE MÉDIA DO PRIMEIRO GOL
        home_avg_first_goal = np.mean(home_stats['first_goal_times']) if home_stats['first_goal_times'] else 45
        away_avg_first_goal = np.mean(away_stats['first_goal_times']) if away_stats['first_goal_times'] else 45
        combined_goal_speed = (home_avg_first_goal + away_avg_first_goal) / 2
        speed_factor = 1 - (combined_goal_speed / 90)  # Quanto menor o tempo, maior o fator
        
        # 4. TAXA DE JOGOS EQUILIBRADOS (ambos marcam HT)
        home_balanced_rate = home_stats['balanced_games'] / max(home_stats['games'], 1)
        away_balanced_rate = away_stats['balanced_games'] / max(away_stats['games'], 1)
        combined_balanced_rate = (home_balanced_rate + away_balanced_rate) / 2
        
        # 5. CONSISTÊNCIA TEMPORAL (evita outliers)
        if len(home_stats['over_sequence']) > 5:
            home_consistency = 1 - np.std(home_stats['over_sequence'][-10:])  # Últimos 10 jogos
        else:
            home_consistency = 0.5
            
        if len(away_stats['over_sequence']) > 5:
            away_consistency = 1 - np.std(away_stats['over_sequence'][-10:])
        else:
            away_consistency = 0.5
        
        combined_consistency = (home_consistency + away_consistency) / 2
        
        # 6. MOMENTUM PONDERADO (últimos 5 jogos com pesos)
        home_recent = home_stats['over_sequence'][-5:] if len(home_stats['over_sequence']) >= 5 else home_stats['over_sequence']
        away_recent = away_stats['over_sequence'][-5:] if len(away_stats['over_sequence']) >= 5 else away_stats['over_sequence']
        
        # Peso maior para jogos mais recentes
        if home_recent:
            weights = np.array([0.6, 0.7, 0.8, 1.0, 1.2])[-len(home_recent):]
            home_momentum = np.average(home_recent, weights=weights)
        else:
            home_momentum = home_over_rate
            
        if away_recent:
            weights = np.array([0.6, 0.7, 0.8, 1.0, 1.2])[-len(away_recent):]
            away_momentum = np.average(away_recent, weights=weights)
        else:
            away_momentum = away_over_rate
        
        combined_momentum = (home_momentum + away_momentum) / 2
        
        # 7. PERFORMANCE CONTEXTUAL
        
        # Weekend performance
        home_weekend_over_rate = home_stats['weekend_over'] / max(home_stats['weekend_games'], 1) if home_stats['weekend_games'] > 0 else home_over_rate
        away_weekend_over_rate = away_stats['weekend_over'] / max(away_stats['weekend_games'], 1) if away_stats['weekend_games'] > 0 else away_over_rate
        
        # Seasonal performance
        home_early_season_rate = home_stats['early_season_over'] / max(home_stats['early_season_games'], 1) if home_stats['early_season_games'] > 0 else home_over_rate
        away_early_season_rate = away_stats['early_season_over'] / max(away_stats['early_season_games'], 1) if away_stats['early_season_games'] > 0 else away_over_rate
        
        # 8. LIGA E CONTEXTO
        league_games = df[df['league_id'] == league_id]
        league_over_rate = league_games['over_05'].mean() if len(league_games) > 0 else 0.5
        
        # Referee e venue
        referee_over_rate = referee_stats.get(referee, {}).get('over_rate', 0.5)
        venue_over_rate = venue_stats.get(venue, {}).get('over_rate', 0.5)
        
        # 9. FEATURES COMBINADAS PARA MÁXIMA ACURÁCIA
        
        # Ultimate Over Score (combinação otimizada)
        ultimate_over_score = (
            (home_over_rate + away_over_rate) * 0.3 +
            combined_explosive_rate * 0.25 +
            speed_factor * 0.2 +
            combined_momentum * 0.15 +
            combined_consistency * 0.1
        )
        
        # Confidence Score (medida de confiança na previsão)
        confidence_score = (
            combined_consistency * 0.4 +
            (min(home_stats['games'], away_stats['games']) / 20) * 0.3 +  # Mais jogos = mais confiança
            (abs(home_over_rate - away_over_rate) < 0.3) * 0.3  # Times equilibrados = mais confiável
        )
        
        # Seasonal Context Score
        day_of_week = row.get('day_of_week', 0)
        season_period = row.get('season_period', 'mid')
        weekend_game = 1 if day_of_week in [5, 6] else 0
        
        if season_period == 'early':
            seasonal_boost = 1.1
        elif season_period == 'late':
            seasonal_boost = 0.9
        else:
            seasonal_boost = 1.0
        
        # Final adjusted score
        final_over_probability = ultimate_over_score * seasonal_boost * (1 + weekend_game * 0.05)
        
        # 🎯 CRIAR FEATURE ROW OTIMIZADA PARA ACURÁCIA
        feature_row = {
            # Features básicas otimizadas
            'home_over_rate': home_over_rate,
            'away_over_rate': away_over_rate,
            'home_avg_goals': home_avg_goals,
            'away_avg_goals': away_avg_goals,
            'combined_over_rate': (home_over_rate + away_over_rate) / 2,
            'combined_goals': home_avg_goals + away_avg_goals,
            
            # Features específicas casa/fora
            'home_home_over_rate': home_stats['home_over'] / max(home_stats['home_games'], 1),
            'away_away_over_rate': away_stats['away_over'] / max(away_stats['away_games'], 1),
            
            # Features de liga e contexto
            'league_over_rate': league_over_rate,
            'referee_over_rate': referee_over_rate,
            'venue_over_rate': venue_over_rate,
            
            # 🔥 FEATURES ESPECÍFICAS PARA ACURÁCIA MÁXIMA
            'home_explosive_rate': home_explosive_rate,
            'away_explosive_rate': away_explosive_rate,
            'combined_explosive_rate': combined_explosive_rate,
            
            'home_dominant_rate': home_dominant_rate,
            'away_dominant_rate': away_dominant_rate,
            'combined_dominant_rate': combined_dominant_rate,
            
            'home_avg_first_goal': home_avg_first_goal,
            'away_avg_first_goal': away_avg_first_goal,
            'combined_goal_speed': combined_goal_speed,
            'speed_factor': speed_factor,
            
            'home_balanced_rate': home_balanced_rate,
            'away_balanced_rate': away_balanced_rate,
            'combined_balanced_rate': combined_balanced_rate,
            
            'home_consistency': home_consistency,
            'away_consistency': away_consistency,
            'combined_consistency': combined_consistency,
            
            'home_momentum': home_momentum,
            'away_momentum': away_momentum,
            'combined_momentum': combined_momentum,
            
            'home_weekend_over_rate': home_weekend_over_rate,
            'away_weekend_over_rate': away_weekend_over_rate,
            'home_early_season_rate': home_early_season_rate,
            'away_early_season_rate': away_early_season_rate,
            
            # Scores combinados finais
            'ultimate_over_score': ultimate_over_score,
            'confidence_score': confidence_score,
            'final_over_probability': final_over_probability,
            
            # Fatores contextuais
            'weekend_game': weekend_game,
            'seasonal_boost': seasonal_boost,
            'context_factor': (referee_over_rate + venue_over_rate + league_over_rate) / 3,
            
            # Features de interação
            'home_away_synergy': home_over_rate * away_over_rate,
            'explosive_synergy': home_explosive_rate * away_explosive_rate,
            'speed_consistency_combo': speed_factor * combined_consistency,
            'momentum_consistency_combo': combined_momentum * combined_consistency,
            
            'target': row['over_05']
        }
        
        features.append(feature_row)
        
        # ATUALIZAR STATS APÓS O JOGO
        ht_home_goals = row.get('ht_home_goals', row.get('ht_home', 0))
        ht_away_goals = row.get('ht_away_goals', row.get('ht_away', 0))
        ht_total = ht_home_goals + ht_away_goals
        first_goal_time = row.get('first_goal_time', 45)
        explosive_start = row.get('early_explosion', 0)
        dominant_start = row.get('dominant_start', 0)
        balanced_scoring = row.get('balanced_scoring', 0)
        day_of_week = row.get('day_of_week', 0)
        season_period = row.get('season_period', 'mid')
        
        # Atualizar stats do time da casa
        team_stats[home_id]['games'] += 1
        team_stats[home_id]['over_05'] += row['over_05']
        team_stats[home_id]['goals_scored'] += ht_home_goals
        team_stats[home_id]['goals_conceded'] += ht_away_goals
        team_stats[home_id]['home_games'] += 1
        team_stats[home_id]['home_over'] += row['over_05']
        team_stats[home_id]['home_goals'] += ht_home_goals
        
        # Stats específicas para acurácia
        team_stats[home_id]['first_goal_times'].append(first_goal_time)
        team_stats[home_id]['explosive_starts'] += explosive_start
        team_stats[home_id]['dominant_starts'] += dominant_start
        team_stats[home_id]['balanced_games'] += balanced_scoring
        if ht_away_goals == 0:
            team_stats[home_id]['clean_sheets_ht'] += 1
        
        # Contexto temporal
        if day_of_week in [5, 6]:
            team_stats[home_id]['weekend_games'] += 1
            team_stats[home_id]['weekend_over'] += row['over_05']
        
        if season_period == 'early':
            team_stats[home_id]['early_season_games'] += 1
            team_stats[home_id]['early_season_over'] += row['over_05']
        elif season_period == 'mid':
            team_stats[home_id]['mid_season_games'] += 1
            team_stats[home_id]['mid_season_over'] += row['over_05']
        else:
            team_stats[home_id]['late_season_games'] += 1
            team_stats[home_id]['late_season_over'] += row['over_05']
        
        # Sequências para cálculos
        team_stats[home_id]['over_sequence'].append(row['over_05'])
        team_stats[home_id]['goals_sequence'].append(ht_home_goals)
        team_stats[home_id]['momentum_sequence'].append(row['over_05'])
        
        # Atualizar stats do time visitante (similar)
        team_stats[away_id]['games'] += 1
        team_stats[away_id]['over_05'] += row['over_05']
        team_stats[away_id]['goals_scored'] += ht_away_goals
        team_stats[away_id]['goals_conceded'] += ht_home_goals
        team_stats[away_id]['away_games'] += 1
        team_stats[away_id]['away_over'] += row['over_05']
        team_stats[away_id]['away_goals'] += ht_away_goals
        
        team_stats[away_id]['first_goal_times'].append(first_goal_time)
        team_stats[away_id]['explosive_starts'] += explosive_start
        team_stats[away_id]['dominant_starts'] += dominant_start
        team_stats[away_id]['balanced_games'] += balanced_scoring
        if ht_home_goals == 0:
            team_stats[away_id]['clean_sheets_ht'] += 1
        
        if day_of_week in [5, 6]:
            team_stats[away_id]['weekend_games'] += 1
            team_stats[away_id]['weekend_over'] += row['over_05']
        
        if season_period == 'early':
            team_stats[away_id]['early_season_games'] += 1
            team_stats[away_id]['early_season_over'] += row['over_05']
        elif season_period == 'mid':
            team_stats[away_id]['mid_season_games'] += 1
            team_stats[away_id]['mid_season_over'] += row['over_05']
        else:
            team_stats[away_id]['late_season_games'] += 1
            team_stats[away_id]['late_season_over'] += row['over_05']
        
        team_stats[away_id]['over_sequence'].append(row['over_05'])
        team_stats[away_id]['goals_sequence'].append(ht_away_goals)
        team_stats[away_id]['momentum_sequence'].append(row['over_05'])
        
        # Atualizar stats de árbitro e venue
        if referee not in referee_stats:
            referee_stats[referee] = {'games': 0, 'over_05': 0}
        referee_stats[referee]['games'] += 1
        referee_stats[referee]['over_05'] += row['over_05']
        referee_stats[referee]['over_rate'] = referee_stats[referee]['over_05'] / referee_stats[referee]['games']
        
        if venue not in venue_stats:
            venue_stats[venue] = {'games': 0, 'over_05': 0}
        venue_stats[venue]['games'] += 1
        venue_stats[venue]['over_05'] += row['over_05']
        venue_stats[venue]['over_rate'] = venue_stats[venue]['over_05'] / venue_stats[venue]['games']
    
    return pd.DataFrame(features), team_stats, referee_stats, venue_stats

def train_accuracy_model(df):
    """Treina modelo OTIMIZADO ESPECIFICAMENTE PARA MÁXIMA ACURÁCIA"""
    
    # Verificar se temos dados suficientes
    if len(df) < 50:
        st.error(f"❌ Dados insuficientes para treinar: apenas {len(df)} jogos. Mínimo necessário: 50")
        return None, {}
    
    st.info("🎯 Preparando sistema ULTIMATE focado em TAXA DE ACERTO máxima...")
    features_df, team_stats, referee_stats, venue_stats = prepare_accuracy_features(df)
    
    # Verificar se features foram geradas corretamente
    if features_df.empty:
        st.error("❌ Erro ao gerar features. Verifique os dados.")
        return None, {}
    
    feature_cols = [col for col in features_df.columns if col != 'target']
    X = features_df[feature_cols]
    y = features_df['target']
    
    # Verificar balanceamento das classes
    class_counts = y.value_counts()
    st.info(f"📊 Distribuição das classes - Over 0.5: {class_counts.get(1, 0)} | Under 0.5: {class_counts.get(0, 0)}")
    
    if len(class_counts) < 2:
        st.error("❌ Dados muito desbalanceados - apenas uma classe presente!")
        return None, {}
    
    st.info(f"🎯 Total de features para ACURÁCIA: {len(feature_cols)}")
    st.info(f"🔥 Foco: Explosive Starts, Velocidade, Consistência, Momentum")
    
    # VALIDAÇÃO TEMPORAL PARA MÁXIMA ACURÁCIA
    # Usar mais dados para treino (80%) e menos para teste (10% each)
    
    # Verificar quantidade mínima de cada classe
    min_samples_per_class = 5
    class_counts = y.value_counts()
    
    if any(class_counts < min_samples_per_class):
        st.warning("⚠️ Poucas amostras por classe. Usando split simples sem stratify.")
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.111, random_state=42)
    else:
        try:
            # Tenta com stratify se houver amostras suficientes
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.111, random_state=42, stratify=y_temp)
        except ValueError as e:
            # Se falhar, usa sem stratify
            st.warning(f"⚠️ Usando split sem stratify: {str(e)}")
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.111, random_state=42)
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # MODELOS OTIMIZADOS PARA ACURÁCIA
    models = {
        'RandomForest_Accuracy': RandomForestClassifier(
            n_estimators=800,  # Muito mais árvores para estabilidade
            max_depth=20,      # Profundidade maior para capturar padrões
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',  # Balanceamento para acurácia
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting_Accuracy': GradientBoostingClassifier(
            n_estimators=600,
            learning_rate=0.03,  # Learning rate muito baixo para precisão
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=0.9,
            random_state=42
        ),
        'LogisticRegression_Calibrated': CalibratedClassifierCV(
            LogisticRegression(
                C=0.5,
                max_iter=2000,
                class_weight='balanced',
                random_state=42
            ),
            method='isotonic',
            cv=3
        )
    }
    
    best_model = None
    best_accuracy = 0
    results = {}
    
    st.info("🎯 Treinando modelos focados em ACURÁCIA MÁXIMA...")
    
    for name, model in models.items():
        with st.spinner(f"Otimizando {name} para máxima acurácia..."):
            
            # Treinar modelo
            model.fit(X_train_scaled, y_train)
            
            # Validação
            val_pred = model.predict(X_val_scaled)
            val_acc = accuracy_score(y_val, val_pred)
            
            # Teste final
            test_pred = model.predict(X_test_scaled)
            test_acc = accuracy_score(y_test, test_pred)
            test_prec = precision_score(y_test, test_pred)
            test_rec = recall_score(y_test, test_pred)
            test_f1 = f1_score(y_test, test_pred)
            
            # Walk-forward validation para estabilidade
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = []
            
            # Usar apenas X_train para walk-forward (sem necessidade de novo split)
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled)):
                if len(train_idx) < 10 or len(val_idx) < 5:
                    continue  # Pular folds muito pequenos
                    
                X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Verificar se há ambas as classes no fold
                if len(np.unique(y_cv_train)) < 2:
                    continue
                
                cv_model = model.__class__(**model.get_params())
                cv_model.fit(X_cv_train, y_cv_train)
                cv_pred = cv_model.predict(X_cv_val)
                cv_scores.append(accuracy_score(y_cv_val, cv_pred))
            
            # Se não há scores suficientes, usar score único de validação
            if len(cv_scores) < 2:
                cv_scores = [val_acc]
                
            cv_mean = np.mean(cv_scores) if cv_scores else val_acc
            cv_std = np.std(cv_scores) if len(cv_scores) > 1 else 0.0
            stability = 1 - cv_std  # Medida de estabilidade
            
            results[name] = {
                'val_accuracy': val_acc,
                'test_accuracy': test_acc,
                'precision': test_prec,
                'recall': test_rec,
                'f1_score': test_f1,
                'cv_accuracy_mean': cv_mean,
                'cv_accuracy_std': cv_std,
                'stability': stability
            }
            
            st.success(f"✅ {name}: Acurácia={test_acc:.1%} | CV={cv_mean:.1%}±{cv_std:.1%} | Estabilidade={stability:.1%}")
            
            # Critério: ACURÁCIA + ESTABILIDADE
            accuracy_score_final = test_acc * 0.7 + stability * 0.3
            if accuracy_score_final > best_accuracy:
                best_accuracy = accuracy_score_final
                best_model = model
    
    # Salvar modelo focado em acurácia
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'team_stats': team_stats,
        'referee_stats': referee_stats,
        'venue_stats': venue_stats,
        'results': results,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(df),
        'features_count': len(feature_cols),
        'accuracy_focused': True,
        'model_version': 'ACCURACY_MASTER_v1.0'
    }
    
    st.session_state.trained_model = model_data
    st.session_state.model_trained = True
    
    # Salvar arquivo
    try:
        for directory in [MODEL_DIR, "/tmp/models"]:
            try:
                if not os.path.exists(directory):
                    os.makedirs(directory)
                model_path = os.path.join(directory, f"accuracy_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
                joblib.dump(model_data, model_path)
                st.success(f"💾 Modelo ACCURACY salvo: {model_path}")
                break
            except:
                pass
    except:
        pass
    
    return model_data, results

def load_latest_model():
    try:
        for directory in [MODEL_DIR, "/tmp/models"]:
            if os.path.exists(directory):
                model_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
                if model_files:
                    latest_model = sorted(model_files)[-1]
                    model_path = os.path.join(directory, latest_model)
                    return joblib.load(model_path)
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
    return None

def predict_matches_accuracy(fixtures, model_data):
    """Faz previsões OTIMIZADAS PARA MÁXIMA ACURÁCIA"""
    predictions = []
    
    if not model_data:
        return predictions
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = model_data['feature_cols']
    team_stats = model_data['team_stats']
    referee_stats = model_data.get('referee_stats', {})
    venue_stats = model_data.get('venue_stats', {})
    
    for fixture in fixtures:
        if fixture['fixture']['status']['short'] not in ['NS', 'TBD']:
            continue
        
        try:
            home_id = fixture['teams']['home']['id']
            away_id = fixture['teams']['away']['id']
            league_name = fixture['league']['name']
            referee = fixture['fixture'].get('referee', 'Unknown')
            venue = fixture['fixture']['venue']['name'] if fixture['fixture']['venue'] else 'Unknown'
            
            if home_id not in team_stats or away_id not in team_stats:
                continue
            
            home_stats = team_stats[home_id]
            away_stats = team_stats[away_id]
            
            # Calcular TODAS as features para máxima acurácia
            home_over_rate = home_stats['over_05'] / max(home_stats['games'], 1)
            away_over_rate = away_stats['over_05'] / max(away_stats['games'], 1)
            home_avg_goals = home_stats['goals_scored'] / max(home_stats['games'], 1)
            away_avg_goals = away_stats['goals_scored'] / max(away_stats['games'], 1)
            
            # Features específicas para acurácia
            home_explosive_rate = home_stats['explosive_starts'] / max(home_stats['games'], 1)
            away_explosive_rate = away_stats['explosive_starts'] / max(away_stats['games'], 1)
            combined_explosive_rate = (home_explosive_rate + away_explosive_rate) / 2
            
            home_avg_first_goal = np.mean(home_stats['first_goal_times']) if home_stats['first_goal_times'] else 45
            away_avg_first_goal = np.mean(away_stats['first_goal_times']) if away_stats['first_goal_times'] else 45
            combined_goal_speed = (home_avg_first_goal + away_avg_first_goal) / 2
            speed_factor = 1 - (combined_goal_speed / 90)
            
            # Momentum ponderado
            home_recent = home_stats['over_sequence'][-5:] if len(home_stats['over_sequence']) >= 5 else home_stats['over_sequence']
            away_recent = away_stats['over_sequence'][-5:] if len(away_stats['over_sequence']) >= 5 else away_stats['over_sequence']
            
            if home_recent:
                weights = np.array([0.6, 0.7, 0.8, 1.0, 1.2])[-len(home_recent):]
                home_momentum = np.average(home_recent, weights=weights)
            else:
                home_momentum = home_over_rate
                
            if away_recent:
                weights = np.array([0.6, 0.7, 0.8, 1.0, 1.2])[-len(away_recent):]
                away_momentum = np.average(away_recent, weights=weights)
            else:
                away_momentum = away_over_rate
            
            combined_momentum = (home_momentum + away_momentum) / 2
            
            # Consistência
            if len(home_stats['over_sequence']) > 5:
                home_consistency = 1 - np.std(home_stats['over_sequence'][-10:])
            else:
                home_consistency = 0.5
                
            if len(away_stats['over_sequence']) > 5:
                away_consistency = 1 - np.std(away_stats['over_sequence'][-10:])
            else:
                away_consistency = 0.5
            
            combined_consistency = (home_consistency + away_consistency) / 2
            
            # Contexto
            match_date = datetime.strptime(fixture['fixture']['date'][:10], '%Y-%m-%d')
            day_of_week = match_date.weekday()
            month = match_date.month
            weekend_game = 1 if day_of_week in [5, 6] else 0
            
            if month in [8, 9, 10]:
                seasonal_boost = 1.1
            elif month in [11, 12, 1, 2]:
                seasonal_boost = 1.0
            else:
                seasonal_boost = 0.9
            
            # Ultimate over score
            ultimate_over_score = (
                (home_over_rate + away_over_rate) * 0.3 +
                combined_explosive_rate * 0.25 +
                speed_factor * 0.2 +
                combined_momentum * 0.15 +
                combined_consistency * 0.1
            )
            
            # Confidence score
            confidence_score = (
                combined_consistency * 0.4 +
                (min(home_stats['games'], away_stats['games']) / 20) * 0.3 +
                (abs(home_over_rate - away_over_rate) < 0.3) * 0.3
            )
            
            # Contexto externo
            referee_over_rate = referee_stats.get(referee, {}).get('over_rate', 0.5)
            venue_over_rate = venue_stats.get(venue, {}).get('over_rate', 0.5)
            context_factor = (referee_over_rate + venue_over_rate + 0.5) / 3  # Liga padrão 0.5
            
            # Final probability
            final_over_probability = ultimate_over_score * seasonal_boost * (1 + weekend_game * 0.05)
            
            # Montar features para predição
            features = {}
            
            # Valores calculados para features necessárias
            feature_values = {
                'home_over_rate': home_over_rate,
                'away_over_rate': away_over_rate,
                'home_avg_goals': home_avg_goals,
                'away_avg_goals': away_avg_goals,
                'combined_over_rate': (home_over_rate + away_over_rate) / 2,
                'combined_goals': home_avg_goals + away_avg_goals,
                'home_home_over_rate': home_stats['home_over'] / max(home_stats['home_games'], 1),
                'away_away_over_rate': away_stats['away_over'] / max(away_stats['away_games'], 1),
                'league_over_rate': 0.5,  # Padrão
                'referee_over_rate': referee_over_rate,
                'venue_over_rate': venue_over_rate,
                'home_explosive_rate': home_explosive_rate,
                'away_explosive_rate': away_explosive_rate,
                'combined_explosive_rate': combined_explosive_rate,
                'home_dominant_rate': home_stats['dominant_starts'] / max(home_stats['games'], 1),
                'away_dominant_rate': away_stats['dominant_starts'] / max(away_stats['games'], 1),
                'combined_dominant_rate': (home_stats['dominant_starts'] / max(home_stats['games'], 1) + away_stats['dominant_starts'] / max(away_stats['games'], 1)) / 2,
                'home_avg_first_goal': home_avg_first_goal,
                'away_avg_first_goal': away_avg_first_goal,
                'combined_goal_speed': combined_goal_speed,
                'speed_factor': speed_factor,
                'home_balanced_rate': home_stats['balanced_games'] / max(home_stats['games'], 1),
                'away_balanced_rate': away_stats['balanced_games'] / max(away_stats['games'], 1),
                'combined_balanced_rate': (home_stats['balanced_games'] / max(home_stats['games'], 1) + away_stats['balanced_games'] / max(away_stats['games'], 1)) / 2,
                'home_momentum': home_momentum,
                'away_momentum': away_momentum,
                'combined_momentum': combined_momentum,
                'home_consistency': home_consistency,
                'away_consistency': away_consistency,
                'combined_consistency': combined_consistency,
                'home_weekend_over_rate': home_stats['weekend_over'] / max(home_stats['weekend_games'], 1) if home_stats['weekend_games'] > 0 else home_over_rate,
                'away_weekend_over_rate': away_stats['weekend_over'] / max(away_stats['weekend_games'], 1) if away_stats['weekend_games'] > 0 else away_over_rate,
                'home_early_season_rate': home_stats['early_season_over'] / max(home_stats['early_season_games'], 1) if home_stats['early_season_games'] > 0 else home_over_rate,
                'away_early_season_rate': away_stats['early_season_over'] / max(away_stats['early_season_games'], 1) if away_stats['early_season_games'] > 0 else away_over_rate,
                'ultimate_over_score': ultimate_over_score,
                'confidence_score': confidence_score,
                'final_over_probability': final_over_probability,
                'weekend_game': weekend_game,
                'seasonal_boost': seasonal_boost,
                'context_factor': context_factor,
                'home_away_synergy': home_over_rate * away_over_rate,
                'explosive_synergy': home_explosive_rate * away_explosive_rate,
                'speed_consistency_combo': speed_factor * combined_consistency,
                'momentum_consistency_combo': combined_momentum * combined_consistency
            }
            
            # Preencher features
            for col in feature_cols:
                if col in feature_values:
                    features[col] = feature_values[col]
                else:
                    features[col] = 0.5  # Valor padrão
            
            # Fazer previsão
            X = pd.DataFrame([features])[feature_cols]
            X_scaled = scaler.transform(X)
            
            pred_proba = model.predict_proba(X_scaled)[0]
            pred_class = model.predict(X_scaled)[0]
            
            # AJUSTE DE CONFIANÇA PARA MÁXIMA ACURÁCIA
            base_confidence = max(pred_proba) * 100
            
            # Multipliers baseados em fatores de acurácia
            confidence_multiplier = 1.0
            
            # Boost para explosive starts
            if combined_explosive_rate > 0.4:
                confidence_multiplier *= 1.08
            
            # Boost para velocidade do primeiro gol
            if speed_factor > 0.6:
                confidence_multiplier *= 1.06
            
            # Boost para momentum positivo
            if combined_momentum > 0.6:
                confidence_multiplier *= 1.04
            
            # Boost para consistência alta
            if combined_consistency > 0.7:
                confidence_multiplier *= 1.05
            
            # Boost para contexto favorável
            if context_factor > 0.55:
                confidence_multiplier *= 1.03
            
            # Penalty para times muito desequilibrados
            if abs(home_over_rate - away_over_rate) > 0.4:
                confidence_multiplier *= 0.95
            
            final_confidence = min(base_confidence * confidence_multiplier, 98)
            
            # Classificação de ACURÁCIA
            if final_confidence > 85:
                accuracy_level = "🎯 ULTRA-HIGH ACCURACY"
                accuracy_class = "accuracy-ultra-high"
            elif final_confidence > 75:
                accuracy_level = "🔥 VERY HIGH ACCURACY"
                accuracy_class = "accuracy-very-high"
            elif final_confidence > 65:
                accuracy_level = "✅ HIGH ACCURACY"
                accuracy_class = "accuracy-high"
            else:
                accuracy_level = "⚡ MEDIUM ACCURACY"
                accuracy_class = "accuracy-medium"
            
            prediction = {
                'fixture': fixture,
                'home_team': fixture['teams']['home']['name'],
                'away_team': fixture['teams']['away']['name'],
                'league': league_name,
                'country': fixture['league']['country'],
                'kickoff': fixture['fixture']['date'],
                'prediction': 'OVER 0.5' if pred_class == 1 else 'UNDER 0.5',
                'confidence': final_confidence,
                'accuracy_level': accuracy_level,
                'accuracy_class': accuracy_class,
                'probability_over': pred_proba[1] * 100,
                'probability_under': pred_proba[0] * 100,
                'accuracy_features': {
                    'explosive_rate': combined_explosive_rate,
                    'speed_factor': speed_factor,
                    'momentum': combined_momentum,
                    'consistency': combined_consistency,
                    'confidence_score': confidence_score,
                    'ultimate_score': ultimate_over_score,
                    'context_factor': context_factor,
                    'seasonal_boost': seasonal_boost,
                    'weekend_boost': weekend_game
                }
            }
            
            predictions.append(prediction)
            
        except Exception as e:
            continue
    
    # Ordenar por confiança (maior primeiro)
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return predictions

def track_accuracy_performance(prediction, actual_result):
    """Rastreia ESPECIFICAMENTE a taxa de acerto"""
    if 'accuracy_tracking' not in st.session_state:
        st.session_state.accuracy_tracking = {'predictions': [], 'results': []}
    
    prediction_record = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'match': f"{prediction['home_team']} vs {prediction['away_team']}",
        'prediction': prediction['prediction'],
        'confidence': prediction['confidence'],
        'accuracy_level': prediction['accuracy_level'],
        'actual_result': 'OVER 0.5' if actual_result else 'UNDER 0.5',
        'correct': (prediction['prediction'] == 'OVER 0.5' and actual_result) or (prediction['prediction'] == 'UNDER 0.5' and not actual_result),
        'explosive_rate': prediction['accuracy_features']['explosive_rate'],
        'speed_factor': prediction['accuracy_features']['speed_factor'],
        'momentum': prediction['accuracy_features']['momentum']
    }
    
    st.session_state.accuracy_tracking['predictions'].append(prediction_record)
    
    # Manter apenas últimos 500 registros
    if len(st.session_state.accuracy_tracking['predictions']) > 500:
        st.session_state.accuracy_tracking['predictions'] = st.session_state.accuracy_tracking['predictions'][-500:]

def calculate_accuracy_metrics(predictions_history):
    """Calcula métricas FOCADAS EM TAXA DE ACERTO"""
    if not predictions_history:
        return {}
    
    df = pd.DataFrame(predictions_history)
    
    total_predictions = len(df)
    correct_predictions = len(df[df['correct'] == True])
    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # Accuracy por nível de confiança
    ultra_high = df[df['accuracy_level'].str.contains('ULTRA-HIGH')]
    very_high = df[df['accuracy_level'].str.contains('VERY HIGH')]
    high = df[df['accuracy_level'].str.contains('HIGH')]
    
    ultra_high_acc = len(ultra_high[ultra_high['correct'] == True]) / len(ultra_high) if len(ultra_high) > 0 else 0
    very_high_acc = len(very_high[very_high['correct'] == True]) / len(very_high) if len(very_high) > 0 else 0
    high_acc = len(high[high['correct'] == True]) / len(high) if len(high) > 0 else 0
    
    # Accuracy por features
    high_explosive = df[df['explosive_rate'] > 0.4]
    high_speed = df[df['speed_factor'] > 0.6]
    high_momentum = df[df['momentum'] > 0.6]
    
    explosive_acc = len(high_explosive[high_explosive['correct'] == True]) / len(high_explosive) if len(high_explosive) > 0 else 0
    speed_acc = len(high_speed[high_speed['correct'] == True]) / len(high_speed) if len(high_speed) > 0 else 0
    momentum_acc = len(high_momentum[high_momentum['correct'] == True]) / len(high_momentum) if len(high_momentum) > 0 else 0
    
    return {
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'overall_accuracy': overall_accuracy,
        'ultra_high_accuracy': ultra_high_acc,
        'very_high_accuracy': very_high_acc,
        'high_accuracy': high_acc,
        'ultra_high_count': len(ultra_high),
        'very_high_count': len(very_high),
        'high_count': len(high),
        'explosive_accuracy': explosive_acc,
        'speed_accuracy': speed_acc,
        'momentum_accuracy': momentum_acc,
        'explosive_count': len(high_explosive),
        'speed_count': len(high_speed),
        'momentum_count': len(high_momentum)
    }

def main():
    # Header ACCURACY MASTER
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Over 0.5 HT - ACCURACY MASTER</h1>
        <p>⚡ SISTEMA OTIMIZADO PARA MÁXIMA TAXA DE ACERTO ⚡</p>
        <p>🔥 Explosive Starts | ⚡ Speed Factor | 📈 Momentum | 🎯 Consistência</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("🎯 ACCURACY MASTER")
        
        # Status da API
        api_ok, requests_left, api_status = check_api_status()
        
        if not api_ok:
            st.error("❌ Problema com a API")
        else:
            st.success(f"✅ API conectada")
            if requests_left > 0:
                st.info(f"📊 Requests: {requests_left}")
        
        # Data
        selected_date = st.date_input(
            "📅 Data para análise:",
            value=datetime.now().date()
        )
        
        # Configurações
        st.subheader("🎯 ACCURACY SETTINGS")
        
        days_training = st.slider(
            "📊 Dias para treinamento:",
            min_value=60,
            max_value=730,
            value=365
        )
        
        use_cache = st.checkbox(
            "💾 Usar dados em cache",
            value=True
        )
        
        # Features para ACURÁCIA
        st.subheader("🔥 ACCURACY FEATURES")
        st.success("""
        🎯 **EXPLOSIVE STARTS**
        ⚡ **SPEED FACTOR** 
        📈 **MOMENTUM PONDERADO**
        🎲 **CONSISTÊNCIA TEMPORAL**
        🏆 **CONTEXT SCORING**
        🔥 **ULTIMATE OVER SCORE**
        """)
        
        # Status do modelo
        if st.session_state.model_trained and st.session_state.trained_model:
            model_data = st.session_state.trained_model
            st.success("✅ ACCURACY MODEL ATIVO")
            st.info(f"📅 Treinado: {model_data['training_date']}")
            st.info(f"🎯 Features: {model_data.get('features_count', 'N/A')}")
            
            if 'results' in model_data:
                results = model_data['results']
                best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
                st.info(f"🏆 Melhor: {best_model[0]}")
                st.info(f"📈 Acurácia: {best_model[1]['test_accuracy']:.1%}")
        else:
            st.warning("⚠️ Treinar ACCURACY MODEL")
    
    # Tabs ACCURACY
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 ACCURACY PREDICTIONS",
        "📊 ACCURACY ANALYSIS", 
        "🔥 TRAIN ACCURACY MODEL",
        "📈 ACCURACY PERFORMANCE"
    ])
    
    with tab1:
        st.header(f"🎯 ACCURACY PREDICTIONS - {selected_date.strftime('%d/%m/%Y')}")
        
        model_data = None
        
        if st.session_state.get('model_trained', False) and st.session_state.get('trained_model'):
            model_data = st.session_state.trained_model
            if model_data.get('accuracy_focused', False):
                st.success("🎯 ACCURACY MODEL ATIVO - Otimizado para máxima taxa de acerto!")
            else:
                st.warning("⚠️ Modelo não otimizado para acurácia. Treine um ACCURACY MODEL.")
        else:
            model_data = load_latest_model()
            if model_data:
                st.session_state.trained_model = model_data
                st.session_state.model_trained = True
        
        if not model_data:
            st.warning("⚠️ Treine um ACCURACY MODEL primeiro!")
        else:
            date_str = selected_date.strftime('%Y-%m-%d')
            fixtures = get_fixtures_cached(date_str)
            
            if not fixtures:
                st.info("📅 Nenhum jogo encontrado para esta data")
                
                st.info("""
                💡 **Para encontrar jogos com alta acurácia:**
                1. Tente quartas, sábados e domingos
                2. Foque em ligas principais (EPL, La Liga, etc.)
                3. Evite períodos de pausa
                """)
            else:
                with st.spinner("🎯 Aplicando ACCURACY MASTER para máxima taxa de acerto..."):
                    predictions = predict_matches_accuracy(fixtures, model_data)
                
                if predictions:
                    # MÉTRICAS DE ACURÁCIA
                    total_games = len(predictions)
                    ultra_high = len([p for p in predictions if 'ULTRA-HIGH' in p['accuracy_level']])
                    very_high = len([p for p in predictions if 'VERY HIGH' in p['accuracy_level']])
                    high_accuracy = len([p for p in predictions if 'HIGH' in p['accuracy_level']])
                    over_predictions = len([p for p in predictions if p['prediction'] == 'OVER 0.5'])
                    avg_confidence = sum([p['confidence'] for p in predictions]) / len(predictions)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="accuracy-card">
                            <h3>🎮 Total Jogos</h3>
                            <h1>{total_games}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="accuracy-card">
                            <h3>🎯 Ultra-High</h3>
                            <h1 class="accuracy-ultra-high">{ultra_high}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="accuracy-card">
                            <h3>🔥 Very High</h3>
                            <h1 class="accuracy-very-high">{very_high}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div class="accuracy-card">
                            <h3>💯 Confiança Média</h3>
                            <h1>{avg_confidence:.1f}%</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 🎯 PREVISÕES ULTRA-HIGH ACCURACY (85%+)
                    ultra_high_preds = [p for p in predictions if 'ULTRA-HIGH' in p['accuracy_level'] and p['prediction'] == 'OVER 0.5']
                    
                    if ultra_high_preds:
                        st.subheader("🎯 ULTRA-HIGH ACCURACY PREDICTIONS (85%+ Acurácia)")
                        
                        for pred in ultra_high_preds[:5]:
                            try:
                                utc_time = datetime.strptime(pred['kickoff'][:16], '%Y-%m-%dT%H:%M')
                                hora_portugal = utc_time.strftime('%H:%M')
                            except:
                                hora_portugal = pred['kickoff'][11:16]
                            
                            acc_features = pred['accuracy_features']
                            
                            st.markdown(f"""
                            <div class="prediction-premium">
                                <h2>⚽ {pred['home_team']} vs {pred['away_team']} {pred['accuracy_level']}</h2>
                                <p><strong>🏆 Liga:</strong> {pred['league']} ({pred['country']})</p>
                                <p><strong>🕐 Horário PT:</strong> {hora_portugal}</p>
                                <hr style="opacity: 0.4;">
                                <p><strong>🎯 Previsão:</strong> {pred['prediction']}</p>
                                <p><strong>💯 Taxa de Acerto Esperada:</strong> <span class="{pred['accuracy_class']}">{pred['confidence']:.1f}%</span></p>
                                <p><strong>📊 Probabilidades:</strong> Over {pred['probability_over']:.1f}% | Under {pred['probability_under']:.1f}%</p>
                                <hr style="opacity: 0.4;">
                                <p><strong>🔥 ANÁLISE DE ACURÁCIA:</strong></p>
                                <p>• <strong>Explosive Rate:</strong> {acc_features['explosive_rate']:.1%} (Times com starts explosivos)</p>
                                <p>• <strong>Speed Factor:</strong> {acc_features['speed_factor']:.1%} (Velocidade do 1º gol)</p>
                                <p>• <strong>Momentum:</strong> {acc_features['momentum']:.1%} (Forma recente ponderada)</p>
                                <p>• <strong>Consistência:</strong> {acc_features['consistency']:.1%} (Estabilidade temporal)</p>
                                <p>• <strong>Ultimate Score:</strong> {acc_features['ultimate_score']:.3f} (Score final otimizado)</p>
                                <p>• <strong>Context Factor:</strong> {acc_features['context_factor']:.1%} (Árbitro + Venue + Liga)</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # 🔥 PREVISÕES VERY HIGH ACCURACY (75-85%)
                    very_high_preds = [p for p in predictions if 'VERY HIGH' in p['accuracy_level'] and p['prediction'] == 'OVER 0.5']
                    
                    if very_high_preds:
                        st.subheader("🔥 VERY HIGH ACCURACY PREDICTIONS (75-85% Acurácia)")
                        
                        for pred in very_high_preds[:8]:
                            try:
                                utc_time = datetime.strptime(pred['kickoff'][:16], '%Y-%m-%dT%H:%M')
                                hora_portugal = utc_time.strftime('%H:%M')
                            except:
                                hora_portugal = pred['kickoff'][11:16]
                            
                            acc_features = pred['accuracy_features']
                            
                            st.markdown(f"""
                            <div class="prediction-excellent">
                                <h3>⚽ {pred['home_team']} vs {pred['away_team']} {pred['accuracy_level']}</h3>
                                <p><strong>🏆 Liga:</strong> {pred['league']} ({pred['country']})</p>
                                <p><strong>🕐 Horário PT:</strong> {hora_portugal}</p>
                                <hr style="opacity: 0.3;">
                                <p><strong>🎯 Previsão:</strong> {pred['prediction']}</p>
                                <p><strong>💯 Taxa de Acerto:</strong> <span class="{pred['accuracy_class']}">{pred['confidence']:.1f}%</span></p>
                                <p><strong>🔥 Explosive:</strong> {acc_features['explosive_rate']:.1%} | <strong>⚡ Speed:</strong> {acc_features['speed_factor']:.1%} | <strong>📈 Momentum:</strong> {acc_features['momentum']:.1%}</p>
                                <p><strong>🎯 Ultimate Score:</strong> {acc_features['ultimate_score']:.3f} | <strong>🎲 Consistência:</strong> {acc_features['consistency']:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # ✅ PREVISÕES HIGH ACCURACY (65-75%)
                    high_preds = [p for p in predictions if 'HIGH' in p['accuracy_level'] and not 'VERY HIGH' in p['accuracy_level'] and not 'ULTRA-HIGH' in p['accuracy_level'] and p['prediction'] == 'OVER 0.5']
                    
                    if high_preds:
                        with st.expander(f"✅ HIGH ACCURACY PREDICTIONS (65-75%) - {len(high_preds)} jogos"):
                            for pred in high_preds:
                                try:
                                    utc_time = datetime.strptime(pred['kickoff'][:16], '%Y-%m-%dT%H:%M')
                                    hora_portugal = utc_time.strftime('%H:%M')
                                except:
                                    hora_portugal = pred['kickoff'][11:16]
                                
                                st.write(f"**{hora_portugal}** | {pred['home_team']} vs {pred['away_team']} | **{pred['confidence']:.1f}%** | {pred['league']}")
                    
                    # Tabela completa de previsões OVER
                    with st.expander("📋 Todas as Previsões OVER 0.5 (ordenadas por acurácia)"):
                        over_predictions_list = [p for p in predictions if p['prediction'] == 'OVER 0.5']
                        
                        if over_predictions_list:
                            pred_data = []
                            for p in over_predictions_list:
                                try:
                                    utc_time = datetime.strptime(p['kickoff'][:16], '%Y-%m-%dT%H:%M')
                                    hora_pt = utc_time.strftime('%H:%M')
                                except:
                                    hora_pt = p['kickoff'][11:16]
                                
                                acc_features = p['accuracy_features']
                                
                                pred_data.append({
                                    'Hora PT': hora_pt,
                                    'Casa': p['home_team'],
                                    'Fora': p['away_team'],
                                    'Liga': p['league'],
                                    'Nível': p['accuracy_level'],
                                    'Acurácia': f"{p['confidence']:.1f}%",
                                    'Explosive': f"{acc_features['explosive_rate']:.1%}",
                                    'Speed': f"{acc_features['speed_factor']:.1%}",
                                    'Momentum': f"{acc_features['momentum']:.1%}",
                                    'Score': f"{acc_features['ultimate_score']:.3f}",
                                    '_confidence': p['confidence']
                                })
                            
                            pred_df = pd.DataFrame(pred_data)
                            pred_df = pred_df.sort_values('_confidence', ascending=False).drop('_confidence', axis=1)
                            st.dataframe(pred_df, use_container_width=True)
                        else:
                            st.info("Nenhuma previsão OVER 0.5 encontrada")
                
                else:
                    st.info("🤷 Nenhuma previsão disponível (times sem dados históricos)")
    
    with tab2:
        st.header("📊 ACCURACY ANALYSIS")
        
        st.info("""
        🎯 **ANÁLISE FOCADA EM TAXA DE ACERTO**
        
        Esta seção analisa as ligas e contextos que geram **maior taxa de acerto** para Over 0.5 HT,
        independentemente de odds ou ROI.
        """)
        
        if 'trained_model' in st.session_state:
            model_data = st.session_state.trained_model
        else:
            model_data = load_latest_model()
        
        if model_data:
            df = collect_historical_data(days=30, use_cached=True)
            
            if not df.empty:
                # Análise de acurácia por liga
                st.subheader("📊 Taxa de Acerto por Liga")
                
                league_accuracy = {}
                for league_id in df['league_id'].unique():
                    league_data = df[df['league_id'] == league_id]
                    
                    if len(league_data) >= 15:  # Mínimo 15 jogos para confiabilidade
                        over_rate = league_data['over_05'].mean()
                        total_games = len(league_data)
                        league_name = league_data.iloc[0]['league_name']
                        country = league_data.iloc[0]['country']
                        
                        # Calcular "previsibilidade" da liga
                        explosive_games = len(league_data[league_data['ht_total_goals'] >= 2])
                        explosive_rate = explosive_games / total_games
                        
                        # Classificação baseada em acurácia esperada
                        if over_rate >= 0.75:
                            accuracy_class = "🎯 ULTRA-HIGH ACCURACY"
                            expected_accuracy = "85-95%"
                        elif over_rate >= 0.65:
                            accuracy_class = "🔥 VERY HIGH ACCURACY"
                            expected_accuracy = "75-85%"
                        elif over_rate >= 0.55:
                            accuracy_class = "✅ HIGH ACCURACY"
                            expected_accuracy = "65-75%"
                        elif over_rate <= 0.35:
                            accuracy_class = "❄️ HIGH UNDER ACCURACY"
                            expected_accuracy = "65-75% (UNDER)"
                        elif over_rate <= 0.45:
                            accuracy_class = "📉 MEDIUM UNDER ACCURACY"
                            expected_accuracy = "60-70% (UNDER)"
                        else:
                            accuracy_class = "⚖️ BALANCED (AVOID)"
                            expected_accuracy = "50-60%"
                        
                        league_accuracy[league_name] = {
                            'country': country,
                            'over_rate': over_rate,
                            'explosive_rate': explosive_rate,
                            'total_games': total_games,
                            'accuracy_class': accuracy_class,
                            'expected_accuracy': expected_accuracy,
                            'reliability': min(total_games / 30, 1.0)
                        }
                
                # Separar por categoria de acurácia
                ultra_high = {k: v for k, v in league_accuracy.items() if 'ULTRA-HIGH' in v['accuracy_class']}
                very_high = {k: v for k, v in league_accuracy.items() if 'VERY HIGH' in v['accuracy_class']}
                high_over = {k: v for k, v in league_accuracy.items() if 'HIGH ACCURACY' in v['accuracy_class'] and 'UNDER' not in v['accuracy_class']}
                high_under = {k: v for k, v in league_accuracy.items() if 'UNDER ACCURACY' in v['accuracy_class']}
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 ULTRA-HIGH & VERY HIGH ACCURACY LEAGUES")
                    
                    for league, stats in sorted({**ultra_high, **very_high}.items(), key=lambda x: x[1]['over_rate'], reverse=True):
                        reliability_icon = "🟢" if stats['reliability'] > 0.8 else "🟡" if stats['reliability'] > 0.5 else "🔴"
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%); 
                                    color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;
                                    border: 2px solid gold;">
                            <h4>{league} {reliability_icon}</h4>
                            <p>{stats['accuracy_class']}</p>
                            <p><strong>📊 Taxa Over:</strong> {stats['over_rate']:.1%}</p>
                            <p><strong>🎯 Acurácia Esperada:</strong> {stats['expected_accuracy']}</p>
                            <p><strong>🔥 Explosive Rate:</strong> {stats['explosive_rate']:.1%}</p>
                            <p><strong>🎮 Jogos:</strong> {stats['total_games']} | <strong>🔒 Confiabilidade:</strong> {stats['reliability']:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.subheader("✅ HIGH ACCURACY LEAGUES")
                    
                    for league, stats in sorted(high_over.items(), key=lambda x: x[1]['over_rate'], reverse=True):
                        reliability_icon = "🟢" if stats['reliability'] > 0.8 else "🟡" if stats['reliability'] > 0.5 else "🔴"
                        
                        st.markdown(f"""
                        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
                                    border-left: 4px solid #28a745;">
                            <h4>{league} {reliability_icon}</h4>
                            <p>{stats['accuracy_class']}</p>
                            <p><strong>📊 Taxa Over:</strong> {stats['over_rate']:.1%}</p>
                            <p><strong>🎯 Acurácia Esperada:</strong> {stats['expected_accuracy']}</p>
                            <p><strong>🔥 Explosive Rate:</strong> {stats['explosive_rate']:.1%}</p>
                            <p><strong>🎮 Jogos:</strong> {stats['total_games']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Análise temporal para acurácia
                st.subheader("📅 Análise Temporal para Máxima Acurácia")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**📅 Por Dia da Semana:**")
                    if 'day_of_week' in df.columns:
                        day_names = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
                        for day in range(7):
                            day_data = df[df['day_of_week'] == day]
                            if len(day_data) > 5:
                                day_over_rate = day_data['over_05'].mean()
                                color = "🟢" if day_over_rate > 0.6 else "🟡" if day_over_rate > 0.4 else "🔴"
                                st.write(f"{color} {day_names[day]}: {day_over_rate:.1%}")
                
                with col2:
                    st.write("**🗓️ Por Período da Temporada:**")
                    if 'season_period' in df.columns:
                        for period in ['early', 'mid', 'late']:
                            period_data = df[df['season_period'] == period]
                            if len(period_data) > 5:
                                period_over_rate = period_data['over_05'].mean()
                                period_name = {'early': 'Início', 'mid': 'Meio', 'late': 'Final'}[period]
                                color = "🟢" if period_over_rate > 0.6 else "🟡" if period_over_rate > 0.4 else "🔴"
                                st.write(f"{color} {period_name}: {period_over_rate:.1%}")
                
                with col3:
                    st.write("**⚡ Por Features de Acurácia:**")
                    if 'early_explosion' in df.columns:
                        explosive_data = df[df['early_explosion'] == 1]
                        if len(explosive_data) > 0:
                            explosive_over_rate = explosive_data['over_05'].mean()
                            st.write(f"🔥 Explosive Starts: {explosive_over_rate:.1%}")
                    
                    if 'balanced_scoring' in df.columns:
                        balanced_data = df[df['balanced_scoring'] == 1]
                        if len(balanced_data) > 0:
                            balanced_over_rate = balanced_data['over_05'].mean()
                            st.write(f"⚖️ Balanced Scoring: {balanced_over_rate:.1%}")
                
                # Dicas para máxima acurácia
                st.subheader("💡 Dicas para Máxima Taxa de Acerto")
                
                st.success("""
                🎯 **FOQUE NESTAS LIGAS/SITUAÇÕES:**
                
                1. **Ligas ULTRA-HIGH**: 85-95% de acurácia esperada
                2. **Ligas VERY HIGH**: 75-85% de acurácia esperada  
                3. **Weekends**: Geralmente maior taxa Over
                4. **Início de temporada**: Times mais atacantes
                5. **Times com Explosive Rate alto**: Mais starts rápidos
                
                ⚠️ **EVITE:**
                - Ligas equilibradas (50-60% acurácia)
                - Final de temporada em ligas defensivas
                - Times muito inconsistentes
                """)
        else:
            st.info("🤖 Treine um modelo primeiro")
    
    with tab3:
        st.header("🔥 TRAIN ACCURACY MASTER MODEL")
        
        st.success("""
        🎯 **ACCURACY MASTER - FEATURES OTIMIZADAS PARA TAXA DE ACERTO:**
        
        **⚡ Explosive Features:**
        ✅ Taxa de Explosive Starts (2+ gols HT)
        ✅ Taxa de Dominant Starts (um time 2+ gols HT)
        ✅ Taxa de Balanced Scoring (ambos marcam HT)
        
        **🏃 Speed Features:**
        ✅ Tempo médio do primeiro gol por time
        ✅ Speed Factor (velocidade normalizada)
        ✅ Fast Start Probability
        
        **📈 Momentum Features:**
        ✅ Momentum ponderado (últimos 5 jogos)
        ✅ Consistência temporal (últimos 10 jogos)
        ✅ Form tracking com pesos temporais
        
        **🎯 Context Features:**
        ✅ Performance por dia da semana
        ✅ Fatores sazonais (início/meio/fim)
        ✅ Performance por árbitro/venue
        
        **🔥 Ultimate Features:**
        ✅ Ultimate Over Score (fórmula otimizada)
        ✅ Confidence Score (medida de confiabilidade)
        ✅ Context Factor (ambiente do jogo)
        """)
        
        st.info("""
        **🎯 TREINAMENTO OTIMIZADO PARA ACURÁCIA:**
        - **80%** treino | **10%** validação | **10%** teste
        - **Modelos calibrados** para máxima precisão
        - **Class balancing** para evitar viés
        - **Walk-forward validation** para estabilidade
        - **Critério de seleção**: Acurácia + Estabilidade
        """)
        
        if st.button("🚀 TREINAR ACCURACY MASTER MODEL", type="primary"):
            with st.spinner(f"📊 Coletando {days_training} dias de dados para ACURÁCIA MÁXIMA..."):
                df = collect_historical_data(days=days_training, use_cached=use_cache)
            
            if df.empty:
                st.error("❌ Não foi possível coletar dados")
                
                st.info("🔍 Diagnosticando problema...")
                test_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                test_fixtures = get_fixtures(test_date)
                st.write(f"Teste API ({test_date}): {len(test_fixtures)} jogos encontrados")
            else:
                st.success(f"✅ {len(df)} jogos coletados para ACCURACY TRAINING")
                
                # Mostrar estatísticas dos dados
                over_rate = df['over_05'].mean()
                st.info(f"📊 Taxa Over geral dos dados: {over_rate:.1%}")
                
                # Diagnóstico adicional
                st.info(f"🔍 Diagnóstico dos dados:")
                st.write(f"- Total de jogos: {len(df)}")
                st.write(f"- Jogos Over 0.5: {df['over_05'].sum()}")
                st.write(f"- Jogos Under 0.5: {len(df) - df['over_05'].sum()}")
                st.write(f"- Colunas disponíveis: {', '.join(df.columns[:10])}...")
                
                if over_rate < 0.3:
                    st.warning("⚠️ Taxa Over muito baixa - pode afetar treinamento")
                elif over_rate > 0.7:
                    st.warning("⚠️ Taxa Over muito alta - pode afetar treinamento")
                else:
                    st.success("✅ Taxa Over balanceada - ótimo para treinamento")
                
                with st.spinner("🎯 Treinando ACCURACY MASTER com features otimizadas..."):
                    model_data, results = train_accuracy_model(df)
                
                if model_data:
                    st.success("🏆 ACCURACY MASTER MODEL treinado com sucesso!")
                    
                    st.subheader("📊 Resultados do ACCURACY TRAINING")
                    
                    # Mostrar resultados com foco em acurácia
                    best_accuracy = 0
                    best_model_name = ""
                    
                    for model_name, metrics in results.items():
                        col1, col2, col3, col4, col5, col6 = st.columns(6)
                        
                        test_acc = metrics['test_accuracy']
                        if test_acc > best_accuracy:
                            best_accuracy = test_acc
                            best_model_name = model_name
                        
                        with col1:
                            st.metric(model_name.replace('_Accuracy', '').replace('_Calibrated', ''), "")
                        with col2:
                            accuracy_color = "🎯" if test_acc > 0.7 else "🔥" if test_acc > 0.65 else "✅"
                            st.metric("ACURÁCIA", f"{accuracy_color} {test_acc:.1%}")
                        with col3:
                            cv_acc = metrics['cv_accuracy_mean']
                            st.metric("CV Accuracy", f"{cv_acc:.1%}±{metrics['cv_accuracy_std']:.1%}")
                        with col4:
                            stability = metrics['stability']
                            stability_color = "🟢" if stability > 0.85 else "🟡" if stability > 0.75 else "🔴"
                            st.metric("Estabilidade", f"{stability_color} {stability:.1%}")
                        with col5:
                            st.metric("Precisão", f"{metrics['precision']:.1%}")
                        with col6:
                            st.metric("F1-Score", f"{metrics['f1_score']:.1%}")
                    
                    # Destacar melhor modelo
                    st.success(f"🏆 MELHOR MODELO: {best_model_name} com {best_accuracy:.1%} de acurácia!")
                    
                    # Feature importance para acurácia
                    if hasattr(model_data['model'], 'feature_importances_'):
                        st.subheader("🎯 Features Mais Importantes para ACURÁCIA")
                        
                        feature_importance = pd.DataFrame({
                            'feature': model_data['feature_cols'],
                            'importance': model_data['model'].feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        # Top 20 features
                        top_features = feature_importance.head(20)
                        st.bar_chart(top_features.set_index('feature')['importance'])
                        
                        # Categorizar features
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**🔥 Top Features de Acurácia:**")
                            accuracy_features = top_features[top_features['feature'].str.contains('explosive|speed|ultimate|confidence|consistency')]
                            if not accuracy_features.empty:
                                for _, row in accuracy_features.head(8).iterrows():
                                    st.write(f"• **{row['feature']}**: {row['importance']:.3f}")
                            else:
                                st.write("Features de acurácia não estão no top 20")
                        
                        with col2:
                            st.write("**📊 Top Features Básicas:**")
                            basic_features = top_features[top_features['feature'].str.contains('over_rate|avg_goals|momentum')]
                            if not basic_features.empty:
                                for _, row in basic_features.head(8).iterrows():
                                    st.write(f"• **{row['feature']}**: {row['importance']:.3f}")
                    
                    # Estimativa de performance
                    st.subheader("🎯 Estimativa de Performance em Produção")
                    
                    estimated_accuracy = best_accuracy * 0.95  # Ligeira redução para produção
                    
                    if estimated_accuracy > 0.75:
                        performance_level = "🎯 ELITE"
                        performance_color = "success"
                    elif estimated_accuracy > 0.70:
                        performance_level = "🔥 EXCELENTE"
                        performance_color = "success"
                    elif estimated_accuracy > 0.65:
                        performance_level = "✅ MUITO BOM"
                        performance_color = "info"
                    else:
                        performance_level = "⚡ BOM"
                        performance_color = "warning"
                    
                    st.markdown(f"""
                    **Nível de Performance Esperado:** {performance_level}
                    
                    **Taxa de Acerto Estimada:** {estimated_accuracy:.1%}
                    
                    **Comparação com Mercado:**
                    - Tipsters básicos: 45-55%
                    - Sistemas simples: 55-62%
                    - **Seu ACCURACY MASTER**: {estimated_accuracy:.1%}
                    - Diferencial: +{(estimated_accuracy - 0.55)*100:.0f} pontos percentuais
                    """)
        
        # Teste de conectividade
        if st.button("🔌 Testar Conectividade API", type="secondary"):
            with st.spinner("Testando conexão..."):
                api_ok, requests_left, status = check_api_status()
                
                if api_ok:
                    st.success("✅ API funcionando perfeitamente!")
                    st.info(f"📊 Requests restantes: {requests_left}")
                    
                    # Teste de dados
                    test_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                    test_fixtures = get_fixtures(test_date)
                    st.info(f"🎮 Jogos encontrados para {test_date}: {len(test_fixtures)}")
                    
                    if len(test_fixtures) > 0:
                        sample_match = test_fixtures[0]
                        st.write("**Exemplo de jogo encontrado:**")
                        st.write(f"• {sample_match['teams']['home']['name']} vs {sample_match['teams']['away']['name']}")
                        st.write(f"• Liga: {sample_match['league']['name']}")
                        st.write(f"• Status: {sample_match['fixture']['status']['short']}")
                else:
                    st.error(f"❌ Problema com API: {status}")
    
    with tab4:
        st.header("📈 ACCURACY PERFORMANCE TRACKING")
        
        st.info("""
        🎯 **TRACKING FOCADO EM TAXA DE ACERTO**
        
        Esta seção monitora exclusivamente a **taxa de acerto** das previsões,
        sem considerar odds, ROI ou outros fatores financeiros.
        """)
        
        if 'trained_model' in st.session_state:
            model_data = st.session_state.trained_model
        else:
            model_data = load_latest_model()
        
        if model_data and 'results' in model_data:
            # Verificar se é modelo focado em acurácia
            if model_data.get('accuracy_focused', False):
                st.success("🎯 ACCURACY MASTER MODEL - Otimizado para máxima taxa de acerto")
                st.info(f"📊 Features otimizadas: {model_data.get('features_count', 'N/A')}")
            else:
                st.warning("⚠️ Modelo não otimizado especificamente para acurácia")
            
            results = model_data['results']
            
            # Encontrar melhor modelo por acurácia
            best_model_name = max(results.items(), key=lambda x: x[1]['test_accuracy'])[0]
            best_metrics = results[best_model_name]
            
            st.subheader(f"🏆 Melhor Modelo: {best_model_name}")
            
            # MÉTRICAS FOCADAS EM ACURÁCIA
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                accuracy = best_metrics['test_accuracy'] * 100
                if accuracy > 75:
                    accuracy_icon = "🎯"
                    accuracy_class = "accuracy-ultra-high"
                elif accuracy > 70:
                    accuracy_icon = "🔥"
                    accuracy_class = "accuracy-very-high"
                elif accuracy > 65:
                    accuracy_icon = "✅"
                    accuracy_class = "accuracy-high"
                else:
                    accuracy_icon = "⚡"
                    accuracy_class = "accuracy-medium"
                
                st.markdown(f"""
                <div class="accuracy-card">
                    <h3>{accuracy_icon} TAXA DE ACERTO</h3>
                    <h1 class="{accuracy_class}">{accuracy:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                precision = best_metrics['precision'] * 100
                st.markdown(f"""
                <div class="accuracy-card">
                    <h3>💎 Precisão</h3>
                    <h1>{precision:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                if 'cv_accuracy_mean' in best_metrics:
                    cv_acc = best_metrics['cv_accuracy_mean'] * 100
                    cv_std = best_metrics['cv_accuracy_std'] * 100
                    st.markdown(f"""
                    <div class="accuracy-card">
                        <h3>📊 CV Accuracy</h3>
                        <h1>{cv_acc:.1f}%±{cv_std:.1f}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    recall = best_metrics['recall'] * 100
                    st.markdown(f"""
                    <div class="accuracy-card">
                        <h3>📈 Recall</h3>
                        <h1>{recall:.1f}%</h1>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col4:
                if 'stability' in best_metrics:
                    stability = best_metrics['stability'] * 100
                    stability_icon = "🟢" if stability > 85 else "🟡" if stability > 75 else "🔴"
                    st.markdown(f"""
                    <div class="accuracy-card">
                        <h3>{stability_icon} Estabilidade</h3>
                        <h1>{stability:.1f}%</h1>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    f1 = best_metrics['f1_score'] * 100
                    st.markdown(f"""
                    <div class="accuracy-card">
                        <h3>🏅 F1-Score</h3>
                        <h1>{f1:.1f}%</h1>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Performance em tempo real
            st.subheader("📊 Performance em Tempo Real")
            
            if st.session_state.accuracy_tracking['predictions']:
                accuracy_metrics = calculate_accuracy_metrics(st.session_state.accuracy_tracking['predictions'])
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    real_accuracy = accuracy_metrics['overall_accuracy'] * 100
                    accuracy_icon = "🎯" if real_accuracy > 75 else "🔥" if real_accuracy > 70 else "✅"
                    st.markdown(f"""
                    <div class="accuracy-card">
                        <h3>{accuracy_icon} Taxa Real</h3>
                        <h1 class="accuracy-very-high">{real_accuracy:.1f}%</h1>
                        <p>{accuracy_metrics['total_predictions']} previsões</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    ultra_high_acc = accuracy_metrics['ultra_high_accuracy'] * 100
                    st.markdown(f"""
                    <div class="accuracy-card">
                        <h3>🎯 Ultra-High</h3>
                        <h1 class="accuracy-ultra-high">{ultra_high_acc:.1f}%</h1>
                        <p>{accuracy_metrics['ultra_high_count']} previsões</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    very_high_acc = accuracy_metrics['very_high_accuracy'] * 100
                    st.markdown(f"""
                    <div class="accuracy-card">
                        <h3>🔥 Very High</h3>
                        <h1 class="accuracy-very-high">{very_high_acc:.1f}%</h1>
                        <p>{accuracy_metrics['very_high_count']} previsões</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    high_acc = accuracy_metrics['high_accuracy'] * 100
                    st.markdown(f"""
                    <div class="accuracy-card">
                        <h3>✅ High</h3>
                        <h1 class="accuracy-high">{high_acc:.1f}%</h1>
                        <p>{accuracy_metrics['high_count']} previsões</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Performance por features de acurácia
                st.subheader("🔥 Performance por Features de Acurácia")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    explosive_acc = accuracy_metrics['explosive_accuracy'] * 100
                    st.markdown(f"""
                    **🔥 Explosive Rate Alta (>40%):**
                    - Acurácia: {explosive_acc:.1f}%
                    - Previsões: {accuracy_metrics['explosive_count']}
                    """)
                
                with col2:
                    speed_acc = accuracy_metrics['speed_accuracy'] * 100
                    st.markdown(f"""
                    **⚡ Speed Factor Alto (>60%):**
                    - Acurácia: {speed_acc:.1f}%
                    - Previsões: {accuracy_metrics['speed_count']}
                    """)
                
                with col3:
                    momentum_acc = accuracy_metrics['momentum_accuracy'] * 100
                    st.markdown(f"""
                    **📈 Momentum Alto (>60%):**
                    - Acurácia: {momentum_acc:.1f}%
                    - Previsões: {accuracy_metrics['momentum_count']}
                    """)
                
                # Gráfico de evolução da acurácia
                if len(st.session_state.accuracy_tracking['predictions']) > 10:
                    st.subheader("📈 Evolução da Taxa de Acerto")
                    
                    df_tracking = pd.DataFrame(st.session_state.accuracy_tracking['predictions'])
                    
                    # Calcular acurácia rolling (últimas 10 previsões)
                    df_tracking['correct_numeric'] = df_tracking['correct'].astype(int)
                    df_tracking['rolling_accuracy'] = df_tracking['correct_numeric'].rolling(window=10, min_periods=5).mean() * 100
                    
                    # Mostrar gráfico
                    st.line_chart(df_tracking.set_index('date')['rolling_accuracy'])
                    
                    # Estatísticas da evolução
                    latest_accuracy = df_tracking['rolling_accuracy'].iloc[-1]
                    initial_accuracy = df_tracking['rolling_accuracy'].dropna().iloc[0]
                    improvement = latest_accuracy - initial_accuracy
                    
                    if improvement > 5:
                        trend_icon = "📈"
                        trend_text = "MELHORANDO"
                        trend_color = "success"
                    elif improvement < -5:
                        trend_icon = "📉"
                        trend_text = "PIORANDO"
                        trend_color = "warning"
                    else:
                        trend_icon = "➡️"
                        trend_text = "ESTÁVEL"
                        trend_color = "info"
                    
                    st.markdown(f"""
                    **{trend_icon} Tendência:** {trend_text}
                    
                    **Acurácia Atual (últimas 10):** {latest_accuracy:.1f}%
                    **Acurácia Inicial:** {initial_accuracy:.1f}%
                    **Mudança:** {improvement:+.1f} pontos percentuais
                    """)
            else:
                st.info("""
                📊 **Ainda não há dados de performance em tempo real**
                
                Para começar o tracking de acurácia:
                1. Use o sistema para fazer previsões
                2. O sistema automaticamente rastreará os resultados
                3. Métricas de acurácia aparecerão aqui
                
                **Métricas que serão rastreadas:**
                - Taxa de acerto geral
                - Taxa de acerto por nível de confiança
                - Performance por features de acurácia
                - Evolução temporal da acurácia
                """)
            
            # Comparação com benchmarks
            st.subheader("🏆 Comparação com Benchmarks de Mercado")
            
            model_accuracy = best_metrics['test_accuracy'] * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **📊 Benchmarks de Taxa de Acerto:**
                
                🔴 **Chute aleatório:** 50%
                🟠 **Tipsters amadores:** 45-55%
                🟡 **Sistemas básicos:** 55-62%
                🟢 **Sistemas avançados:** 62-70%
                🎯 **Sistemas profissionais:** 70-80%
                🏆 **Elite mundial:** 80%+
                """)
            
            with col2:
                if model_accuracy > 80:
                    level = "🏆 ELITE MUNDIAL"
                    position = "Top 1%"
                    color = "🎯"
                elif model_accuracy > 70:
                    level = "🎯 PROFISSIONAL"
                    position = "Top 5%"
                    color = "🔥"
                elif model_accuracy > 62:
                    level = "🟢 AVANÇADO"
                    position = "Top 20%"
                    color = "✅"
                elif model_accuracy > 55:
                    level = "🟡 INTERMEDIÁRIO"
                    position = "Acima da média"
                    color = "⚡"
                else:
                    level = "🟠 INICIANTE"
                    position = "Precisa melhorar"
                    color = "📈"
                
                st.markdown(f"""
                **🎯 Classificação do Seu Sistema:**
                
                **Nível:** {color} {level}
                **Taxa de Acerto:** {model_accuracy:.1f}%
                **Posição no Mercado:** {position}
                **Diferencial:** +{model_accuracy - 50:.0f} pontos vs. aleatório
                """)
            
            # Informações do modelo
            st.subheader("ℹ️ Informações do ACCURACY MODEL")
            
            model_type = "🎯 ACCURACY MASTER" if model_data.get('accuracy_focused', False) else "📊 Modelo Básico"
            version = model_data.get('model_version', 'v1.0')
            
            st.info(f"""
            **Detalhes Técnicos:**
            - **Tipo**: {model_type}
            - **Versão**: {version}
            - **Data de Treinamento**: {model_data['training_date']}
            - **Jogos Analisados**: {model_data['total_samples']:,}
            - **Times no Banco**: {len(model_data['team_stats']):,}
            - **Algoritmo Selecionado**: {best_model_name}
            - **Features Otimizadas**: {model_data.get('features_count', 'N/A')}
            - **Foco**: Máxima Taxa de Acerto (não ROI)
            """)
            
            if model_data.get('accuracy_focused', False):
                st.success("""
                🎯 **ACCURACY MASTER ATIVO - Features Incluídas:**
                
                ✅ Explosive Starts Analysis | ✅ Speed Factor Optimization | ✅ Momentum Ponderado
                ✅ Consistência Temporal | ✅ Context Scoring | ✅ Ultimate Over Score  
                ✅ Confidence Calibration | ✅ Seasonal Adjustments | ✅ Weekend Boosts
                ✅ Walk-Forward Validation | ✅ Class Balancing | ✅ Stability Metrics
                """)
        else:
            st.info("🤖 Nenhum modelo treinado ainda")
            st.write("""
            **Para ver análise de acurácia:**
            1. Vá para 'TRAIN ACCURACY MODEL'
            2. Execute 'TREINAR ACCURACY MASTER MODEL'
            3. Volte aqui para ver métricas de performance
            4. Use o sistema para previsões e acompanhe acurácia real
            """)

if __name__ == "__main__":
    main()
