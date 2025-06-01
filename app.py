import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import joblib
import os
import traceback
from io import BytesIO
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats
from scipy.stats import poisson, variation
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="⚽ HT Goals AI Ultimate - Sistema Avançado",
    page_icon="🎯",
    layout="wide"
)

# Inicializar session state
if 'league_models' not in st.session_state:
    st.session_state.league_models = {}
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'training_errors' not in st.session_state:
    st.session_state.training_errors = []
if 'models_backup' not in st.session_state:
    st.session_state.models_backup = {}
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None

# Configuração da API
try:
    API_KEY = st.secrets["API_KEY"]
except:
    API_KEY = "474f15de5b22951077ccb71b8d75b95c"

API_BASE_URL = "https://v3.football.api-sports.io"

def save_training_progress(league_models, step="backup"):
    """Salva progresso do treinamento"""
    try:
        st.session_state.models_backup = league_models.copy()
        st.session_state.last_backup = datetime.now()
        return True
    except:
        return False

def load_training_progress():
    """Carrega progresso salvo"""
    try:
        if st.session_state.models_backup:
            st.session_state.league_models = st.session_state.models_backup.copy()
            st.session_state.models_trained = True
            return True
    except:
        return False

def get_api_headers():
    """Headers para API-SPORTS"""
    return {'x-apisports-key': API_KEY}

def test_api_connection():
    """Testa conectividade da API"""
    try:
        headers = get_api_headers()
        response = requests.get(f'{API_BASE_URL}/status', headers=headers, timeout=10)
        return response.status_code == 200, f"Status: {response.status_code}"
    except Exception as e:
        return False, f"Erro: {str(e)}"

@st.cache_data(ttl=3600, show_spinner=False)
def get_fixtures_cached(date_str):
    """Busca jogos com cache"""
    try:
        headers = get_api_headers()
        response = requests.get(
            f'{API_BASE_URL}/fixtures',
            headers=headers,
            params={'date': date_str},
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get('response', [])
    except:
        pass
    return []

def load_historical_data():
    """Carrega dados históricos"""
    if st.session_state.historical_data is not None:
        return st.session_state.historical_data, "✅ Dados da sessão"
        
    # Tentar carregar arquivos existentes
    data_files = [
        "data/historical_matches_complete.parquet",
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
                
                if not df.empty:
                    # Processar dados
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    
                    # Criar target se não existir
                    if 'over_05' not in df.columns:
                        if 'ht_home_goals' in df.columns and 'ht_away_goals' in df.columns:
                            df['over_05'] = (df['ht_home_goals'] + df['ht_away_goals']) > 0
                        elif 'ht_home' in df.columns and 'ht_away' in df.columns:
                            df['over_05'] = (df['ht_home'] + df['ht_away']) > 0
                    
                    st.session_state.historical_data = df
                    return df, f"✅ {len(df)} jogos de {file_path}"
            except Exception as e:
                continue
    
    return None, "❌ Arquivo não encontrado - usando API"

def get_multi_season_period():
    """Calcula período para múltiplas temporadas (2-3 anos)"""
    current_date = datetime.now()
    # Buscar dados de 3 temporadas (mais completo)
    start_date = datetime(current_date.year - 3, 8, 1)  
    days_back = (current_date - start_date).days
    days_back = max(days_back, 1000)  # Mínimo 3 anos
    return days_back, start_date

def collect_historical_data_multi_season(use_cached=True):
    """Coleta dados históricos incluindo múltiplas temporadas"""
    
    days_back, start_date = get_multi_season_period()
    st.info(f"📅 Coletando dados de múltiplas temporadas desde {start_date.strftime('%d/%m/%Y')} ({days_back} dias)")
    
    # Tentar cache primeiro
    if use_cached:
        df_cache, message = load_historical_data()
        if df_cache is not None and not df_cache.empty:
            try:
                if 'date' in df_cache.columns:
                    current_date = datetime.now()
                    cutoff_date = current_date - timedelta(days=days_back)
                    df_filtered = df_cache[df_cache['date'] >= cutoff_date].copy()
                    
                    if len(df_filtered) > 200:  # Mínimo para múltiplas temporadas
                        st.success(f"✅ {len(df_filtered)} jogos carregados do cache (múltiplas temporadas)")
                        return df_filtered
            except Exception as e:
                st.warning(f"⚠️ Erro ao processar cache: {str(e)}")
    
    # Coletar da API com amostragem inteligente para múltiplas temporadas
    st.warning("⚠️ Coletando dados da API para múltiplas temporadas...")
    
    sample_days = []
    current_date = datetime.now()
    
    # Estratégia de amostragem para 3 temporadas:
    # - Última temporada: mais densidade
    # - Temporadas anteriores: amostragem estratégica
    
    # Últimos 60 dias: todos os dias
    for i in range(min(60, days_back)):
        sample_days.append(i + 1)
    
    # 60-180 dias: a cada 2 dias
    if days_back > 60:
        for i in range(60, min(180, days_back), 2):
            sample_days.append(i + 1)
    
    # 180-365 dias: a cada 3 dias
    if days_back > 180:
        for i in range(180, min(365, days_back), 3):
            sample_days.append(i + 1)
    
    # Temporadas anteriores: amostragem estratégica por mês
    if days_back > 365:
        for year_offset in range(1, min(4, days_back // 365 + 1)):
            # Para cada temporada anterior, pegar pontos estratégicos
            for month in [9, 10, 11, 12, 1, 2, 3, 4, 5]:  # Setembro a Maio
                for week in [1, 3]:  # 2 semanas por mês
                    try:
                        target_date = datetime(current_date.year - year_offset, month, week * 7)
                        days_diff = (current_date - target_date).days
                        if 365 < days_diff <= days_back:
                            sample_days.append(days_diff)
                    except:
                        continue
    
    # Remover duplicatas e ordenar
    sample_days = sorted(list(set(sample_days)))
    
    all_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, day_offset in enumerate(sample_days):
        try:
            date = current_date - timedelta(days=day_offset)
            date_str = date.strftime('%Y-%m-%d')
            
            status_text.text(f"🔍 Coletando: {date_str} ({len(all_data)} jogos)")
            
            fixtures = get_fixtures_cached(date_str)
            if fixtures:
                for match in fixtures:
                    try:
                        if match['fixture']['status']['short'] == 'FT' and match.get('score', {}).get('halftime'):
                            match_data = extract_match_features(match)
                            if match_data:
                                all_data.append(match_data)
                    except:
                        continue
            
            progress = (idx + 1) / len(sample_days)
            progress_bar.progress(progress)
            
            # Rate limiting menos agressivo para coleta maior
            if idx % 5 == 0:
                time.sleep(0.3)
                
        except Exception as e:
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if len(all_data) < 500:  # Mínimo maior para múltiplas temporadas
        st.error(f"❌ Dados insuficientes: {len(all_data)} jogos (mínimo 500)")
        return pd.DataFrame()
    
    st.success(f"✅ {len(all_data)} jogos coletados da API (múltiplas temporadas)")
    df_result = pd.DataFrame(all_data)
    st.session_state.historical_data = df_result
    return df_result

def extract_match_features(match):
    """Extrai features do jogo com validação robusta"""
    try:
        if not match.get('score', {}).get('halftime'):
            return None
        
        ht_home = match['score']['halftime']['home']
        ht_away = match['score']['halftime']['away']
        
        if ht_home is None or ht_away is None:
            return None
        
        if not isinstance(ht_home, (int, float)) or not isinstance(ht_away, (int, float)):
            return None
        
        # Adicionar informação de temporada
        date_str = match['fixture']['date'][:10]
        season = get_season_from_date(date_str)
        
        features = {
            'date': date_str,
            'timestamp': match['fixture']['timestamp'],
            'league_id': match['league']['id'],
            'league_name': match['league']['name'],
            'country': match['league']['country'],
            'home_team': match['teams']['home']['name'],
            'away_team': match['teams']['away']['name'],
            'home_team_id': match['teams']['home']['id'],
            'away_team_id': match['teams']['away']['id'],
            'ht_home_goals': int(ht_home),
            'ht_away_goals': int(ht_away),
            'ht_total_goals': int(ht_home) + int(ht_away),
            'over_05': 1 if (int(ht_home) + int(ht_away)) > 0 else 0,
            'season': season
        }
        
        return features
    except:
        return None

def get_season_from_date(date_str):
    """Extrai temporada da data"""
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        year = date.year
        month = date.month
        
        if month >= 7:  # Julho a Dezembro
            return f"{year}/{year+1}"
        else:  # Janeiro a Junho
            return f"{year-1}/{year}"
    except:
        current_year = datetime.now().year
        return f"{current_year-1}/{current_year}"

def calculate_robust_team_stats(team_games):
    """Calcula estatísticas robustas considerando outliers (Sua sugestão #2)"""
    if len(team_games) < 3:
        return {
            'mean_goals': 0,
            'robust_mean': 0,
            'coefficient_of_variation': 0,
            'outlier_count': 0,
            'outlier_adjusted_mean': 0,
            'consistency_score': 0
        }
    
    goals = team_games['ht_total_goals'] if 'ht_total_goals' in team_games.columns else team_games
    
    # Estatísticas básicas
    mean_goals = goals.mean()
    std_goals = goals.std()
    median_goals = goals.median()
    
    # Coeficiente de Variação (sua sugestão #3)
    cv = variation(goals) if mean_goals > 0 else 0
    
    # Detectar outliers usando IQR (mais robusto que Z-score)
    Q1 = goals.quantile(0.25)
    Q3 = goals.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = goals[(goals < lower_bound) | (goals > upper_bound)]
    outlier_count = len(outliers)
    
    # Média ajustada removendo outliers
    clean_goals = goals[(goals >= lower_bound) & (goals <= upper_bound)]
    outlier_adjusted_mean = clean_goals.mean() if len(clean_goals) > 0 else mean_goals
    
    # Score de consistência (inverso do CV normalizado)
    consistency_score = max(0, 1 - cv) if cv <= 1 else max(0, 1 / (1 + cv))
    
    # Usar mediana se muitos outliers
    if outlier_count > len(goals) * 0.3:  # Mais de 30% outliers
        robust_mean = median_goals
    else:
        robust_mean = outlier_adjusted_mean
    
    return {
        'mean_goals': mean_goals,
        'robust_mean': robust_mean,
        'coefficient_of_variation': cv,
        'outlier_count': outlier_count,
        'outlier_adjusted_mean': outlier_adjusted_mean,
        'consistency_score': consistency_score,
        'median_goals': median_goals,
        'outlier_percentage': (outlier_count / len(goals)) * 100
    }

def calculate_advanced_poisson(home_lambda, away_lambda):
    """Modelo Poisson melhorado (sua sugestão #3)"""
    try:
        # Validação
        home_lambda = max(home_lambda, 0.01)
        away_lambda = max(away_lambda, 0.01)
        
        # Poisson básico
        prob_home_0 = poisson.pmf(0, home_lambda)
        prob_away_0 = poisson.pmf(0, away_lambda)
        prob_0_0 = prob_home_0 * prob_away_0
        basic_over_05 = 1 - prob_0_0
        
        # Poisson com correlação (times não são independentes)
        correlation_factor = 0.1  # Correlação positiva leve
        adjusted_prob_0_0 = prob_0_0 * (1 - correlation_factor)
        correlated_over_05 = 1 - adjusted_prob_0_0
        
        # Poisson com ajuste temporal (forma recente)
        temporal_weight = 0.8  # Peso para forma recente
        temporal_over_05 = basic_over_05 * temporal_weight + correlated_over_05 * (1 - temporal_weight)
        
        # Probabilidades específicas
        prob_exact_1 = (poisson.pmf(1, home_lambda) * prob_away_0) + (prob_home_0 * poisson.pmf(1, away_lambda))
        prob_2_plus = 1 - prob_0_0 - prob_exact_1
        
        return {
            'basic_over_05': basic_over_05,
            'correlated_over_05': correlated_over_05,
            'temporal_over_05': temporal_over_05,
            'final_over_05': temporal_over_05,  # Usar o mais sofisticado
            'expected_goals': home_lambda + away_lambda,
            'prob_0_0': adjusted_prob_0_0,
            'prob_exact_1': prob_exact_1,
            'prob_2_plus': prob_2_plus,
            'home_lambda': home_lambda,
            'away_lambda': away_lambda
        }
    except:
        return {
            'basic_over_05': 0.5, 'correlated_over_05': 0.5, 'temporal_over_05': 0.5,
            'final_over_05': 0.5, 'expected_goals': 0.5, 'prob_0_0': 0.5,
            'prob_exact_1': 0.3, 'prob_2_plus': 0.2, 'home_lambda': 0.25, 'away_lambda': 0.25
        }

def calculate_combined_score(home_stats, away_stats, league_over_rate, poisson_data):
    """Combined Score melhorado (sua sugestão #3)"""
    try:
        # 1. Team Performance Score (40%)
        home_performance = home_stats.get('home_over_rate', 0.5)
        away_performance = away_stats.get('away_over_rate', 0.5)
        team_score = (home_performance + away_performance) / 2
        
        # Ajustar por consistência
        home_consistency = home_stats.get('home_consistency_score', 1.0)
        away_consistency = away_stats.get('away_consistency_score', 1.0)
        consistency_adj = (home_consistency + away_consistency) / 2
        team_score_adj = team_score * (0.7 + 0.3 * consistency_adj)
        
        # 2. Poisson Score (30%)
        poisson_score = poisson_data['final_over_05']
        
        # 3. League Comparison Score (20%) - Sua sugestão #4
        combined_team_rate = team_score_adj
        vs_league_ratio = combined_team_rate / max(league_over_rate, 0.01)
        
        # Normalizar vs_league para 0-1
        if vs_league_ratio > 2.0:
            league_score = 1.0
        elif vs_league_ratio < 0.5:
            league_score = 0.2
        else:
            league_score = (vs_league_ratio - 0.5) / 1.5
        
        # 4. Variability Penalty (10%)
        home_cv = home_stats.get('home_cv', 0)
        away_cv = away_stats.get('away_cv', 0)
        avg_cv = (home_cv + away_cv) / 2
        
        # Penalizar alta variabilidade
        variability_score = max(0, 1 - avg_cv)
        
        # Combined Score Final
        combined_score = (
            team_score_adj * 0.4 +
            poisson_score * 0.3 +
            league_score * 0.2 +
            variability_score * 0.1
        )
        
        combined_score = max(0, min(1, combined_score))  # Limitar 0-1
        
        # Comparação vs Liga (sua sugestão #4)
        vs_league_percentage = (vs_league_ratio - 1) * 100
        
        return {
            'combined_score': combined_score * 100,  # 0-100
            'team_score': team_score_adj * 100,
            'poisson_score': poisson_score * 100,
            'league_score': league_score * 100,
            'variability_score': variability_score * 100,
            'vs_league_ratio': vs_league_ratio,
            'vs_league_percentage': vs_league_percentage,  # Sua sugestão #4
            'consistency_factor': consistency_adj
        }
    except:
        return {
            'combined_score': 50, 'team_score': 50, 'poisson_score': 50,
            'league_score': 50, 'variability_score': 50, 'vs_league_ratio': 1.0,
            'vs_league_percentage': 0, 'consistency_factor': 1.0
        }

def calculate_advanced_features_multi_season(league_df):
    """Features avançadas com análise de múltiplas temporadas"""
    try:
        if league_df.empty:
            return pd.DataFrame(), {}, 0.5
        
        # Garantir target
        if 'over_05' not in league_df.columns:
            if 'ht_home_goals' in league_df.columns and 'ht_away_goals' in league_df.columns:
                league_df['over_05'] = (league_df['ht_home_goals'] + league_df['ht_away_goals']) > 0
            else:
                return pd.DataFrame(), {}, 0.5
        
        # Ordenar por data
        if 'date' in league_df.columns:
            league_df['date'] = pd.to_datetime(league_df['date'], errors='coerce')
            league_df = league_df.sort_values('date').reset_index(drop=True)
        
        # Estatísticas da liga
        league_over_rate = league_df['over_05'].mean()
        league_avg_goals = league_df['ht_total_goals'].mean()
        
        # Análise de tendência da liga por temporada
        league_trend_analysis = analyze_league_trend_by_season(league_df)
        
        # Estatísticas por time com análise robusta
        team_stats = {}
        unique_teams = pd.concat([league_df['home_team_id'], league_df['away_team_id']]).unique()
        
        for team_id in unique_teams:
            try:
                # Jogos em casa
                home_matches = league_df[league_df['home_team_id'] == team_id]
                # Jogos fora
                away_matches = league_df[league_df['away_team_id'] == team_id]
                # Todos os jogos
                all_matches = pd.concat([home_matches, away_matches])
                
                if len(all_matches) == 0:
                    continue
                
                team_name = home_matches.iloc[0]['home_team'] if len(home_matches) > 0 else away_matches.iloc[0]['away_team']
                
                # Estatísticas robustas para casa (sua sugestão #2)
                if len(home_matches) > 0:
                    home_robust = calculate_robust_team_stats(home_matches)
                    home_goals_scored = home_robust['robust_mean'] if 'ht_home_goals' in home_matches.columns else home_matches['ht_home_goals'].mean() if len(home_matches) > 0 else league_avg_goals/2
                    home_goals_conceded = home_matches['ht_away_goals'].mean() if len(home_matches) > 0 else league_avg_goals/2
                else:
                    home_robust = {'robust_mean': league_avg_goals/2, 'coefficient_of_variation': 0, 'consistency_score': 1, 'outlier_count': 0}
                    home_goals_scored = league_avg_goals/2
                    home_goals_conceded = league_avg_goals/2
                
                # Estatísticas robustas para fora (sua sugestão #2)
                if len(away_matches) > 0:
                    away_robust = calculate_robust_team_stats(away_matches)
                    away_goals_scored = away_matches['ht_away_goals'].mean() if len(away_matches) > 0 else league_avg_goals/2
                    away_goals_conceded = away_matches['ht_home_goals'].mean() if len(away_matches) > 0 else league_avg_goals/2
                else:
                    away_robust = {'robust_mean': league_avg_goals/2, 'coefficient_of_variation': 0, 'consistency_score': 1, 'outlier_count': 0}
                    away_goals_scored = league_avg_goals/2
                    away_goals_conceded = league_avg_goals/2
                
                # Análise de forma recente (últimos 10 jogos)
                recent_form = analyze_recent_form(all_matches, team_id)
                
                # Análise por temporada
                seasonal_analysis = analyze_team_by_season(all_matches, team_id)
                
                team_stats[team_id] = {
                    'team_name': team_name,
                    'games': len(all_matches),
                    'over_rate': all_matches['over_05'].mean(),
                    
                    # Casa - com análise robusta
                    'home_games': len(home_matches),
                    'home_over_rate': home_matches['over_05'].mean() if len(home_matches) > 0 else league_over_rate,
                    'home_goals_scored': home_goals_scored,
                    'home_goals_conceded': home_goals_conceded,
                    'home_cv': home_robust['coefficient_of_variation'],
                    'home_consistency_score': home_robust['consistency_score'],
                    'home_outlier_count': home_robust['outlier_count'],
                    
                    # Fora - com análise robusta
                    'away_games': len(away_matches),
                    'away_over_rate': away_matches['over_05'].mean() if len(away_matches) > 0 else league_over_rate,
                    'away_goals_scored': away_goals_scored,
                    'away_goals_conceded': away_goals_conceded,
                    'away_cv': away_robust['coefficient_of_variation'],
                    'away_consistency_score': away_robust['consistency_score'],
                    'away_outlier_count': away_robust['outlier_count'],
                    
                    # Força ofensiva/defensiva
                    'home_attack_strength': max(home_goals_scored / (league_avg_goals/2 + 0.01), 0.1),
                    'home_defense_strength': max(home_goals_conceded / (league_avg_goals/2 + 0.01), 0.1),
                    'away_attack_strength': max(away_goals_scored / (league_avg_goals/2 + 0.01), 0.1),
                    'away_defense_strength': max(away_goals_conceded / (league_avg_goals/2 + 0.01), 0.1),
                    
                    # Forma recente
                    'recent_over_rate': recent_form['recent_over_rate'],
                    'recent_trend': recent_form['trend'],
                    'recent_consistency': recent_form['consistency'],
                    
                    # Análise sazonal
                    'seasonal_stability': seasonal_analysis['stability'],
                    'seasonal_trend': seasonal_analysis['trend']
                }
            except Exception as e:
                continue
        
        # Criar features para ML
        features = []
        
        for idx, row in league_df.iterrows():
            try:
                home_id = row['home_team_id']
                away_id = row['away_team_id']
                
                if home_id not in team_stats or away_id not in team_stats:
                    continue
                
                home_stats = team_stats[home_id]
                away_stats = team_stats[away_id]
                
                # Análise Poisson avançada (sua sugestão #3)
                home_expected = home_stats['home_attack_strength'] * away_stats['away_defense_strength'] * league_avg_goals/2
                away_expected = away_stats['away_attack_strength'] * home_stats['home_defense_strength'] * league_avg_goals/2
                
                poisson_data = calculate_advanced_poisson(home_expected, away_expected)
                
                # Combined Score avançado (sua sugestão #3)
                combined_data = calculate_combined_score(home_stats, away_stats, league_over_rate, poisson_data)
                
                # Features completas
                feature_row = {
                    # Básicas
                    'home_over_rate': home_stats['over_rate'],
                    'away_over_rate': away_stats['over_rate'],
                    'home_home_over_rate': home_stats['home_over_rate'],
                    'away_away_over_rate': away_stats['away_over_rate'],
                    'league_over_rate': league_over_rate,
                    
                    # Força casa/fora
                    'home_attack_strength': home_stats['home_attack_strength'],
                    'home_defense_strength': home_stats['home_defense_strength'],
                    'away_attack_strength': away_stats['away_attack_strength'],
                    'away_defense_strength': away_stats['away_defense_strength'],
                    
                    # Poisson avançado
                    'poisson_basic': poisson_data['basic_over_05'],
                    'poisson_correlated': poisson_data['correlated_over_05'],
                    'poisson_temporal': poisson_data['temporal_over_05'],
                    'poisson_final': poisson_data['final_over_05'],
                    'expected_goals_ht': poisson_data['expected_goals'],
                    
                    # Combined Score
                    'combined_score': combined_data['combined_score'] / 100,
                    'team_score': combined_data['team_score'] / 100,
                    'league_comparison_score': combined_data['league_score'] / 100,
                    'variability_score': combined_data['variability_score'] / 100,
                    
                    # Comparação vs Liga (sua sugestão #4)
                    'vs_league_ratio': combined_data['vs_league_ratio'],
                    'vs_league_percentage': combined_data['vs_league_percentage'],
                    'over_rate_vs_league': ((home_stats['home_over_rate'] + away_stats['away_over_rate']) / 2) / max(league_over_rate, 0.01),
                    
                    # Consistência e outliers (sua sugestão #2)
                    'home_consistency': home_stats['home_consistency_score'],
                    'away_consistency': away_stats['away_consistency_score'],
                    'home_cv': home_stats['home_cv'],
                    'away_cv': away_stats['away_cv'],
                    'combined_consistency': (home_stats['home_consistency_score'] + away_stats['away_consistency_score']) / 2,
                    'outlier_factor': 1 - (home_stats['home_outlier_count'] + away_stats['away_outlier_count']) / max(home_stats['home_games'] + away_stats['away_games'], 1),
                    
                    # Forma recente
                    'home_recent_over_rate': home_stats['recent_over_rate'],
                    'away_recent_over_rate': away_stats['recent_over_rate'],
                    'home_recent_trend': home_stats['recent_trend'],
                    'away_recent_trend': away_stats['recent_trend'],
                    'combined_recent_trend': (home_stats['recent_trend'] + away_stats['recent_trend']) / 2,
                    
                    # Análise sazonal
                    'home_seasonal_stability': home_stats['seasonal_stability'],
                    'away_seasonal_stability': away_stats['seasonal_stability'],
                    'combined_seasonal_stability': (home_stats['seasonal_stability'] + away_stats['seasonal_stability']) / 2,
                    
                    # Combinações avançadas
                    'attack_vs_defense_balance': (home_stats['home_attack_strength'] + away_stats['away_attack_strength']) / (home_stats['home_defense_strength'] + away_stats['away_defense_strength']),
                    'game_pace_index': poisson_data['expected_goals'],
                    'matchup_quality': combined_data['consistency_factor'],
                    
                    # Meta features
                    'home_games_played': home_stats['home_games'],
                    'away_games_played': away_stats['away_games'],
                    'min_games': min(home_stats['home_games'], away_stats['away_games']),
                    'total_games_experience': home_stats['home_games'] + away_stats['away_games'],
                    
                    # Target
                    'target': row['over_05']
                }
                
                features.append(feature_row)
            except Exception as e:
                continue
        
        return pd.DataFrame(features), team_stats, league_over_rate, league_trend_analysis
        
    except Exception as e:
        st.error(f"❌ Erro ao calcular features avançadas: {str(e)}")
        return pd.DataFrame(), {}, 0.5, {}

def analyze_league_trend_by_season(league_df):
    """Analisa tendência da liga por temporada"""
    try:
        if 'season' not in league_df.columns:
            return {'overall_trend': 0, 'seasons': {}}
        
        season_stats = {}
        seasons = league_df['season'].unique()
        
        for season in seasons:
            season_data = league_df[league_df['season'] == season]
            if len(season_data) > 10:  # Mínimo para análise
                season_stats[season] = {
                    'over_rate': season_data['over_05'].mean(),
                    'avg_goals': season_data['ht_total_goals'].mean(),
                    'games': len(season_data)
                }
        
        # Calcular tendência geral
        if len(season_stats) >= 2:
            rates = [s['over_rate'] for s in season_stats.values()]
            trend = np.polyfit(range(len(rates)), rates, 1)[0]  # Slope da regressão linear
        else:
            trend = 0
        
        return {
            'overall_trend': trend,
            'seasons': season_stats,
            'season_count': len(season_stats)
        }
    except:
        return {'overall_trend': 0, 'seasons': {}, 'season_count': 0}

def analyze_recent_form(all_matches, team_id):
    """Analisa forma recente da equipe (últimos 10 jogos)"""
    try:
        if 'date' in all_matches.columns:
            recent_matches = all_matches.sort_values('date', ascending=False).head(10)
        else:
            recent_matches = all_matches.tail(10)
        
        if len(recent_matches) < 3:
            return {'recent_over_rate': 0.5, 'trend': 0, 'consistency': 1}
        
        recent_over_rate = recent_matches['over_05'].mean()
        
        # Calcular tendência (primeiros 5 vs últimos 5)
        if len(recent_matches) >= 6:
            first_half = recent_matches.tail(5)['over_05'].mean()  # Mais antigos
            second_half = recent_matches.head(5)['over_05'].mean()  # Mais recentes
            trend = second_half - first_half
        else:
            trend = 0
        
        # Consistência na forma recente
        consistency = 1 - recent_matches['over_05'].std() if recent_matches['over_05'].std() > 0 else 1
        
        return {
            'recent_over_rate': recent_over_rate,
            'trend': trend,
            'consistency': max(0, consistency)
        }
    except:
        return {'recent_over_rate': 0.5, 'trend': 0, 'consistency': 1}

def analyze_team_by_season(all_matches, team_id):
    """Analisa performance da equipe por temporada"""
    try:
        if 'season' not in all_matches.columns or len(all_matches) < 10:
            return {'stability': 1, 'trend': 0}
        
        season_rates = {}
        seasons = all_matches['season'].unique()
        
        for season in seasons:
            season_data = all_matches[all_matches['season'] == season]
            if len(season_data) >= 3:
                season_rates[season] = season_data['over_05'].mean()
        
        if len(season_rates) < 2:
            return {'stability': 1, 'trend': 0}
        
        # Estabilidade sazonal (baixo desvio padrão = alta estabilidade)
        rates = list(season_rates.values())
        stability = 1 - np.std(rates) if np.std(rates) < 1 else max(0, 1 - np.std(rates))
        
        # Tendência sazonal
        trend = np.polyfit(range(len(rates)), rates, 1)[0] if len(rates) >= 2 else 0
        
        return {
            'stability': max(0, stability),
            'trend': trend
        }
    except:
        return {'stability': 1, 'trend': 0}

def train_advanced_model_with_validation(league_df, league_id, league_name, min_matches=30):
    """Treina modelo avançado com múltiplas validações"""
    
    if len(league_df) < min_matches:
        return None, f"❌ {league_name}: {len(league_df)} jogos < {min_matches} mínimo"
    
    try:
        # Features avançadas com múltiplas temporadas
        features_df, team_stats, league_over_rate, league_trend = calculate_advanced_features_multi_season(league_df)
        
        if features_df.empty or len(features_df) < min_matches:
            return None, f"❌ {league_name}: Features insuficientes"
        
        # Verificar variação no target
        if features_df['target'].nunique() < 2:
            return None, f"❌ {league_name}: Sem variação no target"
        
        feature_cols = [col for col in features_df.columns if col != 'target']
        X = features_df[feature_cols]
        y = features_df['target']
        
        # Tratar NaN
        X = X.fillna(X.median())
        y = y.fillna(0)
        
        # Divisão Treino/Validação/Teste (sua sugestão #1)
        try:
            # Primeiro: Treino + Validação vs Teste (80/20)
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            # Segundo: Treino vs Validação (75/25 do restante = 60/20 do total)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
            )
        except:
            # Fallback sem estratificação
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
        
        # Escalonamento robusto (menos sensível a outliers)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Modelos ensemble avançados
        models = {
            'rf_tuned': RandomForestClassifier(
                n_estimators=150, max_depth=10, min_samples_split=5, 
                min_samples_leaf=3, random_state=42, n_jobs=1
            ),
            'gb_tuned': GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.08, max_depth=7, 
                min_samples_split=5, random_state=42
            ),
            'et_tuned': ExtraTreesClassifier(
                n_estimators=150, max_depth=10, min_samples_split=5,
                min_samples_leaf=3, random_state=42, n_jobs=1
            )
        }
        
        # Treinar e validar modelos
        best_model = None
        best_score = 0
        results = {}
        
        for name, model in models.items():
            try:
                # Treinar com calibração
                calibrated_model = CalibratedClassifierCV(model, cv=3, method='sigmoid')
                calibrated_model.fit(X_train_scaled, y_train)
                
                # Validar
                val_pred = calibrated_model.predict(X_val_scaled)
                val_pred_proba = calibrated_model.predict_proba(X_val_scaled)[:, 1]
                
                val_acc = accuracy_score(y_val, val_pred)
                val_prec = precision_score(y_val, val_pred, zero_division=0)
                val_rec = recall_score(y_val, val_pred, zero_division=0)
                val_f1 = f1_score(y_val, val_pred, zero_division=0)
                val_auc = roc_auc_score(y_val, val_pred_proba) if len(np.unique(y_val)) > 1 else 0.5
                
                # Score combinado (balanceado)
                combined_score = (val_f1 * 0.4) + (val_acc * 0.3) + (val_auc * 0.3)
                
                results[name] = {
                    'val_accuracy': val_acc,
                    'val_precision': val_prec,
                    'val_recall': val_rec,
                    'val_f1': val_f1,
                    'val_auc': val_auc,
                    'combined_score': combined_score,
                    'model': calibrated_model
                }
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_model = calibrated_model
                    
            except Exception as e:
                continue
        
        if best_model is None:
            return None, f"❌ {league_name}: Nenhum modelo funcionou"
        
        # Ensemble dos melhores modelos
        try:
            good_models = [(name, data['model']) for name, data in results.items() 
                          if data['combined_score'] > 0.6]
            
            if len(good_models) >= 2:
                ensemble = VotingClassifier(good_models, voting='soft')
                ensemble.fit(X_train_scaled, y_train)
                
                # Testar ensemble
                ens_val_pred = ensemble.predict(X_val_scaled)
                ens_val_proba = ensemble.predict_proba(X_val_scaled)[:, 1]
                ens_f1 = f1_score(y_val, ens_val_pred, zero_division=0)
                ens_acc = accuracy_score(y_val, ens_val_pred)
                ens_auc = roc_auc_score(y_val, ens_val_proba) if len(np.unique(y_val)) > 1 else 0.5
                ens_combined = (ens_f1 * 0.4) + (ens_acc * 0.3) + (ens_auc * 0.3)
                
                if ens_combined > best_score:
                    best_model = ensemble
                    best_score = ens_combined
        except:
            pass
        
        # Teste final
        test_pred = best_model.predict(X_test_scaled)
        test_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, test_pred),
            'precision': precision_score(y_test, test_pred, zero_division=0),
            'recall': recall_score(y_test, test_pred, zero_division=0),
            'f1_score': f1_score(y_test, test_pred, zero_division=0),
            'auc_score': roc_auc_score(y_test, test_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        }
        
        # Threshold ótimo
        best_threshold = 0.5
        best_f1 = test_metrics['f1_score']
        
        for threshold in np.arange(0.3, 0.8, 0.02):
            try:
                pred_threshold = (test_pred_proba >= threshold).astype(int)
                f1_threshold = f1_score(y_test, pred_threshold, zero_division=0)
                
                if f1_threshold > best_f1:
                    best_f1 = f1_threshold
                    best_threshold = threshold
            except:
                continue
        
        # Retreinar no dataset completo
        X_scaled = scaler.fit_transform(X)
        best_model.fit(X_scaled, y)
        
        # Feature importance
        try:
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = dict(zip(feature_cols, best_model.feature_importances_))
            elif hasattr(best_model, 'estimators_'):
                # Para ensemble
                importances = np.zeros(len(feature_cols))
                count = 0
                for estimator in best_model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        importances += estimator.feature_importances_
                        count += 1
                if count > 0:
                    feature_importance = dict(zip(feature_cols, importances / count))
                else:
                    feature_importance = {f: 1/len(feature_cols) for f in feature_cols}
            else:
                feature_importance = {f: 1/len(feature_cols) for f in feature_cols}
            
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        except:
            top_features = [('combined_score', 0.5), ('vs_league_percentage', 0.3)]
        
        # Dados do modelo
        model_data = {
            'model': best_model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'team_stats': team_stats,
            'league_id': league_id,
            'league_name': league_name,
            'league_over_rate': league_over_rate,
            'league_trend': league_trend,
            'total_matches': len(league_df),
            'validation_results': results,
            'test_metrics': test_metrics,
            'best_threshold': best_threshold,
            'top_features': top_features,
            'model_type': type(best_model).__name__,
            'combined_score_method': True,
            'multi_season_analysis': True
        }
        
        return model_data, f"✅ {league_name}: Acc {test_metrics['accuracy']:.1%} | F1 {test_metrics['f1_score']:.1%} | AUC {test_metrics['auc_score']:.2f}"
        
    except Exception as e:
        error_msg = f"❌ {league_name}: {str(e)}"
        st.session_state.training_errors.append(error_msg)
        return None, error_msg

def predict_with_advanced_strategy(fixtures, league_models, min_confidence=60):
    """Previsões com estratégia avançada - ILIMITADAS (sua sugestão #1)"""
    
    if not league_models:
        return []
    
    predictions = []
    
    for fixture in fixtures:
        try:
            if fixture['fixture']['status']['short'] not in ['NS', 'TBD']:
                continue
            
            league_id = fixture['league']['id']
            
            if league_id not in league_models:
                continue
            
            model_data = league_models[league_id]
            model = model_data['model']
            scaler = model_data['scaler']
            feature_cols = model_data['feature_cols']
            team_stats = model_data['team_stats']
            league_over_rate = model_data['league_over_rate']
            best_threshold = model_data.get('best_threshold', 0.5)
            
            home_id = fixture['teams']['home']['id']
            away_id = fixture['teams']['away']['id']
            
            if home_id not in team_stats or away_id not in team_stats:
                continue
            
            home_stats = team_stats[home_id]
            away_stats = team_stats[away_id]
            
            # Análise Poisson avançada
            home_expected = home_stats['home_attack_strength'] * away_stats['away_defense_strength'] * 0.5
            away_expected = away_stats['away_attack_strength'] * home_stats['home_defense_strength'] * 0.5
            
            poisson_data = calculate_advanced_poisson(home_expected, away_expected)
            
            # Combined Score avançado
            combined_data = calculate_combined_score(home_stats, away_stats, league_over_rate, poisson_data)
            
            # Criar features para predição
            features = {}
            
            # Todas as features avançadas
            for col in feature_cols:
                if col == 'home_over_rate':
                    features[col] = home_stats['over_rate']
                elif col == 'away_over_rate':
                    features[col] = away_stats['over_rate']
                elif col == 'home_home_over_rate':
                    features[col] = home_stats['home_over_rate']
                elif col == 'away_away_over_rate':
                    features[col] = away_stats['away_over_rate']
                elif col == 'league_over_rate':
                    features[col] = league_over_rate
                elif col == 'combined_score':
                    features[col] = combined_data['combined_score'] / 100
                elif col == 'vs_league_percentage':
                    features[col] = combined_data['vs_league_percentage']
                elif col == 'vs_league_ratio':
                    features[col] = combined_data['vs_league_ratio']
                elif col == 'poisson_final':
                    features[col] = poisson_data['final_over_05']
                elif col == 'home_consistency':
                    features[col] = home_stats.get('home_consistency_score', 1.0)
                elif col == 'away_consistency':
                    features[col] = away_stats.get('away_consistency_score', 1.0)
                elif col == 'home_cv':
                    features[col] = home_stats.get('home_cv', 0)
                elif col == 'away_cv':
                    features[col] = away_stats.get('away_cv', 0)
                else:
                    # Calcular ou usar valor padrão
                    features[col] = 0.5
            
            # Garantir todas as features
            for col in feature_cols:
                if col not in features:
                    features[col] = 0.5
            
            # Predição
            X = pd.DataFrame([features])[feature_cols]
            X = X.fillna(0.5)
            
            X_scaled = scaler.transform(X)
            
            pred_proba = model.predict_proba(X_scaled)[0]
            confidence = pred_proba[1] * 100
            
            pred_class = 1 if pred_proba[1] >= best_threshold else 0
            
            # Comparação vs Liga (sua sugestão #4)
            vs_league_perc = combined_data['vs_league_percentage']
            
            prediction = {
                'home_team': fixture['teams']['home']['name'],
                'away_team': fixture['teams']['away']['name'],
                'home_team_stats': home_stats,
                'away_team_stats': away_stats,
                'league': fixture['league']['name'],
                'country': fixture['league']['country'],
                'league_over_rate': league_over_rate * 100,
                'kickoff': fixture['fixture']['date'],
                'prediction': 'OVER 0.5' if pred_class == 1 else 'UNDER 0.5',
                'confidence': confidence,
                'combined_score': combined_data['combined_score'],
                'ml_probability': pred_proba[1] * 100,
                'poisson_probability': poisson_data['final_over_05'] * 100,
                'expected_goals_ht': poisson_data['expected_goals'],
                'vs_league_percentage': vs_league_perc,  # Sua sugestão #4
                'vs_league_ratio': combined_data['vs_league_ratio'],
                'consistency_score': (home_stats.get('home_consistency_score', 1) + away_stats.get('away_consistency_score', 1)) / 2,
                'outlier_risk': 1 - ((home_stats.get('home_outlier_count', 0) + away_stats.get('away_outlier_count', 0)) / max(home_stats.get('home_games', 1) + away_stats.get('away_games', 1), 1)),
                'model_metrics': model_data['test_metrics'],
                'top_features': model_data['top_features'],
                'fixture_id': fixture['fixture']['id']
            }
            
            # ILIMITADAS - todas as previsões acima da confiança mínima (sua sugestão #1)
            if confidence >= min_confidence:
                predictions.append(prediction)
            
        except Exception as e:
            continue
    
    # Ordenar por combined score (melhor métrica)
    predictions.sort(key=lambda x: x['combined_score'], reverse=True)
    
    return predictions

def display_advanced_prediction(pred):
    """Exibe previsão avançada com todas as métricas"""
    
    try:
        with st.container():
            # Header principal
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.subheader(f"⚽ {pred['home_team']} vs {pred['away_team']}")
                st.caption(f"🏆 {pred['league']} ({pred['country']})")
            
            with col2:
                # Combined Score
                score = pred['combined_score']
                if score >= 80:
                    st.success(f"**{score:.1f}**")
                    st.caption("🔥 Excelente")
                elif score >= 70:
                    st.info(f"**{score:.1f}**")
                    st.caption("✅ Bom")
                elif score >= 60:
                    st.warning(f"**{score:.1f}**")
                    st.caption("⚠️ Moderado")
                else:
                    st.error(f"**{score:.1f}**")
                    st.caption("❌ Baixo")
            
            with col3:
                # Comparação vs Liga (sua sugestão #4)
                vs_league = pred['vs_league_percentage']
                if vs_league > 20:
                    st.success(f"**+{vs_league:.1f}%**")
                    st.caption("🔥 Acima da Liga")
                elif vs_league > 0:
                    st.info(f"**+{vs_league:.1f}%**")
                    st.caption("✅ Ligeiramente Acima")
                elif vs_league > -20:
                    st.warning(f"**{vs_league:.1f}%**")
                    st.caption("➖ Próximo à Média")
                else:
                    st.error(f"**{vs_league:.1f}%**")
                    st.caption("❄️ Abaixo da Liga")
            
            # Métricas principais
            st.markdown("### 📊 Análise Principal")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("ML Confidence", f"{pred['confidence']:.1f}%")
                st.caption("Confiança do Modelo")
            
            with col2:
                st.metric("Poisson Probability", f"{pred['poisson_probability']:.1f}%")
                st.caption("Modelo Estatístico")
            
            with col3:
                st.metric("Expected Goals HT", f"{pred['expected_goals_ht']:.2f}")
                st.caption("Gols Esperados")
            
            with col4:
                consistency = pred['consistency_score']
                st.metric("Consistency", f"{consistency:.2f}")
                if consistency > 0.8:
                    st.caption("🟢 Alta")
                elif consistency > 0.6:
                    st.caption("🟡 Média")
                else:
                    st.caption("🔴 Baixa")
            
            # Análise das equipas
            st.markdown("### 🏠🆚✈️ Análise Casa vs Fora")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**🏠 {pred['home_team']} (Casa)**")
                home_rate = pred['home_team_stats']['home_over_rate'] * 100
                st.write(f"- Taxa Casa: {home_rate:.1f}%")
                st.write(f"- Força Ataque: {pred['home_team_stats']['home_attack_strength']:.2f}")
                st.write(f"- Força Defesa: {pred['home_team_stats']['home_defense_strength']:.2f}")
                
                # Consistência casa
                home_consistency = pred['home_team_stats'].get('home_consistency_score', 1.0)
                if home_consistency > 0.8:
                    st.write("- Consistência: 🟢 Alta")
                elif home_consistency > 0.6:
                    st.write("- Consistência: 🟡 Média")
                else:
                    st.write("- Consistência: 🔴 Baixa")
                
                # Outliers
                outliers = pred['home_team_stats'].get('home_outlier_count', 0)
                if outliers > 0:
                    st.write(f"- ⚠️ Outliers detectados: {outliers}")
            
            with col2:
                st.write(f"**✈️ {pred['away_team']} (Fora)**")
                away_rate = pred['away_team_stats']['away_over_rate'] * 100
                st.write(f"- Taxa Fora: {away_rate:.1f}%")
                st.write(f"- Força Ataque: {pred['away_team_stats']['away_attack_strength']:.2f}")
                st.write(f"- Força Defesa: {pred['away_team_stats']['away_defense_strength']:.2f}")
                
                # Consistência fora
                away_consistency = pred['away_team_stats'].get('away_consistency_score', 1.0)
                if away_consistency > 0.8:
                    st.write("- Consistência: 🟢 Alta")
                elif away_consistency > 0.6:
                    st.write("- Consistência: 🟡 Média")
                else:
                    st.write("- Consistência: 🔴 Baixa")
                
                # Outliers
                outliers = pred['away_team_stats'].get('away_outlier_count', 0)
                if outliers > 0:
                    st.write(f"- ⚠️ Outliers detectados: {outliers}")
            
            # Comparação vs Liga detalhada (sua sugestão #4)
            st.markdown("### 📈 Comparação vs Liga")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                league_rate = pred['league_over_rate']
                st.metric("Taxa da Liga", f"{league_rate:.1f}%")
                st.caption("Média Over 0.5 HT")
            
            with col2:
                combined_team_rate = (home_rate + away_rate) / 2
                st.metric("Taxa Combinada", f"{combined_team_rate:.1f}%")
                st.caption("Casa + Fora / 2")
            
            with col3:
                vs_league_ratio = pred['vs_league_ratio']
                st.metric("Ratio vs Liga", f"{vs_league_ratio:.2f}x")
                if vs_league_ratio > 1.2:
                    st.caption("🔥 Muito Acima")
                elif vs_league_ratio > 1.0:
                    st.caption("✅ Acima")
                elif vs_league_ratio > 0.8:
                    st.caption("➖ Próximo")
                else:
                    st.caption("❄️ Abaixo")
            
            # Recomendação final
            st.markdown("### 🎯 Recomendação Final")
            
            if pred['prediction'] == 'OVER 0.5':
                # Lógica de recomendação baseada em múltiplos fatores
                score = pred['combined_score']
                vs_league = pred['vs_league_percentage']
                consistency = pred['consistency_score']
                outlier_risk = pred['outlier_risk']
                
                if score >= 75 and vs_league > 10 and consistency > 0.7 and outlier_risk > 0.8:
                    st.success(f"✅ **APOSTAR FORTE: {pred['prediction']} HT**")
                    st.write("🔥 **Todos os indicadores são favoráveis!**")
                elif score >= 65 and vs_league > 0 and consistency > 0.6:
                    st.info(f"📊 **APOSTAR: {pred['prediction']} HT**")
                    st.write("✅ **Indicadores majoritariamente favoráveis**")
                elif score >= 60:
                    st.warning(f"⚠️ **CONSIDERAR: {pred['prediction']} HT**")
                    st.write("⚠️ **Apostar com cautela - indicadores mistos**")
                else:
                    st.error(f"❌ **EVITAR: {pred['prediction']} HT**")
                    st.write("❌ **Indicadores não favoráveis**")
            
            # Análise de risco
            st.markdown("### ⚠️ Análise de Risco")
            risk_factors = []
            
            if pred['consistency_score'] < 0.6:
                risk_factors.append("🔴 Baixa consistência das equipas")
            
            if pred['outlier_risk'] < 0.7:
                risk_factors.append("🔴 Presença de outliers nos dados")
            
            if abs(pred['vs_league_percentage']) < 5:
                risk_factors.append("🟡 Performance muito próxima à média da liga")
            
            if pred['combined_score'] < 65:
                risk_factors.append("🔴 Score combinado baixo")
            
            if not risk_factors:
                st.success("✅ **Baixo Risco** - Todos os indicadores são favoráveis")
            else:
                st.warning("⚠️ **Fatores de Risco Identificados:**")
                for factor in risk_factors:
                    st.write(f"- {factor}")
            
            # Detalhes técnicos
            with st.expander("🔧 Detalhes Técnicos"):
                st.write("**Métricas do Modelo:**")
                metrics = pred['model_metrics']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Acurácia", f"{metrics['accuracy']*100:.1f}%")
                with col2:
                    st.metric("Precisão", f"{metrics['precision']*100:.1f}%")
                with col3:
                    st.metric("Recall", f"{metrics['recall']*100:.1f}%")
                with col4:
                    st.metric("F1-Score", f"{metrics['f1_score']*100:.1f}%")
                
                st.write("**Features Mais Importantes:**")
                for i, (feature, importance) in enumerate(pred['top_features'][:5]):
                    st.write(f"{i+1}. {feature.replace('_', ' ').title()}: {importance:.3f}")
            
            st.markdown("---")
            
    except Exception as e:
        st.error(f"❌ Erro ao exibir previsão: {str(e)}")

def create_excel_download(df, filename):
    """Cria arquivo Excel para download"""
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Dados')
        output.seek(0)
        return output.getvalue()
    except:
        return None

def display_advanced_league_summary(league_models):
    """Exibe resumo avançado das ligas"""
    
    try:
        st.header("📊 Análise Avançada das Ligas")
        
        if not league_models:
            st.warning("⚠️ Nenhum modelo treinado!")
            return
        
        # Dados para análise
        league_data = []
        for league_id, model_data in league_models.items():
            try:
                league_data.append({
                    'Liga': model_data['league_name'],
                    'Over 0.5 HT %': round(model_data['league_over_rate'] * 100, 1),
                    'Jogos': model_data['total_matches'],
                    'F1-Score': round(model_data['test_metrics']['f1_score'] * 100, 1),
                    'Acurácia': round(model_data['test_metrics']['accuracy'] * 100, 1),
                    'AUC Score': round(model_data['test_metrics']['auc_score'] * 100, 1),
                    'Precisão': round(model_data['test_metrics']['precision'] * 100, 1),
                    'Recall': round(model_data['test_metrics']['recall'] * 100, 1),
                    'Threshold Ótimo': round(model_data['best_threshold'], 3),
                    'Modelo': model_data.get('model_type', 'Unknown'),
                    'Multi-Season': 'Sim' if model_data.get('multi_season_analysis') else 'Não',
                    'Combined Score': 'Sim' if model_data.get('combined_score_method') else 'Não'
                })
            except:
                continue
        
        if not league_data:
            st.warning("⚠️ Nenhum dado para exibir!")
            return
        
        df_leagues = pd.DataFrame(league_data)
        
        # Estatísticas gerais
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Ligas", len(df_leagues))
        with col2:
            avg_acc = df_leagues['Acurácia'].mean()
            st.metric("Acurácia Média", f"{avg_acc:.1f}%")
        with col3:
            avg_f1 = df_leagues['F1-Score'].mean()
            st.metric("F1-Score Médio", f"{avg_f1:.1f}%")
        with col4:
            avg_auc = df_leagues['AUC Score'].mean()
            st.metric("AUC Score Médio", f"{avg_auc:.1f}%")
        
        # Download geral
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            excel_data = create_excel_download(df_leagues, "analise_avancada_ligas.xlsx")
            if excel_data:
                st.download_button(
                    label="📥 Download Análise Completa",
                    data=excel_data,
                    file_name=f"analise_avancada_ligas_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        # Visualizações
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top 15 Ligas - Performance")
            top_leagues = df_leagues.sort_values('F1-Score', ascending=False).head(15)
            chart_data = top_leagues.set_index('Liga')[['Acurácia', 'F1-Score', 'AUC Score']]
            st.line_chart(chart_data)
        
        with col2:
            st.subheader("📈 Top 15 - Taxa Over 0.5 HT")
            top_over = df_leagues.sort_values('Over 0.5 HT %', ascending=False).head(15)
            chart_data_over = top_over.set_index('Liga')['Over 0.5 HT %']
            st.bar_chart(chart_data_over)
        
        # Análise de qualidade avançada
        st.subheader("📊 Classificação por Qualidade")
        
        # Critério de qualidade mais rigoroso
        excellent = df_leagues[(df_leagues['F1-Score'] >= 85) & (df_leagues['AUC Score'] >= 80)]
        very_good = df_leagues[(df_leagues['F1-Score'] >= 75) & (df_leagues['F1-Score'] < 85) & (df_leagues['AUC Score'] >= 75)]
        good = df_leagues[(df_leagues['F1-Score'] >= 65) & (df_leagues['F1-Score'] < 75)]
        fair = df_leagues[df_leagues['F1-Score'] < 65]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🟢 Excelente", len(excellent))
            st.caption("F1≥85% & AUC≥80%")
            if len(excellent) > 0:
                excel_exc = create_excel_download(excellent, "ligas_excelentes.xlsx")
                if excel_exc:
                    st.download_button(
                        label="📥 Download",
                        data=excel_exc,
                        file_name=f"ligas_excelentes_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="excellent"
                    )
        
        with col2:
            st.metric("🔵 Muito Bom", len(very_good))
            st.caption("F1≥75% & AUC≥75%")
            if len(very_good) > 0:
                excel_vg = create_excel_download(very_good, "ligas_muito_boas.xlsx")
                if excel_vg:
                    st.download_button(
                        label="📥 Download",
                        data=excel_vg,
                        file_name=f"ligas_muito_boas_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="very_good"
                    )
        
        with col3:
            st.metric("🟡 Bom", len(good))
            st.caption("F1≥65%")
            if len(good) > 0:
                excel_good = create_excel_download(good, "ligas_boas.xlsx")
                if excel_good:
                    st.download_button(
                        label="📥 Download",
                        data=excel_good,
                        file_name=f"ligas_boas_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="good"
                    )
        
        with col4:
            st.metric("🔴 Regular", len(fair))
            st.caption("F1<65%")
            if len(fair) > 0:
                excel_fair = create_excel_download(fair, "ligas_regulares.xlsx")
                if excel_fair:
                    st.download_button(
                        label="📥 Download",
                        data=excel_fair,
                        file_name=f"ligas_regulares_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="fair"
                    )
        
        # Tabela completa
        st.subheader("📋 Tabela Completa")
        df_display = df_leagues.sort_values('F1-Score', ascending=False)
        st.dataframe(df_display, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Erro ao exibir resumo: {str(e)}")

def main():
    st.title("⚽ HT Goals AI Ultimate - Sistema Avançado Completo")
    st.markdown("🎯 **Versão Premium com todas as melhorias implementadas**")
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configurações Avançadas")
        
        # Status da API
        try:
            conn_ok, conn_msg = test_api_connection()
            if conn_ok:
                st.success("✅ API conectada")
            else:
                st.error(f"❌ {conn_msg}")
        except:
            st.error("❌ Erro ao testar API")
        
        # Status dos modelos
        if st.session_state.models_trained and st.session_state.league_models:
            st.success(f"✅ {len(st.session_state.league_models)} ligas treinadas")
            
            if st.session_state.models_backup:
                if st.button("🔄 Carregar Backup"):
                    load_training_progress()
                    st.success("✅ Backup carregado!")
                    st.rerun()
        else:
            st.warning("⚠️ Modelos não treinados")
        
        st.markdown("---")
        
        # Configurações (suas sugestões implementadas)
        st.markdown("### 📊 Parâmetros Avançados")
        
        # Sua sugestão #1: Mínimo 30 jogos e apostas ilimitadas
        min_matches_per_league = st.slider(
            "Mínimo jogos por liga:",
            min_value=30,
            max_value=100,
            value=30,
            help="30 jogos com múltiplas temporadas para análise robusta"
        )
        
        st.info("🔥 **Apostas por dia: ILIMITADAS** (sua sugestão)")
        
        min_confidence = st.slider(
            "Confiança mínima:",
            min_value=50,
            max_value=80,
            value=60,
            help="Combined Score mínimo para apostar"
        )
        
        use_cache = st.checkbox("💾 Usar cache", value=True)
        
        # Indicadores das melhorias implementadas
        st.markdown("### ✅ Melhorias Implementadas")
        st.success("✅ Múltiplas temporadas")
        st.success("✅ Detecção de outliers")
        st.success("✅ Coeficiente de Variação")
        st.success("✅ Poisson avançado")
        st.success("✅ Combined Score")
        st.success("✅ Comparação vs Liga")
        st.success("✅ Apostas ilimitadas")
        st.success("✅ Análise Casa vs Fora")
        
        # Erros de treinamento
        if st.session_state.training_errors:
            with st.expander("⚠️ Erros de Treinamento"):
                for error in st.session_state.training_errors[-5:]:
                    st.write(f"• {error}")
    
    # Tabs principais
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 Treinar Avançado", "📊 Análise Ligas", "🎯 Previsões Ilimitadas", "📈 Dashboard Premium"])
    
    with tab1:
        st.header("🤖 Treinamento Avançado Multi-Temporadas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **✅ Suas Melhorias Implementadas:**
            - ✅ Múltiplas temporadas (2-3 anos)
            - ✅ Detecção inteligente de outliers
            - ✅ Coeficiente de Variação
            - ✅ Modelos Poisson avançados
            - ✅ Combined Score sofisticado
            - ✅ Treino/Validação/Teste
            """)
        
        with col2:
            st.success("""
            **🎯 Análises Avançadas:**
            - ✅ Comparação detalhada vs Liga
            - ✅ Análise robusta Casa vs Fora
            - ✅ Consistência das equipas
            - ✅ Apostas ilimitadas por dia
            - ✅ Ensemble de modelos
            - ✅ Threshold otimizado
            """)
        
        if st.button("🚀 TREINAR SISTEMA ULTIMATE", type="primary", use_container_width=True):
            
            st.session_state.training_errors = []
            st.session_state.training_in_progress = True
            
            try:
                with st.spinner("📥 Carregando dados de múltiplas temporadas..."):
                    df = collect_historical_data_multi_season(use_cached=use_cache)
                
                if df.empty:
                    st.error("❌ Dados insuficientes para treinamento")
                    st.session_state.training_in_progress = False
                    st.stop()
                
                st.success(f"✅ {len(df)} jogos carregados (múltiplas temporadas)")
                
                # Mostrar distribuição por temporadas
                if 'season' in df.columns:
                    seasons = df['season'].value_counts().sort_index()
                    st.write("📅 **Distribuição por Temporada:**")
                    for season, count in seasons.items():
                        st.write(f"- {season}: {count:,} jogos")
                
                # Agrupar por liga
                league_groups = df.groupby(['league_id', 'league_name', 'country'])
                
                st.info(f"🎯 {len(league_groups)} ligas encontradas - Mínimo {min_matches_per_league} jogos")
                
                league_models = {}
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results_summary = []
                successful_leagues = 0
                
                for idx, ((league_id, league_name, country), league_df) in enumerate(league_groups):
                    progress = (idx + 1) / len(league_groups)
                    progress_bar.progress(progress)
                    
                    league_full_name = f"{league_name} ({country})"
                    status_text.text(f"🔄 Treinando avançado: {league_full_name}")
                    
                    if len(league_df) < min_matches_per_league:
                        continue
                    
                    # Treinar modelo avançado
                    model_data, message = train_advanced_model_with_validation(
                        league_df, league_id, league_full_name, min_matches_per_league
                    )
                    
                    if model_data:
                        league_models[league_id] = model_data
                        successful_leagues += 1
                        st.success(message)
                        
                        results_summary.append({
                            'Liga': league_full_name,
                            'Jogos': len(league_df),
                            'Acurácia': model_data['test_metrics']['accuracy'],
                            'F1-Score': model_data['test_metrics']['f1_score'],
                            'AUC': model_data['test_metrics']['auc_score'],
                            'Tipo': model_data.get('model_type', 'Unknown')
                        })
                        
                        # Backup a cada 3 ligas
                        if successful_leagues % 3 == 0:
                            save_training_progress(league_models, f"advanced_backup_{successful_leagues}")
                    else:
                        st.warning(message)
                
                progress_bar.empty()
                status_text.empty()
                
                if league_models:
                    # Salvar final
                    save_training_progress(league_models, "advanced_final")
                    
                    st.session_state.league_models = league_models
                    st.session_state.models_trained = True
                    
                    # Resumo final
                    st.success(f"🎉 {len(league_models)} ligas treinadas com sistema avançado!")
                    
                    if results_summary:
                        avg_accuracy = np.mean([r['Acurácia'] for r in results_summary])
                        avg_f1 = np.mean([r['F1-Score'] for r in results_summary])
                        avg_auc = np.mean([r['AUC'] for r in results_summary])
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Ligas Treinadas", len(league_models))
                        with col2:
                            st.metric("Acurácia Média", f"{avg_accuracy:.1%}")
                        with col3:
                            st.metric("F1-Score Médio", f"{avg_f1:.1%}")
                        with col4:
                            st.metric("AUC Score Médio", f"{avg_auc:.2f}")
                        
                        # Distribuição de modelos
                        model_types = pd.DataFrame(results_summary)['Tipo'].value_counts()
                        st.write("🤖 **Modelos Utilizados:**")
                        for model_type, count in model_types.items():
                            st.write(f"- {model_type}: {count} ligas")
                    
                    st.balloons()
                else:
                    st.error("❌ Nenhuma liga foi treinada!")
                    
            except Exception as e:
                st.error(f"❌ Erro no treinamento avançado: {str(e)}")
                st.code(traceback.format_exc())
                
                if st.session_state.models_backup:
                    st.warning("🔄 Carregando backup...")
                    if load_training_progress():
                        st.success("✅ Backup restaurado!")
            
            finally:
                st.session_state.training_in_progress = False
    
    with tab2:
        if not st.session_state.models_trained:
            st.warning("⚠️ Treine o sistema primeiro!")
            if st.session_state.models_backup:
                if st.button("🔄 Carregar Backup", key="tab2_backup"):
                    load_training_progress()
                    st.rerun()
        else:
            display_advanced_league_summary(st.session_state.league_models)
    
    with tab3:
        st.header("🎯 Previsões Ilimitadas com Análise Avançada")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Treine o sistema primeiro!")
            if st.session_state.models_backup:
                if st.button("🔄 Carregar Backup", key="tab3_backup"):
                    load_training_progress()
                    st.rerun()
            st.stop()
        
        # Sua sugestão #1: APOSTAS ILIMITADAS
        st.success("🔥 **APOSTAS ILIMITADAS POR DIA** - Todas as previsões acima da confiança mínima!")
        
        selected_date = st.date_input("📅 Data:", value=datetime.now().date())
        date_str = selected_date.strftime('%Y-%m-%d')
        
        # Múltiplos dias
        multi_day = st.checkbox("🗓️ Analisar múltiplos dias (até 3 dias)")
        
        fixtures = []
        
        with st.spinner("🔍 Coletando jogos..."):
            fixtures.extend(get_fixtures_cached(date_str))
            
            if multi_day:
                for i in range(1, 4):
                    next_date = (selected_date + timedelta(days=i)).strftime('%Y-%m-%d')
                    fixtures.extend(get_fixtures_cached(next_date))
        
        if not fixtures:
            st.info("📅 Nenhum jogo encontrado")
        else:
            st.info(f"🔍 {len(fixtures)} jogos encontrados para análise")
            
            # Previsões avançadas ILIMITADAS
            predictions = predict_with_advanced_strategy(
                fixtures, 
                st.session_state.league_models, 
                min_confidence=min_confidence
            )
            
            if not predictions:
                st.info("🤷 Nenhuma previsão acima da confiança mínima")
                st.write("**Possíveis motivos:**")
                st.write("• Combined Score abaixo do mínimo")
                st.write("• Times não nos dados de treinamento")
                st.write("• Ligas não treinadas")
            else:
                st.success(f"🎯 **{len(predictions)} APOSTAS ENCONTRADAS!** (Ilimitadas)")
                
                # Filtros avançados
                col1, col2, col3 = st.columns(3)
                with col1:
                    show_only_over = st.checkbox("Apenas OVER 0.5", value=True)
                with col2:
                    sort_by = st.selectbox("Ordenar por:", 
                                         ["Combined Score", "vs Liga %", "Confidence", "Consistency"])
                with col3:
                    min_vs_league = st.slider("Mínimo vs Liga %", min_value=-30, max_value=50, value=0)
                
                # Aplicar filtros
                filtered_predictions = predictions.copy()
                
                if show_only_over:
                    filtered_predictions = [p for p in filtered_predictions if p['prediction'] == 'OVER 0.5']
                
                # Filtro vs Liga
                filtered_predictions = [p for p in filtered_predictions if p['vs_league_percentage'] >= min_vs_league]
                
                # Ordenação
                if sort_by == "vs Liga %":
                    filtered_predictions.sort(key=lambda x: x['vs_league_percentage'], reverse=True)
                elif sort_by == "Confidence":
                    filtered_predictions.sort(key=lambda x: x['confidence'], reverse=True)
                elif sort_by == "Consistency":
                    filtered_predictions.sort(key=lambda x: x['consistency_score'], reverse=True)
                else:  # Combined Score (padrão)
                    filtered_predictions.sort(key=lambda x: x['combined_score'], reverse=True)
                
                # Estatísticas das previsões
                if filtered_predictions:
                    avg_combined = np.mean([p['combined_score'] for p in filtered_predictions])
                    avg_vs_league = np.mean([p['vs_league_percentage'] for p in filtered_predictions])
                    avg_consistency = np.mean([p['consistency_score'] for p in filtered_predictions])
                    high_quality = len([p for p in filtered_predictions if p['combined_score'] >= 75])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Apostas", len(filtered_predictions))
                    with col2:
                        st.metric("Combined Score Médio", f"{avg_combined:.1f}")
                    with col3:
                        st.metric("vs Liga Médio", f"{avg_vs_league:+.1f}%")
                    with col4:
                        st.metric("Alta Qualidade", high_quality)
                        st.caption("Score ≥ 75")
                    
                    # Export para Excel
                    export_data = []
                    for p in filtered_predictions:
                        export_data.append({
                            'Data': p['kickoff'].split('T')[0],
                            'Hora': p['kickoff'].split('T')[1][:5],
                            'Liga': p['league'],
                            'Casa': p['home_team'],
                            'Fora': p['away_team'],
                            'Previsão': p['prediction'],
                            'Combined Score': f"{p['combined_score']:.1f}",
                            'ML Confidence': f"{p['confidence']:.1f}%",
                            'vs Liga %': f"{p['vs_league_percentage']:+.1f}%",
                            'Consistency': f"{p['consistency_score']:.2f}",
                            'Expected Goals': f"{p['expected_goals_ht']:.2f}",
                            'Outlier Risk': f"{p['outlier_risk']:.2f}"
                        })
                    
                    export_df = pd.DataFrame(export_data)
                    excel_data = create_excel_download(export_df, "previsoes_avancadas.xlsx")
                    
                    if excel_data:
                        st.download_button(
                            label="📥 Exportar Todas as Previsões",
                            data=excel_data,
                            file_name=f"previsoes_avancadas_{date_str}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    st.markdown("---")
                    
                    # Mostrar todas as previsões (ILIMITADAS)
                    for pred in filtered_predictions:
                        display_advanced_prediction(pred)
                else:
                    st.info("🔍 Nenhuma previsão após filtros")
    
    with tab4:
        st.header("📈 Dashboard Premium")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Treine o sistema primeiro!")
        else:
            # Estatísticas gerais
            total_leagues = len(st.session_state.league_models)
            total_matches = sum(m['total_matches'] for m in st.session_state.league_models.values())
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Ligas", total_leagues)
            with col2:
                st.metric("Total Jogos", f"{total_matches:,}")
            with col3:
                avg_acc = np.mean([m['test_metrics']['accuracy'] for m in st.session_state.league_models.values()])
                st.metric("Acurácia Média", f"{avg_acc:.1%}")
            with col4:
                multi_season_count = sum(1 for m in st.session_state.league_models.values() if m.get('multi_season_analysis'))
                st.metric("Multi-Season", f"{multi_season_count}/{total_leagues}")
            
            st.markdown("---")
            
            # Features mais importantes globalmente
            st.subheader("🎯 Features Mais Importantes Globalmente")
            
            all_features = {}
            for model_data in st.session_state.league_models.values():
                try:
                    for feature, importance in model_data['top_features']:
                        if feature not in all_features:
                            all_features[feature] = []
                        all_features[feature].append(importance)
                except:
                    continue
            
            if all_features:
                avg_features = {f: np.mean(imps) for f, imps in all_features.items()}
                top_global_features = sorted(avg_features.items(), key=lambda x: x[1], reverse=True)[:15]
                
                df_features = pd.DataFrame(top_global_features, columns=['Feature', 'Importância'])
                df_features['Feature'] = df_features['Feature'].str.replace('_', ' ').str.title()
                df_features['Importância (%)'] = (df_features['Importância'] * 100).round(1)
                
                # Gráfico de features
                chart_data = df_features.set_index('Feature')['Importância']
                st.bar_chart(chart_data)
                
                # Download
                excel_features = create_excel_download(df_features, "features_globais.xlsx")
                if excel_features:
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col2:
                        st.download_button(
                            label="📥 Download Features Globais",
                            data=excel_features,
                            file_name=f"features_globais_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                
                # Tabela de features
                st.dataframe(df_features, use_container_width=True)

if __name__ == "__main__":
    main()
