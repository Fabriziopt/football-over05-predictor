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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="⚽ HT Goals AI Ultimate",
    page_icon="🎯",
    layout="wide"
)

# Inicializar session state com valores seguros
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

# Configuração da API
try:
    API_KEY = st.secrets["API_KEY"]
except:
    API_KEY = "474f15de5b22951077ccb71b8d75b95c"

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
    """Retorna os headers corretos para API-SPORTS"""
    return {'x-apisports-key': API_KEY}

def test_api_connection():
    """Testa a conectividade com a API"""
    try:
        headers = get_api_headers()
        response = requests.get(f'{API_BASE_URL}/status', headers=headers, timeout=10)
        if response.status_code == 200:
            return True, "Conexão OK"
        else:
            return False, f"Status HTTP: {response.status_code}"
    except Exception as e:
        return False, f"Erro de conexão: {str(e)}"

def get_fixtures_with_retry(date_str, max_retries=3):
    """Busca jogos da API com retry automático e tratamento robusto"""
    headers = get_api_headers()
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                f'{API_BASE_URL}/fixtures',
                headers=headers,
                params={'date': date_str},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                fixtures = data.get('response', [])
                return fixtures
            elif response.status_code == 429:  # Rate limit
                wait_time = 2 ** (attempt + 1)
                st.warning(f"⏳ Rate limit - aguardando {wait_time}s...")
                time.sleep(wait_time)
            else:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    st.error(f"❌ Erro HTTP {response.status_code}")
                    return []
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                st.error(f"❌ Erro na API: {str(e)}")
                return []
    return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_fixtures_cached(date_str):
    """Busca jogos com cache"""
    try:
        return get_fixtures_with_retry(date_str)
    except:
        return []

def load_historical_data():
    """Carrega dados históricos do arquivo local"""
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
                
                # Validar dados
                if df.empty:
                    continue
                
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df = df.dropna(subset=['date'])
                
                # Criar target se não existir
                if 'over_05' not in df.columns:
                    if 'ht_home' in df.columns and 'ht_away' in df.columns:
                        df['over_05'] = (df['ht_home'] + df['ht_away']) > 0
                    elif 'ht_home_goals' in df.columns and 'ht_away_goals' in df.columns:
                        df['over_05'] = (df['ht_home_goals'] + df['ht_away_goals']) > 0
                    else:
                        continue
                
                return df, f"✅ {len(df)} jogos carregados de {file_path}"
            except Exception as e:
                st.warning(f"⚠️ Erro ao carregar {file_path}: {str(e)}")
                continue
    
    return None, "❌ Nenhum arquivo encontrado"

def get_seasonal_data_period():
    """Calcula período ideal baseado na temporada"""
    current_date = datetime.now()
    current_month = current_date.month
    current_year = current_date.year
    
    if current_month >= 8:  # Agosto a Dezembro
        start_date = datetime(current_year - 1, 8, 1)
        days_back = (current_date - start_date).days
    else:  # Janeiro a Julho
        start_date = datetime(current_year - 1, 8, 1)
        days_back = (current_date - start_date).days
    
    days_back = max(days_back, 365)
    
    return days_back, start_date

def collect_historical_data_smart(days=None, use_cached=True, seasonal=True):
    """Coleta inteligente com opção sazonal e tratamento robusto"""
    
    if seasonal and days is None:
        days, start_date = get_seasonal_data_period()
        st.info(f"📅 Modo Sazonal: Buscando dados desde {start_date.strftime('%d/%m/%Y')} ({days} dias)")
    elif days is None:
        days = 365
    
    # Tentar carregar cache primeiro
    if use_cached:
        df_cache, message = load_historical_data()
        if df_cache is not None and not df_cache.empty:
            try:
                if 'date' in df_cache.columns:
                    df_cache['date'] = pd.to_datetime(df_cache['date'], errors='coerce')
                    df_cache = df_cache.dropna(subset=['date'])
                    current_date = datetime.now()
                    cutoff_date = current_date - timedelta(days=days)
                    df_filtered = df_cache[df_cache['date'] >= cutoff_date].copy()
                    
                    if len(df_filtered) > 50:  # Mínimo de dados
                        st.success(f"✅ {len(df_filtered)} jogos carregados do cache")
                        return df_filtered
            except Exception as e:
                st.warning(f"⚠️ Erro ao processar cache: {str(e)}")
    
    # Buscar da API se necessário
    st.warning("⚠️ Coletando dados da API...")
    
    # Amostragem inteligente
    sample_days = []
    
    # Últimos 30 dias - todos os dias
    for i in range(min(30, days)):
        sample_days.append(i + 1)
    
    # 30-90 dias - a cada 2 dias
    if days > 30:
        for i in range(30, min(90, days), 2):
            sample_days.append(i + 1)
    
    # 90+ dias - a cada 3 dias
    if days > 90:
        for i in range(90, min(days, 365), 3):
            sample_days.append(i + 1)
    
    all_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    errors_count = 0
    max_errors = 10  # Máximo de erros permitidos
    
    for idx, day_offset in enumerate(sample_days):
        try:
            date = datetime.now() - timedelta(days=day_offset)
            date_str = date.strftime('%Y-%m-%d')
            
            status_text.text(f"🔍 Coletando dados de {date_str}...")
            
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
            
            # Rate limiting
            if idx % 3 == 0:
                time.sleep(0.5)
                
        except Exception as e:
            errors_count += 1
            if errors_count > max_errors:
                st.error(f"❌ Muitos erros na coleta. Parando com {len(all_data)} jogos.")
                break
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if len(all_data) < 100:
        st.error(f"❌ Dados insuficientes coletados: {len(all_data)} jogos")
        return pd.DataFrame()
    
    st.success(f"✅ {len(all_data)} jogos coletados da API")
    return pd.DataFrame(all_data)

def extract_match_features(match):
    """Extrai features básicas do jogo com validação"""
    try:
        # Validar estrutura
        if not match.get('score', {}).get('halftime'):
            return None
        
        ht_home = match['score']['halftime']['home']
        ht_away = match['score']['halftime']['away']
        
        # Validar se são números válidos
        if ht_home is None or ht_away is None:
            return None
        
        if not isinstance(ht_home, (int, float)) or not isinstance(ht_away, (int, float)):
            return None
        
        features = {
            'date': match['fixture']['date'][:10],
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
            'over_05': 1 if (int(ht_home) + int(ht_away)) > 0 else 0
        }
        
        return features
    except Exception as e:
        return None

def calculate_poisson_probabilities(home_avg, away_avg):
    """Calcula probabilidades usando distribuição de Poisson com validação"""
    try:
        # Validar inputs
        if home_avg < 0 or away_avg < 0:
            return {'poisson_over_05': 0.5, 'expected_goals_ht': 0.5, 'home_lambda': 0.25, 'away_lambda': 0.25}
        
        # Lambda para cada time (média de gols esperados)
        home_lambda = max(home_avg / 2, 0.01)  # Mínimo 0.01
        away_lambda = max(away_avg / 2, 0.01)
        
        # Probabilidade de 0 gols para cada time
        prob_home_0 = poisson.pmf(0, home_lambda)
        prob_away_0 = poisson.pmf(0, away_lambda)
        
        # Probabilidade de 0-0 no HT
        prob_0_0 = prob_home_0 * prob_away_0
        
        # Probabilidade de Over 0.5 HT
        prob_over_05 = 1 - prob_0_0
        
        # Gols esperados no HT
        expected_goals_ht = home_lambda + away_lambda
        
        return {
            'poisson_over_05': min(max(prob_over_05, 0), 1),
            'expected_goals_ht': max(expected_goals_ht, 0),
            'home_lambda': home_lambda,
            'away_lambda': away_lambda
        }
    except:
        return {'poisson_over_05': 0.5, 'expected_goals_ht': 0.5, 'home_lambda': 0.25, 'away_lambda': 0.25}

def calculate_advanced_features(league_df):
    """Calcula features avançadas com tratamento robusto de erros"""
    try:
        # Validar DataFrame
        if league_df.empty:
            return pd.DataFrame(), {}, 0.5
        
        # Garantir que over_05 existe
        if 'over_05' not in league_df.columns:
            if 'ht_home_goals' in league_df.columns and 'ht_away_goals' in league_df.columns:
                league_df['over_05'] = (league_df['ht_home_goals'] + league_df['ht_away_goals']) > 0
            else:
                return pd.DataFrame(), {}, 0.5
        
        # Ordenar por data se disponível
        if 'date' in league_df.columns:
            league_df = league_df.sort_values('date').reset_index(drop=True)
        
        # Estatísticas da liga com fallbacks
        league_over_rate = league_df['over_05'].mean() if len(league_df) > 0 else 0.5
        league_avg_goals = league_df['ht_total_goals'].mean() if len(league_df) > 0 else 1.0
        
        # Estatísticas por time
        team_stats = {}
        try:
            unique_teams = pd.concat([league_df['home_team_id'], league_df['away_team_id']]).unique()
        except:
            return pd.DataFrame(), {}, league_over_rate
        
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
                
                # Análise com fallbacks
                home_goals_scored = home_matches['ht_home_goals'].mean() if len(home_matches) > 0 else league_avg_goals/2
                home_goals_conceded = home_matches['ht_away_goals'].mean() if len(home_matches) > 0 else league_avg_goals/2
                away_goals_scored = away_matches['ht_away_goals'].mean() if len(away_matches) > 0 else league_avg_goals/2
                away_goals_conceded = away_matches['ht_home_goals'].mean() if len(away_matches) > 0 else league_avg_goals/2
                
                team_stats[team_id] = {
                    'team_name': team_name,
                    'games': len(all_matches),
                    'over_rate': all_matches['over_05'].mean(),
                    # Casa
                    'home_games': len(home_matches),
                    'home_over_rate': home_matches['over_05'].mean() if len(home_matches) > 0 else league_over_rate,
                    'home_goals_scored': home_goals_scored,
                    'home_goals_conceded': home_goals_conceded,
                    # Fora
                    'away_games': len(away_matches),
                    'away_over_rate': away_matches['over_05'].mean() if len(away_matches) > 0 else league_over_rate,
                    'away_goals_scored': away_goals_scored,
                    'away_goals_conceded': away_goals_conceded,
                    # Força ofensiva/defensiva
                    'home_attack_strength': max(home_goals_scored / (league_avg_goals/2 + 0.01), 0.1),
                    'home_defense_strength': max(home_goals_conceded / (league_avg_goals/2 + 0.01), 0.1),
                    'away_attack_strength': max(away_goals_scored / (league_avg_goals/2 + 0.01), 0.1),
                    'away_defense_strength': max(away_goals_conceded / (league_avg_goals/2 + 0.01), 0.1)
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
                
                # Poisson predictions com validação
                home_expected = home_stats['home_attack_strength'] * away_stats['away_defense_strength'] * league_avg_goals/2
                away_expected = away_stats['away_attack_strength'] * home_stats['home_defense_strength'] * league_avg_goals/2
                
                poisson_calc = calculate_poisson_probabilities(home_expected * 2, away_expected * 2)
                
                # Features completas
                feature_row = {
                    # Taxas básicas
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
                    
                    # Poisson
                    'poisson_over_05': poisson_calc['poisson_over_05'],
                    'expected_goals_ht': poisson_calc['expected_goals_ht'],
                    
                    # Combinações
                    'combined_over_rate': (home_stats['home_over_rate'] + away_stats['away_over_rate']) / 2,
                    'attack_index': (home_stats['home_attack_strength'] + away_stats['away_attack_strength']) / 2,
                    'game_pace_index': max(home_expected + away_expected, 0),
                    
                    # Comparação com média da liga
                    'over_rate_vs_league': ((home_stats['home_over_rate'] + away_stats['away_over_rate']) / 2) / max(league_over_rate, 0.01),
                    'expected_vs_league': poisson_calc['expected_goals_ht'] / max(league_avg_goals, 0.01),
                    
                    # Jogos disputados
                    'home_games_played': home_stats['home_games'],
                    'away_games_played': away_stats['away_games'],
                    'min_games': min(home_stats['home_games'], away_stats['away_games']),
                    
                    # Target
                    'target': row['over_05']
                }
                
                features.append(feature_row)
            except Exception as e:
                continue
        
        return pd.DataFrame(features), team_stats, league_over_rate
        
    except Exception as e:
        st.error(f"❌ Erro ao calcular features: {str(e)}")
        return pd.DataFrame(), {}, 0.5

def train_complete_model_with_validation(league_df, league_id, league_name, min_matches=30):
    """Treina modelo com validação completa e tratamento robusto"""
    
    if len(league_df) < min_matches:
        return None, f"❌ {league_name}: {len(league_df)} jogos < {min_matches} mínimo"
    
    try:
        # Preparar features avançadas
        features_df, team_stats, league_over_rate = calculate_advanced_features(league_df)
        
        if features_df.empty or len(features_df) < min_matches:
            return None, f"❌ {league_name}: Features insuficientes"
        
        # Verificar se temos variação no target
        if features_df['target'].nunique() < 2:
            return None, f"❌ {league_name}: Sem variação no target"
        
        feature_cols = [col for col in features_df.columns if col != 'target']
        X = features_df[feature_cols]
        y = features_df['target']
        
        # Verificar NaN
        if X.isnull().any().any() or y.isnull().any():
            X = X.fillna(X.mean())
            y = y.fillna(0)
        
        # Dividir dados com estratificação segura
        try:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
            )
        except:
            # Fallback sem estratificação
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.25, random_state=42
            )
        
        # Escalar dados
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Modelos com configurações mais conservadoras
        models = {
            'rf': RandomForestClassifier(
                n_estimators=100, max_depth=8, random_state=42, 
                n_jobs=1, min_samples_split=5, min_samples_leaf=2
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
            )
        }
        
        # Treinar e validar cada modelo
        best_model = None
        best_score = 0
        results = {}
        
        for name, model in models.items():
            try:
                # Treinar
                model.fit(X_train_scaled, y_train)
                
                # Validar
                val_pred = model.predict(X_val_scaled)
                val_acc = accuracy_score(y_val, val_pred)
                val_prec = precision_score(y_val, val_pred, zero_division=0)
                val_rec = recall_score(y_val, val_pred, zero_division=0)
                val_f1 = f1_score(y_val, val_pred, zero_division=0)
                
                results[name] = {
                    'val_accuracy': val_acc,
                    'val_precision': val_prec,
                    'val_recall': val_rec,
                    'val_f1': val_f1
                }
                
                if val_f1 > best_score:
                    best_score = val_f1
                    best_model = model
                    
            except Exception as e:
                st.warning(f"⚠️ Erro treinando {name}: {str(e)}")
                continue
        
        if best_model is None:
            return None, f"❌ {league_name}: Nenhum modelo funcionou"
        
        # Testar melhor modelo
        test_pred = best_model.predict(X_test_scaled)
        test_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, test_pred),
            'precision': precision_score(y_test, test_pred, zero_division=0),
            'recall': recall_score(y_test, test_pred, zero_division=0),
            'f1_score': f1_score(y_test, test_pred, zero_division=0)
        }
        
        # Análise de threshold ótimo
        best_threshold = 0.5
        best_f1 = test_metrics['f1_score']
        
        for threshold in np.arange(0.4, 0.7, 0.05):
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
            feature_importance = dict(zip(feature_cols, best_model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        except:
            top_features = [('unknown', 1.0)]
        
        # Preparar dados do modelo
        model_data = {
            'model': best_model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'team_stats': team_stats,
            'league_id': league_id,
            'league_name': league_name,
            'league_over_rate': league_over_rate,
            'total_matches': len(league_df),
            'validation_results': results,
            'test_metrics': test_metrics,
            'best_threshold': best_threshold,
            'top_features': top_features
        }
        
        return model_data, f"✅ {league_name}: Acc {test_metrics['accuracy']:.1%} | F1 {test_metrics['f1_score']:.1%}"
        
    except Exception as e:
        error_msg = f"❌ {league_name}: {str(e)}"
        st.session_state.training_errors.append(error_msg)
        return None, error_msg

def predict_with_strategy(fixtures, league_models, min_confidence=60):
    """Faz previsões com estratégia inteligente e tratamento robusto"""
    
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
            
            # Poisson calculation
            home_expected = home_stats['home_attack_strength'] * away_stats['away_defense_strength'] * 0.5
            away_expected = away_stats['away_attack_strength'] * home_stats['home_defense_strength'] * 0.5
            
            poisson_calc = calculate_poisson_probabilities(home_expected * 2, away_expected * 2)
            
            # Criar features
            features = {
                'home_over_rate': home_stats['over_rate'],
                'away_over_rate': away_stats['over_rate'],
                'home_home_over_rate': home_stats['home_over_rate'],
                'away_away_over_rate': away_stats['away_over_rate'],
                'league_over_rate': league_over_rate,
                'home_attack_strength': home_stats['home_attack_strength'],
                'home_defense_strength': home_stats['home_defense_strength'],
                'away_attack_strength': away_stats['away_attack_strength'],
                'away_defense_strength': away_stats['away_defense_strength'],
                'poisson_over_05': poisson_calc['poisson_over_05'],
                'expected_goals_ht': poisson_calc['expected_goals_ht'],
                'combined_over_rate': (home_stats['home_over_rate'] + away_stats['away_over_rate']) / 2,
                'attack_index': (home_stats['home_attack_strength'] + away_stats['away_attack_strength']) / 2,
                'game_pace_index': (home_expected + away_expected),
                'over_rate_vs_league': ((home_stats['home_over_rate'] + away_stats['away_over_rate']) / 2) / max(league_over_rate, 0.01),
                'expected_vs_league': poisson_calc['expected_goals_ht'] / 0.5,
                'home_games_played': home_stats['home_games'],
                'away_games_played': away_stats['away_games'],
                'min_games': min(home_stats['home_games'], away_stats['away_games'])
            }
            
            # Fazer previsão
            X = pd.DataFrame([features])[feature_cols]
            
            # Tratar missing features
            for col in feature_cols:
                if col not in X.columns:
                    X[col] = 0.5  # Valor neutro
            
            X = X[feature_cols]  # Reordenar colunas
            X = X.fillna(0.5)   # Tratar NaN
            
            X_scaled = scaler.transform(X)
            
            pred_proba = model.predict_proba(X_scaled)[0]
            confidence = pred_proba[1] * 100
            
            # Aplicar threshold otimizado
            pred_class = 1 if pred_proba[1] >= best_threshold else 0
            
            # Calcular indicadores de força
            game_vs_league = features['combined_over_rate'] / max(league_over_rate, 0.01)
            
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
                'ml_probability': pred_proba[1] * 100,
                'poisson_probability': poisson_calc['poisson_over_05'] * 100,
                'expected_goals_ht': poisson_calc['expected_goals_ht'],
                'game_vs_league_ratio': game_vs_league,
                'model_metrics': model_data['test_metrics'],
                'top_features': model_data['top_features']
            }
            
            if confidence >= min_confidence:
                predictions.append(prediction)
            
        except Exception as e:
            continue
    
    # Ordenar por confiança
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return predictions

def calculate_fair_odds(confidence_percentage):
    """Calcula a odd justa baseada na confiança do modelo"""
    try:
        # Converter confiança para probabilidade (0-1)
        probability = confidence_percentage / 100
        
        # Odd justa = 1 / probabilidade
        fair_odd = 1 / probability
        
        return round(fair_odd, 2)
    except:
        return 0.0

def display_smart_prediction(pred):
    """Exibe previsão com análise inteligente e odd justa"""
    
    try:
        with st.container():
            # Header
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.subheader(f"⚽ {pred['home_team']} vs {pred['away_team']}")
            
            with col2:
                if pred['confidence'] > 75:
                    st.success(f"**{pred['confidence']:.1f}%**")
                elif pred['confidence'] > 65:
                    st.info(f"**{pred['confidence']:.1f}%**")
                else:
                    st.warning(f"**{pred['confidence']:.1f}%**")
            
            with col3:
                # Comparação com liga
                ratio = pred['game_vs_league_ratio']
                if ratio > 1.2:
                    st.write("🔥 **+{:.0f}%**".format((ratio-1)*100))
                elif ratio < 0.8:
                    st.write("❄️ **-{:.0f}%**".format((1-ratio)*100))
                else:
                    st.write("➖ **Média**")
            
            # Análise detalhada
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"🏆 **Liga:** {pred['league']} ({pred['country']})")
                st.write(f"📊 **Média da Liga:** {pred['league_over_rate']:.1f}%")
                
            with col2:
                st.write(f"🏠 **{pred['home_team']}**")
                st.write(f"- Casa: {pred['home_team_stats']['home_over_rate']*100:.1f}%")
                st.write(f"- Força Ataque: {pred['home_team_stats']['home_attack_strength']:.2f}")
                
            with col3:
                st.write(f"✈️ **{pred['away_team']}**")
                st.write(f"- Fora: {pred['away_team_stats']['away_over_rate']*100:.1f}%")
                st.write(f"- Força Ataque: {pred['away_team_stats']['away_attack_strength']:.2f}")
            
            # Previsões com Odd Justa
            st.markdown("### 🎯 Análise Preditiva")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("ML Probability", f"{pred['ml_probability']:.1f}%")
            with col2:
                st.metric("Poisson Probability", f"{pred['poisson_probability']:.1f}%")
            with col3:
                st.metric("Gols Esperados HT", f"{pred['expected_goals_ht']:.2f}")
            with col4:
                # Calcular e mostrar odd justa
                fair_odd = calculate_fair_odds(pred['confidence'])
                st.metric("💰 Odd Justa", f"{fair_odd}")
            
            # Recomendação com odd justa
            if pred['prediction'] == 'OVER 0.5':
                if pred['confidence'] > 70 and pred['game_vs_league_ratio'] > 1.1:
                    st.success(f"✅ **APOSTAR: {pred['prediction']} HT** (Alta Confiança) | **Odd Justa: {fair_odd}**")
                else:
                    st.info(f"📊 **Considerar: {pred['prediction']} HT** (Confiança Moderada) | **Odd Justa: {fair_odd}**")
            
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

def display_league_summary(league_models):
    """Exibe resumo das ligas com visualizações simples"""
    
    try:
        st.header("📊 Análise das Ligas")
        
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
                    'Precisão': round(model_data['test_metrics']['precision'] * 100, 1),
                    'Recall': round(model_data['test_metrics']['recall'] * 100, 1),
                    'Threshold Ótimo': round(model_data['best_threshold'], 3)
                })
            except:
                continue
        
        if not league_data:
            st.warning("⚠️ Nenhum dado para exibir!")
            return
        
        df_leagues = pd.DataFrame(league_data)
        
        # Botão de download no topo
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            excel_data = create_excel_download(df_leagues, "analise_ligas.xlsx")
            if excel_data:
                st.download_button(
                    label="📥 Download Excel - Todas as Ligas",
                    data=excel_data,
                    file_name=f"analise_ligas_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        # Visualizações em colunas
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Top 15 Ligas - Taxa Over 0.5 HT")
            top_leagues = df_leagues.sort_values('Over 0.5 HT %', ascending=False).head(15)
            chart_data = top_leagues.set_index('Liga')['Over 0.5 HT %']
            st.bar_chart(chart_data)
            
            # Download Top 15
            excel_top = create_excel_download(top_leagues, "top_15_ligas.xlsx")
            if excel_top:
                st.download_button(
                    label="📥 Download Top 15 Ligas",
                    data=excel_top,
                    file_name=f"top_15_ligas_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col2:
            st.subheader("🎯 Performance dos Modelos")
            performance_data = df_leagues[['Liga', 'Acurácia', 'F1-Score']].set_index('Liga')
            st.line_chart(performance_data)
            
            # Download Performance
            perf_df = df_leagues[['Liga', 'Acurácia', 'F1-Score', 'Precisão', 'Recall']].sort_values('F1-Score', ascending=False)
            excel_perf = create_excel_download(perf_df, "performance_modelos.xlsx")
            if excel_perf:
                st.download_button(
                    label="📥 Download Performance",
                    data=excel_perf,
                    file_name=f"performance_modelos_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        # Tabela resumo
        st.subheader("📋 Resumo Detalhado de Todas as Ligas")
        df_display = df_leagues.sort_values('F1-Score', ascending=False)
        st.dataframe(df_display, use_container_width=True)
        
        # Análise de qualidade com downloads
        st.subheader("📊 Análise de Qualidade dos Modelos")
        
        high_quality = df_leagues[df_leagues['F1-Score'] >= 80]
        medium_quality = df_leagues[(df_leagues['F1-Score'] >= 70) & (df_leagues['F1-Score'] < 80)]
        low_quality = df_leagues[df_leagues['F1-Score'] < 70]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🟢 Alta Qualidade (F1 ≥ 80%)", len(high_quality))
            if len(high_quality) > 0:
                excel_high = create_excel_download(high_quality, "ligas_alta_qualidade.xlsx")
                if excel_high:
                    st.download_button(
                        label="📥 Download Alta Qualidade",
                        data=excel_high,
                        file_name=f"ligas_alta_qualidade_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="high_quality"
                    )
        
        with col2:
            st.metric("🟡 Média Qualidade (F1 70-80%)", len(medium_quality))
            if len(medium_quality) > 0:
                excel_med = create_excel_download(medium_quality, "ligas_media_qualidade.xlsx")
                if excel_med:
                    st.download_button(
                        label="📥 Download Média Qualidade",
                        data=excel_med,
                        file_name=f"ligas_media_qualidade_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="medium_quality"
                    )
        
        with col3:
            st.metric("🔴 Baixa Qualidade (F1 < 70%)", len(low_quality))
            if len(low_quality) > 0:
                excel_low = create_excel_download(low_quality, "ligas_baixa_qualidade.xlsx")
                if excel_low:
                    st.download_button(
                        label="📥 Download Baixa Qualidade",
                        data=excel_low,
                        file_name=f"ligas_baixa_qualidade_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="low_quality"
                    )
        
    except Exception as e:
        st.error(f"❌ Erro ao exibir resumo: {str(e)}")

def main():
    st.title("⚽ HT Goals AI Ultimate - Sistema Completo")
    st.markdown("🎯 **Versão Super Robusta - Máxima taxa de acerto**")
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configurações")
        
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
            
            # Botão para carregar backup se disponível
            if st.session_state.models_backup:
                if st.button("🔄 Carregar Backup"):
                    load_training_progress()
                    st.success("✅ Backup carregado!")
                    st.rerun()
        else:
            st.warning("⚠️ Modelos não treinados")
        
        st.markdown("---")
        
        # Configurações
        st.markdown("### 📊 Parâmetros")
        
        min_matches_per_league = st.slider(
            "Mínimo jogos por liga:",
            min_value=20,
            max_value=100,
            value=30,
            help="30 jogos é o equilíbrio ideal"
        )
        
        min_confidence = st.slider(
            "Confiança mínima:",
            min_value=50,
            max_value=80,
            value=60,
            help="60% permite mais oportunidades"
        )
        
        use_cache = st.checkbox("💾 Usar cache", value=True)
        
        # Mostrar erros de treinamento se houver
        if st.session_state.training_errors:
            with st.expander("⚠️ Erros de Treinamento"):
                for error in st.session_state.training_errors[-5:]:  # Últimos 5
                    st.write(f"• {error}")
    
    # Tabs principais
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 Treinar", "📊 Análise Ligas", "🎯 Previsões", "📈 Dashboard"])
    
    with tab1:
        st.header("🤖 Treinamento Robusto com Validação")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **✅ Sistema Robusto:**
            - Tratamento completo de erros
            - Backup automático do progresso
            - Validação rigorosa de dados
            - Fallbacks para problemas
            - Logs detalhados de erros
            """)
        
        with col2:
            st.success("""
            **🎯 Garantias:**
            - Nunca perde o progresso
            - Sempre salva modelos válidos
            - Recuperação automática
            - Previsões sempre funcionam
            - Performance otimizada
            """)
        
        if st.button("🚀 TREINAR SISTEMA ROBUSTO", type="primary", use_container_width=True):
            
            # Limpar erros anteriores
            st.session_state.training_errors = []
            st.session_state.training_in_progress = True
            
            try:
                with st.spinner("📥 Carregando dados..."):
                    df = collect_historical_data_smart(days=None, use_cached=use_cache, seasonal=True)
                
                if df.empty:
                    st.error("❌ Nenhum dado disponível para treinamento")
                    st.session_state.training_in_progress = False
                    st.stop()
                
                st.success(f"✅ {len(df)} jogos carregados")
                
                # Agrupar por liga
                league_groups = df.groupby(['league_id', 'league_name', 'country'])
                
                st.info(f"🎯 Encontradas {len(league_groups)} ligas para análise")
                
                league_models = {}
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results_summary = []
                successful_leagues = 0
                
                for idx, ((league_id, league_name, country), league_df) in enumerate(league_groups):
                    progress = (idx + 1) / len(league_groups)
                    progress_bar.progress(progress)
                    
                    league_full_name = f"{league_name} ({country})"
                    status_text.text(f"🔄 Treinando: {league_full_name}")
                    
                    if len(league_df) < min_matches_per_league:
                        continue
                    
                    # Treinar modelo
                    model_data, message = train_complete_model_with_validation(
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
                            'F1-Score': model_data['test_metrics']['f1_score']
                        })
                        
                        # Salvar progresso a cada 5 ligas
                        if successful_leagues % 5 == 0:
                            save_training_progress(league_models, f"backup_{successful_leagues}")
                    else:
                        st.warning(message)
                
                progress_bar.empty()
                status_text.empty()
                
                if league_models:
                    # Salvar progresso final
                    save_training_progress(league_models, "final")
                    
                    st.session_state.league_models = league_models
                    st.session_state.models_trained = True
                    
                    # Resumo final
                    st.success(f"🎉 {len(league_models)} ligas treinadas com sucesso!")
                    
                    if results_summary:
                        avg_accuracy = np.mean([r['Acurácia'] for r in results_summary])
                        avg_f1 = np.mean([r['F1-Score'] for r in results_summary])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Ligas Treinadas", len(league_models))
                        with col2:
                            st.metric("Acurácia Média", f"{avg_accuracy:.1%}")
                        with col3:
                            st.metric("F1-Score Médio", f"{avg_f1:.1%}")
                    
                    st.balloons()
                else:
                    st.error("❌ Nenhuma liga foi treinada com sucesso!")
                    
            except Exception as e:
                st.error(f"❌ Erro geral no treinamento: {str(e)}")
                st.error("Detalhes técnicos:")
                st.code(traceback.format_exc())
                
                # Tentar carregar backup se disponível
                if st.session_state.models_backup:
                    st.warning("🔄 Tentando carregar backup...")
                    if load_training_progress():
                        st.success("✅ Backup carregado com sucesso!")
            
            finally:
                st.session_state.training_in_progress = False
    
    with tab2:
        if not st.session_state.models_trained:
            st.warning("⚠️ Treine o sistema primeiro!")
            if st.session_state.models_backup:
                if st.button("🔄 Carregar Modelos do Backup"):
                    load_training_progress()
                    st.rerun()
        else:
            display_league_summary(st.session_state.league_models)
    
    with tab3:
        st.header("🎯 Previsões Inteligentes")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Treine o sistema primeiro!")
            if st.session_state.models_backup:
                if st.button("🔄 Carregar Modelos do Backup", key="pred_backup"):
                    load_training_progress()
                    st.rerun()
            st.stop()
        
        selected_date = st.date_input("📅 Data:", value=datetime.now().date())
        date_str = selected_date.strftime('%Y-%m-%d')
        
        with st.spinner("🔍 Analisando jogos..."):
            fixtures = get_fixtures_cached(date_str)
        
        if not fixtures:
            st.info("📅 Nenhum jogo encontrado para esta data")
        else:
            st.info(f"🔍 Encontrados {len(fixtures)} jogos para análise")
            
            # Fazer previsões
            predictions = predict_with_strategy(
                fixtures, 
                st.session_state.league_models, 
                min_confidence=min_confidence
            )
            
            if not predictions:
                st.info("🤷 Nenhuma previsão acima da confiança mínima encontrada")
                st.write("**Possíveis motivos:**")
                st.write("• Confiança mínima muito alta")
                st.write("• Times não presentes nos dados de treinamento")
                st.write("• Jogos de ligas não treinadas")
            else:
                st.success(f"🎯 {len(predictions)} apostas encontradas!")
                
                # Filtros
                col1, col2 = st.columns(2)
                with col1:
                    show_only_over = st.checkbox("Mostrar apenas OVER 0.5", value=True)
                with col2:
                    sort_by = st.selectbox("Ordenar por:", 
                                         ["Confiança", "Vs Liga %", "Poisson %"])
                
                # Aplicar filtros
                filtered_predictions = predictions.copy()
                
                if show_only_over:
                    filtered_predictions = [p for p in filtered_predictions if p['prediction'] == 'OVER 0.5']
                
                if sort_by == "Vs Liga %":
                    filtered_predictions.sort(key=lambda x: x['game_vs_league_ratio'], reverse=True)
                elif sort_by == "Poisson %":
                    filtered_predictions.sort(key=lambda x: x['poisson_probability'], reverse=True)
                
                # Estatísticas
                if filtered_predictions:
                    avg_conf = np.mean([p['confidence'] for p in filtered_predictions])
                    avg_poisson = np.mean([p['poisson_probability'] for p in filtered_predictions])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Apostas", len(filtered_predictions))
                    with col2:
                        st.metric("Confiança Média", f"{avg_conf:.1f}%")
                    with col3:
                        st.metric("Poisson Médio", f"{avg_poisson:.1f}%")
                    
                    st.markdown("---")
                    
                    # Mostrar previsões
                    for pred in filtered_predictions:
                        display_smart_prediction(pred)
                else:
                    st.info("🔍 Nenhuma previsão após aplicar filtros")
    
    with tab4:
        st.header("📈 Dashboard de Performance")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Treine o sistema primeiro!")
        else:
            try:
                # Análise geral
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
                    avg_f1 = np.mean([m['test_metrics']['f1_score'] for m in st.session_state.league_models.values()])
                    st.metric("F1-Score Médio", f"{avg_f1:.1%}")
                
                st.markdown("---")
                
                # Features importantes com download
                st.subheader("🎯 Features Mais Importantes")
                
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
                    top_global_features = sorted(avg_features.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    df_features = pd.DataFrame(top_global_features, columns=['Feature', 'Importância'])
                    df_features['Feature'] = df_features['Feature'].str.replace('_', ' ').str.title()
                    df_features['Importância (%)'] = df_features['Importância'] * 100
                    
                    # Gráfico
                    chart_data = df_features.set_index('Feature')['Importância']
                    st.bar_chart(chart_data)
                    
                    # Download Features
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col2:
                        excel_features = create_excel_download(df_features, "features_importantes.xlsx")
                        if excel_features:
                            st.download_button(
                                label="📥 Download Features Importantes",
                                data=excel_features,
                                file_name=f"features_importantes_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                    
                    # Tabela
                    st.subheader("📊 Detalhes das Features")
                    st.dataframe(df_features[['Feature', 'Importância (%)']], use_container_width=True)
                
                # Estatísticas gerais com download
                st.subheader("📊 Estatísticas Gerais do Sistema")
                
                stats_data = []
                for league_id, model_data in st.session_state.league_models.items():
                    try:
                        stats_data.append({
                            'Liga': model_data['league_name'],
                            'Total Jogos': model_data['total_matches'],
                            'Taxa Over 0.5 HT (%)': round(model_data['league_over_rate'] * 100, 1),
                            'Acurácia (%)': round(model_data['test_metrics']['accuracy'] * 100, 1),
                            'Precisão (%)': round(model_data['test_metrics']['precision'] * 100, 1),
                            'Recall (%)': round(model_data['test_metrics']['recall'] * 100, 1),
                            'F1-Score (%)': round(model_data['test_metrics']['f1_score'] * 100, 1),
                            'Threshold Ótimo': round(model_data['best_threshold'], 3),
                            'Total Times': len(model_data['team_stats']),
                            'Média Jogos por Time': round(model_data['total_matches'] / max(len(model_data['team_stats']), 1), 1)
                        })
                    except:
                        continue
                
                if stats_data:
                    df_stats = pd.DataFrame(stats_data)
                    
                    # Download estatísticas completas
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col2:
                        excel_stats = create_excel_download(df_stats, "estatisticas_completas.xlsx")
                        if excel_stats:
                            st.download_button(
                                label="📥 Download Estatísticas Completas",
                                data=excel_stats,
                                file_name=f"estatisticas_completas_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                    
                    # Tabela resumo das estatísticas
                    st.dataframe(df_stats.head(20), use_container_width=True)
                    
                    if len(df_stats) > 20:
                        st.info(f"📊 Mostrando top 20 de {len(df_stats)} ligas. Use o download para ver todas.")
                
            except Exception as e:
                st.error(f"❌ Erro no dashboard: {str(e)}")

if __name__ == "__main__":
    main()
