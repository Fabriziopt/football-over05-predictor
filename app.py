import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import joblib
import os
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import poisson
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="⚽ HT Goals AI Ultimate",
    page_icon="🎯",
    layout="wide"
)

# Inicializar session state
if 'league_models' not in st.session_state:
    st.session_state.league_models = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False

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
    except:
        return False, "Erro de conexão"

def get_fixtures_with_retry(date_str, max_retries=3):
    """Busca jogos da API com retry automático"""
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
            else:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return []
        except:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
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
                
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                if 'ht_home' in df.columns and 'ht_away' in df.columns:
                    df['over_05'] = (df['ht_home'] + df['ht_away']) > 0
                elif 'ht_home_goals' in df.columns and 'ht_away_goals' in df.columns:
                    df['over_05'] = (df['ht_home_goals'] + df['ht_away_goals']) > 0
                
                return df, f"✅ {len(df)} jogos carregados"
            except:
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
    """Coleta inteligente com opção sazonal"""
    
    if seasonal and days is None:
        days, start_date = get_seasonal_data_period()
        st.info(f"📅 Modo Sazonal: Buscando dados desde {start_date.strftime('%d/%m/%Y')} ({days} dias)")
    elif days is None:
        days = 365
    
    if use_cached:
        df_cache, message = load_historical_data()
        if df_cache is not None and not df_cache.empty:
            if 'date' in df_cache.columns:
                df_cache['date'] = pd.to_datetime(df_cache['date'])
                current_date = datetime.now()
                cutoff_date = current_date - timedelta(days=days)
                df_filtered = df_cache[df_cache['date'] >= cutoff_date].copy()
                
                if len(df_filtered) > 0:
                    return df_filtered
    
    # Buscar da API se necessário
    st.warning("⚠️ Coletando dados da API...")
    
    sample_days = []
    for i in range(min(60, days)):
        sample_days.append(i + 1)
    if days > 60:
        for i in range(60, min(180, days), 3):
            sample_days.append(i + 1)
    if days > 180:
        for i in range(180, min(365, days), 5):
            sample_days.append(i + 1)
    if days > 365:
        for i in range(365, days, 7):
            sample_days.append(i + 1)
    
    all_data = []
    progress_bar = st.progress(0)
    
    for idx, day_offset in enumerate(sample_days):
        date = datetime.now() - timedelta(days=day_offset)
        date_str = date.strftime('%Y-%m-%d')
        
        try:
            fixtures = get_fixtures_cached(date_str)
            if fixtures:
                for match in fixtures:
                    if match['fixture']['status']['short'] == 'FT' and match.get('score', {}).get('halftime'):
                        match_data = extract_match_features(match)
                        if match_data:
                            all_data.append(match_data)
        except:
            continue
        
        progress = (idx + 1) / len(sample_days)
        progress_bar.progress(progress)
        
        if idx % 5 == 0:
            time.sleep(0.3)
    
    progress_bar.empty()
    
    return pd.DataFrame(all_data)

def extract_match_features(match):
    """Extrai features básicas do jogo"""
    try:
        ht_home = match['score']['halftime']['home']
        ht_away = match['score']['halftime']['away']
        
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
            'ht_home_goals': ht_home,
            'ht_away_goals': ht_away,
            'ht_total_goals': ht_home + ht_away,
            'over_05': 1 if (ht_home + ht_away) > 0 else 0
        }
        
        return features
    except:
        return None

def calculate_poisson_probabilities(home_avg, away_avg):
    """Calcula probabilidades usando distribuição de Poisson"""
    
    # Lambda para cada time (média de gols esperados)
    home_lambda = home_avg / 2  # Dividir por 2 para HT
    away_lambda = away_avg / 2
    
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
        'poisson_over_05': prob_over_05,
        'expected_goals_ht': expected_goals_ht,
        'home_lambda': home_lambda,
        'away_lambda': away_lambda
    }

def calculate_advanced_features(league_df):
    """Calcula features avançadas incluindo Poisson e análise casa/fora"""
    
    if 'over_05' not in league_df.columns:
        if 'ht_home_goals' in league_df.columns and 'ht_away_goals' in league_df.columns:
            league_df['over_05'] = (league_df['ht_home_goals'] + league_df['ht_away_goals']) > 0
    
    if 'date' in league_df.columns:
        league_df = league_df.sort_values('date').reset_index(drop=True)
    
    # Estatísticas da liga
    league_over_rate = league_df['over_05'].mean()
    league_avg_goals = league_df['ht_total_goals'].mean()
    
    # Estatísticas por time
    team_stats = {}
    unique_teams = pd.concat([league_df['home_team_id'], league_df['away_team_id']]).unique()
    
    for team_id in unique_teams:
        # Jogos em casa
        home_matches = league_df[league_df['home_team_id'] == team_id]
        # Jogos fora
        away_matches = league_df[league_df['away_team_id'] == team_id]
        # Todos os jogos
        all_matches = pd.concat([home_matches, away_matches])
        
        if len(all_matches) > 0:
            team_name = home_matches.iloc[0]['home_team'] if len(home_matches) > 0 else away_matches.iloc[0]['away_team']
            
            # Análise detalhada casa/fora
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
                'home_attack_strength': home_goals_scored / (league_avg_goals/2 + 0.01),
                'home_defense_strength': home_goals_conceded / (league_avg_goals/2 + 0.01),
                'away_attack_strength': away_goals_scored / (league_avg_goals/2 + 0.01),
                'away_defense_strength': away_goals_conceded / (league_avg_goals/2 + 0.01)
            }
    
    # Criar features para ML
    features = []
    
    for idx, row in league_df.iterrows():
        home_id = row['home_team_id']
        away_id = row['away_team_id']
        
        if home_id not in team_stats or away_id not in team_stats:
            continue
        
        home_stats = team_stats[home_id]
        away_stats = team_stats[away_id]
        
        # Poisson predictions
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
            'game_pace_index': (home_expected + away_expected),
            
            # Comparação com média da liga
            'over_rate_vs_league': ((home_stats['home_over_rate'] + away_stats['away_over_rate']) / 2) / league_over_rate,
            'expected_vs_league': poisson_calc['expected_goals_ht'] / league_avg_goals,
            
            # Jogos disputados
            'home_games_played': home_stats['home_games'],
            'away_games_played': away_stats['away_games'],
            'min_games': min(home_stats['home_games'], away_stats['away_games']),
            
            # Target
            'target': row['over_05']
        }
        
        features.append(feature_row)
    
    return pd.DataFrame(features), team_stats, league_over_rate

def train_complete_model_with_validation(league_df, league_id, league_name):
    """Treina modelo com validação completa (treino/validação/teste)"""
    
    if len(league_df) < 30:  # Reduzido para 30 como solicitado
        return None, f"❌ Dados insuficientes para {league_name} (mínimo 30 jogos)"
    
    try:
        # Preparar features avançadas
        features_df, team_stats, league_over_rate = calculate_advanced_features(league_df)
        
        if len(features_df) < 30:
            return None, f"❌ Features insuficientes para {league_name}"
        
        feature_cols = [col for col in features_df.columns if col != 'target']
        X = features_df[feature_cols]
        y = features_df['target']
        
        # Dividir dados: 60% treino, 20% validação, 20% teste
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
        
        # Escalar dados
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Modelos
        models = {
            'rf': RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
            'gb': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=8, random_state=42),
            'et': ExtraTreesClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
        }
        
        # Treinar e validar cada modelo
        best_model = None
        best_score = 0
        results = {}
        
        for name, model in models.items():
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
            pred_threshold = (test_pred_proba >= threshold).astype(int)
            f1_threshold = f1_score(y_test, pred_threshold, zero_division=0)
            
            if f1_threshold > best_f1:
                best_f1 = f1_threshold
                best_threshold = threshold
        
        # Retreinar no dataset completo com melhor modelo
        X_scaled = scaler.fit_transform(X)
        best_model.fit(X_scaled, y)
        
        # Feature importance
        feature_importance = dict(zip(feature_cols, best_model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
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
        return None, f"❌ Erro ao treinar {league_name}: {str(e)}"

def predict_with_strategy(fixtures, league_models, min_confidence=60):
    """Faz previsões com estratégia inteligente"""
    
    predictions = []
    
    for fixture in fixtures:
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
        
        try:
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
                'over_rate_vs_league': ((home_stats['home_over_rate'] + away_stats['away_over_rate']) / 2) / league_over_rate,
                'expected_vs_league': poisson_calc['expected_goals_ht'] / 0.5,
                'home_games_played': home_stats['home_games'],
                'away_games_played': away_stats['away_games'],
                'min_games': min(home_stats['home_games'], away_stats['away_games'])
            }
            
            # Fazer previsão
            X = pd.DataFrame([features])[feature_cols]
            X_scaled = scaler.transform(X)
            
            pred_proba = model.predict_proba(X_scaled)[0]
            confidence = pred_proba[1] * 100
            
            # Aplicar threshold otimizado
            pred_class = 1 if pred_proba[1] >= best_threshold else 0
            
            # Calcular indicadores de força
            game_vs_league = features['combined_over_rate'] / league_over_rate
            
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

def display_smart_prediction(pred):
    """Exibe previsão com análise inteligente"""
    
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
            st.write(f"📈 **Média do Jogo:** {(pred['home_team_stats']['home_over_rate'] + pred['away_team_stats']['away_over_rate'])*50:.1f}%")
        
        with col2:
            st.write(f"🏠 **{pred['home_team']}**")
            st.write(f"- Casa: {pred['home_team_stats']['home_over_rate']*100:.1f}%")
            st.write(f"- Força Ataque: {pred['home_team_stats']['home_attack_strength']:.2f}")
            st.write(f"- Jogos Casa: {pred['home_team_stats']['home_games']}")
        
        with col3:
            st.write(f"✈️ **{pred['away_team']}**")
            st.write(f"- Fora: {pred['away_team_stats']['away_over_rate']*100:.1f}%")
            st.write(f"- Força Ataque: {pred['away_team_stats']['away_attack_strength']:.2f}")
            st.write(f"- Jogos Fora: {pred['away_team_stats']['away_games']}")
        
        # Previsões
        st.markdown("### 🎯 Análise Preditiva")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("ML Probability", f"{pred['ml_probability']:.1f}%")
        with col2:
            st.metric("Poisson Probability", f"{pred['poisson_probability']:.1f}%")
        with col3:
            st.metric("Gols Esperados HT", f"{pred['expected_goals_ht']:.2f}")
        
        # Recomendação
        if pred['prediction'] == 'OVER 0.5':
            if pred['confidence'] > 70 and pred['game_vs_league_ratio'] > 1.1:
                st.success(f"✅ **APOSTAR: {pred['prediction']} HT** (Alta Confiança)")
            else:
                st.info(f"📊 **Considerar: {pred['prediction']} HT** (Confiança Moderada)")
        
        with st.expander("📊 Análise Detalhada"):
            st.write("**Fatores principais:**")
            for feature, importance in pred['top_features'][:3]:
                feature_name = feature.replace('_', ' ').title()
                st.write(f"- {feature_name}: {importance:.2%}")
            
            st.write("\n**Performance do Modelo:**")
            metrics = pred['model_metrics']
            st.write(f"- Acurácia: {metrics['accuracy']:.1%}")
            st.write(f"- Precisão: {metrics['precision']:.1%}")
            st.write(f"- F1-Score: {metrics['f1_score']:.1%}")
        
        st.markdown("---")

def display_league_summary(league_models):
    """Exibe resumo das ligas com gráficos"""
    
    st.header("📊 Análise das Ligas")
    
    # Dados para gráfico
    league_data = []
    for league_id, model_data in league_models.items():
        league_data.append({
            'Liga': model_data['league_name'],
            'Over_05_HT': model_data['league_over_rate'] * 100,
            'Jogos': model_data['total_matches'],
            'F1_Score': model_data['test_metrics']['f1_score'] * 100,
            'Acuracia': model_data['test_metrics']['accuracy'] * 100
        })
    
    df_leagues = pd.DataFrame(league_data)
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de barras - Taxa Over 0.5 HT por Liga (usando nomes de coluna seguros)
        df_top = df_leagues.sort_values('Over_05_HT', ascending=False).head(15)
        fig = px.bar(df_top, 
                     x='Liga', y='Over_05_HT',
                     title='Top 15 Ligas - Taxa Over 0.5 HT',
                     color='Over_05_HT',
                     color_continuous_scale='RdYlGn')
        fig.update_layout(xaxis_tickangle=-45, yaxis_title="Over 0.5 HT (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Scatter plot - Acurácia vs F1-Score
        fig = px.scatter(df_leagues, x='Acuracia', y='F1_Score',
                        size='Jogos', hover_name='Liga',
                        title='Performance dos Modelos por Liga',
                        labels={'Acuracia': 'Acurácia (%)', 'F1_Score': 'F1-Score (%)', 'size': 'Número de Jogos'})
        fig.add_hline(y=75, line_dash="dash", line_color="red", 
                     annotation_text="Mínimo Recomendado")
        st.plotly_chart(fig, use_container_width=True)
    
    # Tabela resumo
    st.subheader("📋 Resumo Detalhado")
    
    # Preparar dados para exibição
    df_display = df_leagues.sort_values('F1_Score', ascending=False)
    df_display['Over 0.5 HT %'] = df_display['Over_05_HT'].apply(lambda x: f"{x:.1f}%")
    df_display['F1-Score %'] = df_display['F1_Score'].apply(lambda x: f"{x:.1f}%")
    df_display['Acurácia %'] = df_display['Acuracia'].apply(lambda x: f"{x:.1f}%")
    
    # Mostrar apenas colunas relevantes para display
    display_cols = ['Liga', 'Over 0.5 HT %', 'Jogos', 'Acurácia %', 'F1-Score %']
    st.dataframe(df_display[display_cols], use_container_width=True)

def main():
    st.title("⚽ HT Goals AI Ultimate - Sistema Completo")
    st.markdown("🎯 **Máxima taxa de acerto com análise inteligente**")
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configurações")
        
        # Status
        conn_ok, conn_msg = test_api_connection()
        if conn_ok:
            st.success("✅ API conectada")
        else:
            st.error(f"❌ {conn_msg}")
        
        if st.session_state.models_trained and st.session_state.league_models:
            st.success(f"✅ {len(st.session_state.league_models)} ligas treinadas")
        else:
            st.warning("⚠️ Modelos não treinados")
        
        st.markdown("---")
        
        # Configurações flexíveis
        st.markdown("### 📊 Parâmetros")
        
        min_matches_per_league = st.slider(
            "Mínimo jogos por liga:",
            min_value=20,
            max_value=100,
            value=30,
            help="30 jogos já permite boas previsões"
        )
        
        min_confidence = st.slider(
            "Confiança mínima:",
            min_value=50,
            max_value=80,
            value=60,
            help="60% permite mais apostas mantendo qualidade"
        )
        
        st.info("""
        💡 **Dicas:**
        - Mínimo 30 jogos: Bom equilíbrio
        - Confiança 60%+: Mais oportunidades
        - Sem limite de apostas/dia
        - Análise Poisson incluída
        """)
        
        use_cache = st.checkbox("💾 Usar cache", value=True)
    
    # Tabs principais
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 Treinar", "📊 Análise Ligas", "🎯 Previsões", "📈 Dashboard"])
    
    with tab1:
        st.header("🤖 Treinamento Completo com Validação")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **✅ Sistema Completo:**
            - Treino / Validação / Teste
            - Análise Casa vs Fora
            - Modelo Poisson integrado
            - Comparação com média da liga
            - Threshold otimizado
            """)
        
        with col2:
            st.success("""
            **🎯 Estratégias:**
            - Mínimo 30 jogos por liga
            - Sem limite de apostas
            - Múltiplos indicadores
            - Feature importance
            - Taxa de acerto maximizada
            """)
        
        if st.button("🚀 TREINAR SISTEMA COMPLETO", type="primary", use_container_width=True):
            with st.spinner("📥 Carregando dados sazonais..."):
                df = collect_historical_data_smart(days=None, use_cached=use_cache, seasonal=True)
            
            if df.empty:
                st.error("❌ Sem dados")
                st.stop()
            
            st.success(f"✅ {len(df)} jogos carregados")
            
            # Agrupar por liga
            league_groups = df.groupby(['league_id', 'league_name', 'country'])
            
            league_models = {}
            progress_bar = st.progress(0)
            
            results_summary = []
            
            for idx, ((league_id, league_name, country), league_df) in enumerate(league_groups):
                progress = (idx + 1) / len(league_groups)
                progress_bar.progress(progress)
                
                if len(league_df) < min_matches_per_league:
                    continue
                
                # Treinar modelo com validação completa
                model_data, message = train_complete_model_with_validation(
                    league_df, league_id, f"{league_name} ({country})"
                )
                
                if model_data:
                    league_models[league_id] = model_data
                    st.success(message)
                    
                    results_summary.append({
                        'Liga': f"{league_name} ({country})",
                        'Jogos': len(league_df),
                        'Acurácia': model_data['test_metrics']['accuracy'],
                        'F1-Score': model_data['test_metrics']['f1_score']
                    })
            
            progress_bar.empty()
            
            if league_models:
                st.session_state.league_models = league_models
                st.session_state.models_trained = True
                
                # Resumo
                st.success(f"🎉 {len(league_models)} ligas treinadas com sucesso!")
                
                # Estatísticas gerais
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
    
    with tab2:
        if not st.session_state.models_trained:
            st.warning("⚠️ Treine o sistema primeiro!")
            st.stop()
        
        display_league_summary(st.session_state.league_models)
    
    with tab3:
        st.header("🎯 Previsões Inteligentes")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Treine o sistema primeiro!")
            st.stop()
        
        selected_date = st.date_input("📅 Data:", value=datetime.now().date())
        date_str = selected_date.strftime('%Y-%m-%d')
        
        with st.spinner("🔍 Analisando jogos..."):
            fixtures = get_fixtures_cached(date_str)
        
        if not fixtures:
            st.info("📅 Sem jogos hoje")
        else:
            # Fazer previsões
            predictions = predict_with_strategy(
                fixtures, 
                st.session_state.league_models, 
                min_confidence=min_confidence
            )
            
            if not predictions:
                st.info("🤷 Nenhuma previsão acima da confiança mínima")
            else:
                st.success(f"🎯 {len(predictions)} apostas encontradas!")
                
                # Filtros adicionais
                col1, col2 = st.columns(2)
                with col1:
                    show_only_over = st.checkbox("Mostrar apenas OVER 0.5", value=True)
                with col2:
                    sort_by = st.selectbox("Ordenar por:", 
                                         ["Confiança", "Vs Liga %", "Poisson %"])
                
                # Filtrar e ordenar
                if show_only_over:
                    predictions = [p for p in predictions if p['prediction'] == 'OVER 0.5']
                
                if sort_by == "Vs Liga %":
                    predictions.sort(key=lambda x: x['game_vs_league_ratio'], reverse=True)
                elif sort_by == "Poisson %":
                    predictions.sort(key=lambda x: x['poisson_probability'], reverse=True)
                
                # Estatísticas
                if predictions:
                    avg_conf = np.mean([p['confidence'] for p in predictions])
                    avg_poisson = np.mean([p['poisson_probability'] for p in predictions])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Apostas", len(predictions))
                    with col2:
                        st.metric("Confiança Média", f"{avg_conf:.1f}%")
                    with col3:
                        st.metric("Poisson Médio", f"{avg_poisson:.1f}%")
                    
                    st.markdown("---")
                    
                    # Mostrar previsões
                    for pred in predictions:
                        display_smart_prediction(pred)
    
    with tab4:
        st.header("📈 Dashboard de Performance")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Treine o sistema primeiro!")
            st.stop()
        
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
        
        # Top features globais
        st.subheader("🎯 Features Mais Importantes (Global)")
        
        all_features = {}
        for model_data in st.session_state.league_models.values():
            for feature, importance in model_data['top_features']:
                if feature not in all_features:
                    all_features[feature] = []
                all_features[feature].append(importance)
        
        avg_features = {f: np.mean(imps) for f, imps in all_features.items()}
        top_global_features = sorted(avg_features.items(), key=lambda x: x[1], reverse=True)[:10]
        
        df_features = pd.DataFrame(top_global_features, columns=['Feature', 'Importancia'])
        df_features['Feature'] = df_features['Feature'].str.replace('_', ' ').str.title()
        
        fig = px.bar(df_features, x='Importancia', y='Feature', orientation='h',
                    title='Top 10 Features Globais')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
