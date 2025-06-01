import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import joblib
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
# XGBoost e LightGBM não disponíveis no Streamlit Cloud
# Vamos usar apenas RandomForest e GradientBoosting
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="⚽ HT Goals AI - Alta Precisão",
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
                try:
                    data = response.json()
                    fixtures = data.get('response', [])
                    return fixtures
                except:
                    return []
            else:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return []
        except:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
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
        "data/historical_matches_seasonal.parquet",
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
    """Calcula período ideal baseado na temporada - NOVO!"""
    current_date = datetime.now()
    current_month = current_date.month
    current_year = current_date.year
    
    # Lógica sazonal para futebol europeu
    if current_month >= 8:  # Agosto a Dezembro
        # Estamos no início/meio da temporada
        # Pegar temporada anterior completa + atual até agora
        start_date = datetime(current_year - 1, 8, 1)  # Agosto do ano passado
        days_back = (current_date - start_date).days
    else:  # Janeiro a Julho
        # Estamos no fim da temporada
        # Pegar desde agosto do ano passado
        start_date = datetime(current_year - 1, 8, 1)
        days_back = (current_date - start_date).days
    
    # Garantir mínimo de 365 dias
    days_back = max(days_back, 365)
    
    return days_back, start_date

def collect_historical_data_smart(days=None, use_cached=True, seasonal=True):
    """Coleta inteligente com opção sazonal - MELHORADO!"""
    
    # Se modo sazonal, calcular período ideal
    if seasonal and days is None:
        days, start_date = get_seasonal_data_period()
        st.info(f"📅 Modo Sazonal: Buscando dados desde {start_date.strftime('%d/%m/%Y')} ({days} dias)")
    elif days is None:
        days = 365  # Padrão 1 ano
    
    if use_cached:
        df_cache, message = load_historical_data()
        if df_cache is not None and not df_cache.empty:
            if 'date' in df_cache.columns:
                df_cache['date'] = pd.to_datetime(df_cache['date'])
                current_date = datetime.now()
                cutoff_date = current_date - timedelta(days=days)
                df_filtered = df_cache[df_cache['date'] >= cutoff_date].copy()
                
                if len(df_filtered) > 0:
                    # Verificar se temos dados suficientes
                    actual_days = (df_filtered['date'].max() - df_filtered['date'].min()).days
                    if actual_days >= (days * 0.7):  # 70% do período desejado
                        st.success(f"✅ Cache com {len(df_filtered)} jogos ({actual_days} dias)")
                        return df_filtered
                    else:
                        st.warning(f"⚠️ Cache insuficiente: apenas {actual_days} dias")
    
    # Se não tem cache adequado, buscar da API
    st.warning("⚠️ Coletando dados históricos completos da API...")
    
    # Para períodos longos, amostragem inteligente
    sample_days = []
    
    # Últimos 60 dias completos (jogos recentes)
    for i in range(min(60, days)):
        sample_days.append(i + 1)
    
    # 60-180 dias: a cada 3 dias
    if days > 60:
        for i in range(60, min(180, days), 3):
            sample_days.append(i + 1)
    
    # 180-365 dias: a cada 5 dias
    if days > 180:
        for i in range(180, min(365, days), 5):
            sample_days.append(i + 1)
    
    # Mais de 1 ano: a cada 7 dias
    if days > 365:
        for i in range(365, days, 7):
            sample_days.append(i + 1)
    
    all_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, day_offset in enumerate(sample_days):
        date = datetime.now() - timedelta(days=day_offset)
        date_str = date.strftime('%Y-%m-%d')
        
        status_text.text(f"📅 Coletando: {date_str}")
        
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
    status_text.empty()
    
    if all_data:
        df_new = pd.DataFrame(all_data)
        
        # Salvar cache atualizado
        try:
            cache_file = "data/historical_matches_seasonal.parquet"
            os.makedirs("data", exist_ok=True)
            df_new.to_parquet(cache_file)
            st.success(f"💾 Cache sazonal salvo com {len(df_new)} jogos")
        except:
            pass
        
        return df_new
    
    return pd.DataFrame()

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

def calculate_h2h_features(home_id, away_id, df, current_date, num_games=5):
    """Calcula features de confronto direto (H2H) - NOVO!"""
    
    # Filtrar confrontos anteriores
    h2h_matches = df[
        ((df['home_team_id'] == home_id) & (df['away_team_id'] == away_id)) |
        ((df['home_team_id'] == away_id) & (df['away_team_id'] == home_id))
    ]
    
    if 'date' in h2h_matches.columns:
        h2h_matches = h2h_matches[h2h_matches['date'] < current_date]
    
    # Pegar últimos N confrontos
    h2h_matches = h2h_matches.sort_values('date', ascending=False).head(num_games)
    
    if len(h2h_matches) == 0:
        return {
            'h2h_games': 0,
            'h2h_over_rate': 0.5,
            'h2h_avg_goals': 0.5,
            'h2h_home_advantage': 0.5
        }
    
    # Calcular estatísticas H2H
    h2h_over_rate = h2h_matches['over_05'].mean()
    h2h_avg_goals = h2h_matches['ht_total_goals'].mean()
    
    # Vantagem do mandante nos confrontos diretos
    home_games = h2h_matches[h2h_matches['home_team_id'] == home_id]
    if len(home_games) > 0:
        h2h_home_advantage = home_games['over_05'].mean()
    else:
        h2h_home_advantage = 0.5
    
    return {
        'h2h_games': len(h2h_matches),
        'h2h_over_rate': h2h_over_rate,
        'h2h_avg_goals': h2h_avg_goals,
        'h2h_home_advantage': h2h_home_advantage
    }

def calculate_form_features(team_id, df, current_date, last_n_games=5):
    """Calcula forma recente do time - NOVO!"""
    
    # Jogos do time
    team_matches = df[
        (df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)
    ]
    
    if 'date' in team_matches.columns:
        team_matches = team_matches[team_matches['date'] < current_date]
    
    # Últimos N jogos
    recent_matches = team_matches.sort_values('date', ascending=False).head(last_n_games)
    
    if len(recent_matches) == 0:
        return {
            'form_games': 0,
            'form_over_rate': 0.5,
            'form_trend': 0.0,
            'days_since_last': 30
        }
    
    # Taxa de over recente
    form_over_rate = recent_matches['over_05'].mean()
    
    # Tendência (comparar primeira metade com segunda metade)
    if len(recent_matches) >= 4:
        first_half = recent_matches.iloc[len(recent_matches)//2:]['over_05'].mean()
        second_half = recent_matches.iloc[:len(recent_matches)//2]['over_05'].mean()
        form_trend = second_half - first_half
    else:
        form_trend = 0.0
    
    # Dias desde último jogo
    last_game_date = pd.to_datetime(recent_matches.iloc[0]['date'])
    days_since_last = (pd.to_datetime(current_date) - last_game_date).days
    
    return {
        'form_games': len(recent_matches),
        'form_over_rate': form_over_rate,
        'form_trend': form_trend,
        'days_since_last': min(days_since_last, 30)
    }

def prepare_advanced_features_v2(league_df):
    """Prepara features avançadas incluindo H2H e forma - MELHORADO!"""
    
    if 'over_05' not in league_df.columns:
        if 'ht_home_goals' in league_df.columns and 'ht_away_goals' in league_df.columns:
            league_df['over_05'] = (league_df['ht_home_goals'] + league_df['ht_away_goals']) > 0
    
    if 'date' in league_df.columns:
        league_df = league_df.sort_values('date').reset_index(drop=True)
    
    # Estatísticas gerais
    league_over_rate = league_df['over_05'].mean()
    
    # Inicializar estatísticas dos times
    team_stats = {}
    unique_teams = pd.concat([league_df['home_team_id'], league_df['away_team_id']]).unique()
    
    for team_id in unique_teams:
        team_home_matches = league_df[league_df['home_team_id'] == team_id]
        team_away_matches = league_df[league_df['away_team_id'] == team_id]
        team_all_matches = pd.concat([team_home_matches, team_away_matches])
        
        if len(team_all_matches) > 0:
            team_name = team_home_matches.iloc[0]['home_team'] if len(team_home_matches) > 0 else team_away_matches.iloc[0]['away_team']
            
            team_stats[team_id] = {
                'team_name': team_name,
                'games': 0,
                'over_05': 0,
                'over_rate': 0,
                'home_games': 0,
                'home_over': 0,
                'home_over_rate': 0,
                'away_games': 0,
                'away_over': 0,
                'away_over_rate': 0,
                'goals_list': [],
                'over_list': []
            }
    
    features = []
    
    for idx, row in league_df.iterrows():
        home_id = row['home_team_id']
        away_id = row['away_team_id']
        current_date = row['date'] if 'date' in row else datetime.now()
        
        if home_id not in team_stats or away_id not in team_stats:
            continue
        
        home_stats = team_stats[home_id]
        away_stats = team_stats[away_id]
        
        # Features básicas
        home_over_rate = home_stats['over_05'] / max(home_stats['games'], 1)
        away_over_rate = away_stats['over_05'] / max(away_stats['games'], 1)
        
        # Features H2H - NOVO!
        h2h_features = calculate_h2h_features(home_id, away_id, league_df[:idx], current_date)
        
        # Features de forma - NOVO!
        home_form = calculate_form_features(home_id, league_df[:idx], current_date)
        away_form = calculate_form_features(away_id, league_df[:idx], current_date)
        
        # Criar feature row completa
        feature_row = {
            # Features básicas
            'home_over_rate': home_over_rate,
            'away_over_rate': away_over_rate,
            'home_home_over_rate': home_stats['home_over_rate'],
            'away_away_over_rate': away_stats['away_over_rate'],
            'league_over_rate': league_over_rate,
            'combined_over_rate': (home_over_rate + away_over_rate) / 2,
            'over_rate_diff': abs(home_over_rate - away_over_rate),
            'home_games_played': home_stats['games'],
            'away_games_played': away_stats['games'],
            
            # Features H2H - NOVO!
            'h2h_games': h2h_features['h2h_games'],
            'h2h_over_rate': h2h_features['h2h_over_rate'],
            'h2h_avg_goals': h2h_features['h2h_avg_goals'],
            'h2h_home_advantage': h2h_features['h2h_home_advantage'],
            
            # Features de forma - NOVO!
            'home_form_over': home_form['form_over_rate'],
            'away_form_over': away_form['form_over_rate'],
            'home_form_trend': home_form['form_trend'],
            'away_form_trend': away_form['form_trend'],
            'home_days_rest': home_form['days_since_last'],
            'away_days_rest': away_form['days_since_last'],
            
            # Features combinadas - NOVO!
            'form_advantage': home_form['form_over_rate'] - away_form['form_over_rate'],
            'rest_advantage': home_form['days_since_last'] - away_form['days_since_last'],
            
            # Target
            'target': row['over_05']
        }
        
        features.append(feature_row)
        
        # Atualizar estatísticas
        ht_home_goals = row.get('ht_home_goals', 0)
        ht_away_goals = row.get('ht_away_goals', 0)
        
        # Atualizar home team
        team_stats[home_id]['games'] += 1
        team_stats[home_id]['over_05'] += row['over_05']
        team_stats[home_id]['home_games'] += 1
        team_stats[home_id]['home_over'] += row['over_05']
        team_stats[home_id]['home_over_rate'] = team_stats[home_id]['home_over'] / max(team_stats[home_id]['home_games'], 1)
        
        # Atualizar away team
        team_stats[away_id]['games'] += 1
        team_stats[away_id]['over_05'] += row['over_05']
        team_stats[away_id]['away_games'] += 1
        team_stats[away_id]['away_over'] += row['over_05']
        team_stats[away_id]['away_over_rate'] = team_stats[away_id]['away_over'] / max(team_stats[away_id]['away_games'], 1)
    
    return pd.DataFrame(features), team_stats, league_over_rate

def train_ensemble_model(league_df, league_id, league_name):
    """Treina ensemble de modelos para máxima precisão - NOVO!"""
    
    if len(league_df) < 100:  # Aumentado mínimo para 100
        return None, f"❌ Dados insuficientes para {league_name} (mínimo 100 jogos)"
    
    try:
        # Preparar features avançadas
        features_df, team_stats, league_over_rate = prepare_advanced_features_v2(league_df)
        
        if len(features_df) < 100:
            return None, f"❌ Features insuficientes para {league_name}"
        
        # Remover amostras com poucos dados
        features_df = features_df[
            (features_df['home_games_played'] >= 5) & 
            (features_df['away_games_played'] >= 5)
        ]
        
        if len(features_df) < 80:
            return None, f"❌ Poucos jogos com dados suficientes para {league_name}"
        
        feature_cols = [col for col in features_df.columns if col != 'target']
        X = features_df[feature_cols]
        y = features_df['target']
        
        # Validação temporal - CRÍTICO!
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Modelos base disponíveis no Streamlit
        rf_model = RandomForestClassifier(
            n_estimators=500,  # Aumentado para compensar
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        gb_model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        
        # Extra trees para diversidade
        from sklearn.ensemble import ExtraTreesClassifier
        et_model = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Ensemble voting com 3 modelos
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('gb', gb_model),
                ('et', et_model)
            ],
            voting='soft'
        )
        
        # Treinar e avaliar com validação temporal
        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Escalar dados
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Treinar ensemble
            ensemble.fit(X_train_scaled, y_train)
            
            # Avaliar
            pred = ensemble.predict(X_test_scaled)
            score = f1_score(y_test, pred)
            scores.append(score)
        
        avg_score = np.mean(scores)
        
        # Treinar no dataset completo se score é bom
        if avg_score < 0.75:  # Rejeitar modelos ruins
            return None, f"❌ Modelo com baixa performance para {league_name} (F1: {avg_score:.2f})"
        
        # Treinar modelo final
        scaler_final = StandardScaler()
        X_scaled = scaler_final.fit_transform(X)
        
        # Calibrar probabilidades para maior confiabilidade
        calibrated_ensemble = CalibratedClassifierCV(ensemble, cv=3, method='sigmoid')
        calibrated_ensemble.fit(X_scaled, y)
        
        # Calcular feature importance (média dos modelos)
        feature_importance = {}
        
        # RF importance
        rf_model.fit(X_scaled, y)
        rf_importance = rf_model.feature_importances_
        
        # GB importance
        gb_model.fit(X_scaled, y)
        gb_importance = gb_model.feature_importances_
        
        # ET importance
        et_model.fit(X_scaled, y)
        et_importance = et_model.feature_importances_
        
        # Média dos 3 modelos
        for i, col in enumerate(feature_cols):
            feature_importance[col] = (rf_importance[i] + gb_importance[i] + et_importance[i]) / 3
        
        # Top features
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Preparar dados do modelo
        model_data = {
            'model': calibrated_ensemble,
            'scaler': scaler_final,
            'feature_cols': feature_cols,
            'team_stats': team_stats,
            'league_id': league_id,
            'league_name': league_name,
            'league_over_rate': league_over_rate,
            'total_matches': len(league_df),
            'avg_cv_score': avg_score,
            'top_features': top_features,
            'min_games_required': 5  # Mínimo de jogos para fazer previsão
        }
        
        return model_data, f"✅ Ensemble treinado para {league_name} (F1: {avg_score:.2f})"
        
    except Exception as e:
        return None, f"❌ Erro ao treinar {league_name}: {str(e)}"

def predict_with_confidence_filter(fixtures, league_models, min_confidence=75):
    """Faz previsões apenas com alta confiança - MELHORADO!"""
    
    predictions = []
    
    for fixture in fixtures:
        if fixture['fixture']['status']['short'] not in ['NS', 'TBD']:
            continue
        
        league_id = fixture['league']['id']
        
        if league_id not in league_models:
            continue
        
        model_data = league_models[league_id]
        
        # Verificar se modelo tem boa performance
        if model_data['avg_cv_score'] < 0.75:
            continue
        
        model = model_data['model']
        scaler = model_data['scaler']
        feature_cols = model_data['feature_cols']
        team_stats = model_data['team_stats']
        league_over_rate = model_data['league_over_rate']
        
        try:
            home_id = fixture['teams']['home']['id']
            away_id = fixture['teams']['away']['id']
            
            if home_id not in team_stats or away_id not in team_stats:
                continue
            
            home_stats = team_stats[home_id]
            away_stats = team_stats[away_id]
            
            # Filtro rigoroso: mínimo de jogos
            if (home_stats['games'] < model_data['min_games_required'] or 
                away_stats['games'] < model_data['min_games_required']):
                continue
            
            # Criar features incluindo H2H e forma
            current_date = datetime.now()
            
            # Features básicas
            features = {
                'home_over_rate': home_stats['over_05'] / max(home_stats['games'], 1),
                'away_over_rate': away_stats['over_05'] / max(away_stats['games'], 1),
                'home_home_over_rate': home_stats['home_over_rate'],
                'away_away_over_rate': away_stats['away_over_rate'],
                'league_over_rate': league_over_rate,
                'combined_over_rate': (home_stats['over_05'] / max(home_stats['games'], 1) + 
                                      away_stats['over_05'] / max(away_stats['games'], 1)) / 2,
                'over_rate_diff': abs(home_stats['over_05'] / max(home_stats['games'], 1) - 
                                    away_stats['over_05'] / max(away_stats['games'], 1)),
                'home_games_played': home_stats['games'],
                'away_games_played': away_stats['games']
            }
            
            # Preencher features faltantes com valores padrão
            for col in feature_cols:
                if col not in features:
                    if 'h2h' in col:
                        features[col] = 0.5  # Sem histórico H2H
                    elif 'form' in col:
                        features[col] = features.get('home_over_rate', 0.5)  # Usar média geral
                    elif 'days' in col or 'rest' in col:
                        features[col] = 7  # Assumir 1 semana
                    else:
                        features[col] = 0
            
            # Fazer previsão
            X = pd.DataFrame([features])[feature_cols]
            X_scaled = scaler.transform(X)
            
            pred_proba = model.predict_proba(X_scaled)[0]
            pred_class = model.predict(X_scaled)[0]
            confidence = max(pred_proba) * 100
            
            # FILTRO DE CONFIANÇA RIGOROSO
            if confidence < min_confidence:
                continue
            
            # Calcular força da previsão
            strength_score = 0
            if features['h2h_games'] > 0 and features['h2h_over_rate'] > 0.7:
                strength_score += 1
            if features['combined_over_rate'] > 0.7:
                strength_score += 1
            if home_stats['home_over_rate'] > 0.7 and away_stats['away_over_rate'] > 0.7:
                strength_score += 1
            
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
                'probability_over': pred_proba[1] * 100,
                'probability_under': pred_proba[0] * 100,
                'model_score': model_data['avg_cv_score'],
                'strength_score': strength_score,
                'top_features': model_data['top_features']
            }
            
            predictions.append(prediction)
            
        except Exception as e:
            continue
    
    # Ordenar por confiança E força
    predictions.sort(key=lambda x: (x['confidence'], x['strength_score']), reverse=True)
    
    # Retornar apenas as melhores
    return predictions[:10]  # Máximo 10 apostas por dia

def display_high_confidence_prediction(pred):
    """Exibe apenas previsões de alta confiança com detalhes - NOVO!"""
    
    with st.container():
        # Header com indicadores de qualidade
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.subheader(f"⚽ {pred['home_team']} vs {pred['away_team']}")
        
        with col2:
            # Badge de confiança
            st.success(f"**{pred['confidence']:.1f}%**")
        
        with col3:
            # Indicador de força
            strength_stars = "⭐" * pred['strength_score']
            st.write(f"**{strength_stars}**")
        
        # Informações essenciais
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"🏆 **Liga:** {pred['league']} ({pred['country']})")
            st.write(f"📊 **Média da Liga:** {pred['league_over_rate']:.1f}%")
        
        with col2:
            home_rate = pred['home_team_stats']['over_05'] / max(pred['home_team_stats']['games'], 1) * 100
            st.write(f"🏠 **{pred['home_team']}:** {home_rate:.1f}%")
            st.write(f"- Casa: {pred['home_team_stats']['home_over_rate']*100:.1f}%")
        
        with col3:
            away_rate = pred['away_team_stats']['over_05'] / max(pred['away_team_stats']['games'], 1) * 100
            st.write(f"✈️ **{pred['away_team']}:** {away_rate:.1f}%")
            st.write(f"- Fora: {pred['away_team_stats']['away_over_rate']*100:.1f}%")
        
        # Previsão
        st.success(f"🎯 **APOSTAR: {pred['prediction']} HT** - Confiança: {pred['confidence']:.1f}%")
        
        # Features importantes
        with st.expander("📊 Por que esta aposta?"):
            st.write("**Fatores principais:**")
            for feature, importance in pred['top_features'][:3]:
                feature_name = feature.replace('_', ' ').title()
                st.write(f"- {feature_name}: {importance:.2%}")
            
            st.write(f"\n**Qualidade do modelo:** F1-Score {pred['model_score']:.2f}")
        
        st.markdown("---")

def main():
    st.title("⚽ HT Goals AI - Sistema de Alta Precisão")
    st.markdown("🎯 **Objetivo: 80%+ de taxa de acerto com filtros rigorosos**")
    
    # Avisos importantes
    st.warning("""
    ⚠️ **REGRAS PARA MÁXIMA PRECISÃO:**
    - ✅ Só apostas com **75%+ de confiança**
    - ✅ Apenas times com **5+ jogos** de histórico
    - ✅ Máximo **10 apostas/dia** (qualidade > quantidade)
    - ✅ Modelos com F1-Score **>75%**
    """)
    
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
            st.success("✅ Modelos ativos")
            
            # Mostrar apenas ligas com boa performance
            st.markdown("### 🏆 Ligas de Alta Precisão:")
            for league_id, model_data in st.session_state.league_models.items():
                if model_data['avg_cv_score'] >= 0.75:
                    league_name = model_data['league_name']
                    score = model_data['avg_cv_score']
                    st.write(f"• {league_name}: F1 {score:.2f}")
        else:
            st.warning("⚠️ Modelos não treinados")
        
        st.markdown("---")
        
        # Configurações otimizadas
        st.markdown("### ⚙️ Configurações")
        
        data_mode = st.radio(
            "📊 Modo de dados:",
            ["Sazonal Inteligente", "Período Fixo"],
            index=0,
            help="Sazonal: Ajusta automaticamente baseado na época do ano"
        )
        
        if data_mode == "Período Fixo":
            days_training = st.slider(
                "📅 Dias para treinamento:",
                min_value=90,
                max_value=730,
                value=365,
                step=30,
                help="365 dias (1 ano) garante dados em qualquer época"
            )
        else:
            days_training = None  # Modo sazonal calcula automaticamente
            st.info("""
            📅 **Modo Sazonal Ativo:**
            - Agosto-Dezembro: Busca temporada anterior completa
            - Janeiro-Julho: Busca desde agosto anterior
            - Garante dados para início de temporada!
            """)
        
        min_confidence = st.slider(
            "🎯 Confiança mínima:",
            min_value=70,
            max_value=90,
            value=75,
            help="75%+ para alta precisão"
        )
        
        use_cache = st.checkbox("💾 Usar cache", value=True)
    
    # Tabs principais
    tab1, tab2 = st.tabs(["🤖 Treinar Sistema", "🎯 Apostas do Dia"])
    
    with tab1:
        st.header("🤖 Treinamento de Alta Precisão")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **📊 Sistema Ensemble:**
            - RandomForest + GradientBoosting + ExtraTrees
            - Validação temporal (não aleatória)
            - Calibração de probabilidades
            - Features H2H e forma recente
            """)
        
        with col2:
            st.success("""
            **🎯 Filtros de Qualidade:**
            - Mínimo 100 jogos por liga
            - Mínimo 5 jogos por time
            - F1-Score > 75% obrigatório
            - Máximo 10 apostas/dia
            """)
        
        if st.button("🚀 TREINAR SISTEMA DE ALTA PRECISÃO", type="primary", use_container_width=True):
            with st.spinner("📥 Carregando dados..."):
                seasonal = (data_mode == "Sazonal Inteligente")
                df = collect_historical_data_smart(
                    days=days_training, 
                    use_cached=use_cache,
                    seasonal=seasonal
                )
            
            if df.empty:
                st.error("❌ Sem dados suficientes")
                st.stop()
            
            st.success(f"✅ {len(df)} jogos carregados")
            
            # Agrupar por liga
            league_groups = df.groupby(['league_id', 'league_name', 'country'])
            
            league_models = {}
            successful = 0
            
            progress_bar = st.progress(0)
            
            for idx, ((league_id, league_name, country), league_df) in enumerate(league_groups):
                progress = (idx + 1) / len(league_groups)
                progress_bar.progress(progress)
                
                if len(league_df) < 100:
                    continue
                
                # Treinar ensemble
                model_data, message = train_ensemble_model(league_df, league_id, f"{league_name} ({country})")
                
                if model_data and model_data['avg_cv_score'] >= 0.75:
                    league_models[league_id] = model_data
                    successful += 1
                    st.success(f"✅ {league_name}: F1-Score {model_data['avg_cv_score']:.2f}")
            
            progress_bar.empty()
            
            if league_models:
                st.session_state.league_models = league_models
                st.session_state.models_trained = True
                st.success(f"🎉 {successful} ligas com alta precisão!")
                st.balloons()
            else:
                st.error("❌ Nenhum modelo atingiu precisão mínima")
    
    with tab2:
        st.header("🎯 Apostas de Alta Confiança")
        
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
            # Fazer previsões com filtros rigorosos
            predictions = predict_with_confidence_filter(
                fixtures, 
                st.session_state.league_models, 
                min_confidence=min_confidence
            )
            
            if not predictions:
                st.info("🤷 Nenhuma aposta atende aos critérios de alta confiança hoje")
            else:
                st.success(f"🎯 {len(predictions)} apostas de alta confiança encontradas!")
                
                # Estatísticas
                avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
                st.metric("💯 Confiança Média", f"{avg_confidence:.1f}%")
                
                st.markdown("---")
                
                # Mostrar apostas
                for pred in predictions:
                    display_high_confidence_prediction(pred)

if __name__ == "__main__":
    main()
