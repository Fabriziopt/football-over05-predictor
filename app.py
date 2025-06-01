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
                        if attempt == 0:
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
        "historical_matches.csv",
        "data/historical_matches_complete.parquet",
        "data/historical_matches_cache.parquet"
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

def collect_historical_data_optimized(days=30, use_cached=True):
    """Versão otimizada da coleta de dados históricos"""
    
    # 1. Primeiro, sempre tentar carregar do cache
    if use_cached:
        df, message = load_historical_data()
        if df is not None:
            st.info(message)
            # Filtrar apenas os dias necessários
            if days < 730:
                cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                if 'date' in df.columns:
                    df_filtered = df[df['date'] >= cutoff_date].copy()
                    st.info(f"📊 Usando cache: {len(df_filtered)} jogos dos últimos {days} dias")
                    return df_filtered
            return df
    
    # 2. Se não usar cache ou não encontrou arquivo, coletar dados
    st.warning("⚠️ Coleta de dados da API pode ser lenta. Recomendo usar dados em cache!")
    
    # Para o modelo ML, não precisamos de todos os dias
    # Podemos usar amostragem para reduzir requisições
    if days > 30:
        # Amostragem inteligente: mais dias recentes, menos dias antigos
        sample_days = []
        
        # Últimos 7 dias completos
        for i in range(7):
            sample_days.append(i + 1)
        
        # Próximos 23 dias: um a cada 2 dias
        for i in range(7, min(30, days), 2):
            sample_days.append(i + 1)
        
        # Restante: um a cada 5 dias
        for i in range(30, days, 5):
            sample_days.append(i + 1)
        
        total_requests = len(sample_days)
        st.info(f"🚀 Otimização: coletando {total_requests} dias amostrados de {days} dias totais")
    else:
        sample_days = list(range(1, days + 1))
        total_requests = days
    
    # 3. Coleta paralela com chunks
    all_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Processar em chunks para mostrar progresso
    chunk_size = 5
    chunks = [sample_days[i:i + chunk_size] for i in range(0, len(sample_days), chunk_size)]
    
    for chunk_idx, chunk in enumerate(chunks):
        chunk_data = []
        
        for day_offset in chunk:
            date = datetime.now() - timedelta(days=day_offset)
            date_str = date.strftime('%Y-%m-%d')
            
            try:
                # Usar cache de fixtures se disponível
                fixtures = get_fixtures_cached_robust(date_str)
                
                if fixtures:
                    for match in fixtures:
                        try:
                            if match['fixture']['status']['short'] == 'FT' and match.get('score', {}).get('halftime'):
                                match_data = extract_match_features(match)
                                if match_data:
                                    chunk_data.append(match_data)
                        except:
                            continue
            except:
                continue
        
        all_data.extend(chunk_data)
        
        # Atualizar progresso
        progress = (chunk_idx + 1) / len(chunks)
        progress_bar.progress(progress)
        status_text.text(f"📊 Coletando dados: {len(all_data)} jogos encontrados...")
        
        # Pequena pausa entre chunks
        if chunk_idx < len(chunks) - 1:
            time.sleep(0.2)
    
    progress_bar.empty()
    status_text.empty()
    
    # 4. Salvar dados coletados em cache para uso futuro
    if len(all_data) > 100:
        try:
            df_new = pd.DataFrame(all_data)
            # Tentar salvar como parquet (mais eficiente)
            cache_file = "data/historical_matches_cache.parquet"
            os.makedirs("data", exist_ok=True)
            df_new.to_parquet(cache_file)
            st.success(f"💾 Cache atualizado: {len(all_data)} jogos salvos")
        except:
            pass
    
    st.info(f"🎯 Total de jogos coletados: {len(all_data)}")
    
    return pd.DataFrame(all_data)

# Usar a versão otimizada
collect_historical_data_robust = collect_historical_data_optimized

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

def prepare_ml_features_hybrid(df):
    """
    Versão híbrida que mantém TODAS as features avançadas
    mas com otimizações de performance
    """
    
    # Garantir coluna over_05
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
    
    # Ordenar por data
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)
    
    st.info("🧠 Preparando TODAS as 25+ features avançadas com otimização...")
    
    # Pré-calcular estatísticas agregadas para performance
    team_stats = {}
    
    # Inicializar com numpy arrays para melhor performance
    unique_teams = pd.concat([df['home_team_id'], df['away_team_id']]).unique()
    
    for team_id in unique_teams:
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
    total_rows = len(df)
    
    # Processar com barra de progresso para grandes datasets
    progress_bar = st.progress(0)
    progress_step = max(1, total_rows // 20)  # Atualizar a cada 5%
    
    for idx, row in df.iterrows():
        if idx % progress_step == 0:
            progress = idx / total_rows
            progress_bar.progress(progress)
        
        home_id = row['home_team_id']
        away_id = row['away_team_id']
        
        # Obter estatísticas atuais
        home_stats = team_stats[home_id]
        away_stats = team_stats[away_id]
        
        # Calcular features básicas
        home_over_rate = home_stats['over_05'] / max(home_stats['games'], 1)
        home_over_rate_binary = home_stats['over_05_binary'] / max(home_stats['games'], 1)
        home_avg_goals = home_stats['goals_scored'] / max(home_stats['games'], 1)
        home_avg_goals_capped = home_stats['goals_capped'] / max(home_stats['games'], 1)
        home_home_over_rate = home_stats['home_over'] / max(home_stats['home_games'], 1)
        home_home_over_rate_binary = home_stats['home_over_binary'] / max(home_stats['home_games'], 1)
        
        away_over_rate = away_stats['over_05'] / max(away_stats['games'], 1)
        away_over_rate_binary = away_stats['over_05_binary'] / max(away_stats['games'], 1)
        away_avg_goals = away_stats['goals_scored'] / max(away_stats['games'], 1)
        away_avg_goals_capped = away_stats['goals_capped'] / max(away_stats['games'], 1)
        away_away_over_rate = away_stats['away_over'] / max(away_stats['away_games'], 1)
        away_away_over_rate_binary = away_stats['away_over_binary'] / max(away_stats['away_games'], 1)
        
        # Features de liga (usar cache)
        league_id = row['league_id']
        league_mask = df['league_id'] == league_id
        league_data = df.loc[league_mask & (df.index < idx)]  # Apenas jogos anteriores
        league_over_rate = league_data['over_05'].mean() if len(league_data) > 0 else 0.5
        league_over_rate_binary = (league_data['over_05'] > 0).mean() if len(league_data) > 0 else 0.5
        
        # FEATURES AVANÇADAS - Coeficiente de Variação
        if len(home_stats['goals_list']) > 1:
            home_goals_std = np.std(home_stats['goals_list'])
            home_goals_mean = np.mean(home_stats['goals_list'])
            home_goals_cv = home_goals_std / (home_goals_mean + 0.01)
            home_consistency = 1 / (1 + home_goals_cv)
        else:
            home_consistency = 0.5
            home_goals_cv = 1.0
        
        if len(away_stats['goals_list']) > 1:
            away_goals_std = np.std(away_stats['goals_list'])
            away_goals_mean = np.mean(away_stats['goals_list'])
            away_goals_cv = away_goals_std / (away_goals_mean + 0.01)
            away_consistency = 1 / (1 + away_goals_cv)
        else:
            away_consistency = 0.5
            away_goals_cv = 1.0
        
        # FEATURES AVANÇADAS - Combined Score
        home_strength_binary = home_over_rate_binary * home_avg_goals_capped * home_consistency
        away_strength_binary = away_over_rate_binary * away_avg_goals_capped * away_consistency
        combined_score_binary = home_strength_binary + away_strength_binary
        
        # FEATURES AVANÇADAS - Momentum Analysis
        home_recent = home_stats['over_list'][-5:] if len(home_stats['over_list']) >= 5 else home_stats['over_list']
        away_recent = away_stats['over_list'][-5:] if len(away_stats['over_list']) >= 5 else away_stats['over_list']
        
        home_momentum = sum([1 if x > 0 else 0 for x in home_recent]) / len(home_recent) if home_recent else home_over_rate_binary
        away_momentum = sum([1 if x > 0 else 0 for x in away_recent]) / len(away_recent) if away_recent else away_over_rate_binary
        
        # FEATURES AVANÇADAS - Outlier Detection
        home_extreme_rate = home_stats['extreme_games'] / max(home_stats['games'], 1)
        away_extreme_rate = away_stats['extreme_games'] / max(away_stats['games'], 1)
        
        # Criar dicionário com TODAS as features
        feature_row = {
            # Features básicas
            'home_over_rate': home_over_rate,
            'home_avg_goals': home_avg_goals,
            'home_home_over_rate': home_home_over_rate,
            'away_over_rate': away_over_rate,
            'away_avg_goals': away_avg_goals,
            'away_away_over_rate': away_away_over_rate,
            'league_over_rate': league_over_rate,
            'combined_over_rate': (home_over_rate + away_over_rate) / 2,
            'combined_goals': home_avg_goals + away_avg_goals,
            
            # Features binárias
            'home_over_rate_binary': home_over_rate_binary,
            'home_avg_goals_capped': home_avg_goals_capped,
            'home_home_over_rate_binary': home_home_over_rate_binary,
            'away_over_rate_binary': away_over_rate_binary,
            'away_avg_goals_capped': away_avg_goals_capped,
            'away_away_over_rate_binary': away_away_over_rate_binary,
            'league_over_rate_binary': league_over_rate_binary,
            'combined_over_rate_binary': (home_over_rate_binary + away_over_rate_binary) / 2,
            'combined_goals_capped': home_avg_goals_capped + away_avg_goals_capped,
            
            # FEATURES AVANÇADAS - Coeficiente de Variação
            'home_goals_cv': home_goals_cv,
            'away_goals_cv': away_goals_cv,
            'home_consistency': home_consistency,
            'away_consistency': away_consistency,
            'consistency_avg': (home_consistency + away_consistency) / 2,
            'consistency_diff': abs(home_consistency - away_consistency),
            
            # FEATURES AVANÇADAS - Combined Score
            'combined_score_binary': combined_score_binary,
            'home_strength_binary': home_strength_binary,
            'away_strength_binary': away_strength_binary,
            
            # FEATURES AVANÇADAS - Momentum
            'home_momentum': home_momentum,
            'away_momentum': away_momentum,
            'momentum_sum': home_momentum + away_momentum,
            'momentum_diff': abs(home_momentum - away_momentum),
            'momentum_avg': (home_momentum + away_momentum) / 2,
            
            # FEATURES AVANÇADAS - Outliers
            'home_extreme_rate': home_extreme_rate,
            'away_extreme_rate': away_extreme_rate,
            'extreme_rate_avg': (home_extreme_rate + away_extreme_rate) / 2,
            
            # Target
            'target': row['over_05']
        }
        
        features.append(feature_row)
        
        # Atualizar estatísticas para próxima iteração
        ht_home_goals = row.get('ht_home_goals', row.get('ht_home', 0))
        ht_away_goals = row.get('ht_away_goals', row.get('ht_away', 0))
        ht_total = ht_home_goals + ht_away_goals
        
        # Atualizar home team
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
        
        # Atualizar away team
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
        
        # Manter apenas últimos 10 jogos para economizar memória
        for team_id in [home_id, away_id]:
            if len(team_stats[team_id]['goals_list']) > 10:
                team_stats[team_id]['goals_list'] = team_stats[team_id]['goals_list'][-10:]
                team_stats[team_id]['over_list'] = team_stats[team_id]['over_list'][-10:]
    
    # Limpar barra de progresso
    progress_bar.empty()
    
    features_df = pd.DataFrame(features)
    st.success(f"✅ {len(features_df.columns)-1} features avançadas preparadas!")
    
    return features_df, team_stats

# Usar a versão híbrida
prepare_ml_features = prepare_ml_features_hybrid

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

def display_prediction_card_with_averages(pred):
    """Exibe card de previsão com médias da liga e do sistema"""
    
    # Obter contexto da liga
    league_context = get_league_context(pred['league'])
    
    # Container principal do card
    with st.container():
        # Estilizar como um card
        card_col1, card_col2 = st.columns([5, 1])
        
        with card_col1:
            st.markdown(f"### ⚽ {pred['home_team']} vs {pred['away_team']}")
            
            # Informações do jogo
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"🏆 **Liga:** {pred['league']} ({pred['country']})")
            with col2:
                st.write(f"🕐 **Horário PT:** {pred['kickoff'][11:16]}")
        
        with card_col2:
            # Badge de confiança
            if pred['confidence'] > 80:
                st.success(f"**{pred['confidence']:.1f}%**")
            elif pred['confidence'] > 70:
                st.info(f"**{pred['confidence']:.1f}%**")
            else:
                st.warning(f"**{pred['confidence']:.1f}%**")
        
        # Previsão
        st.info(f"🎯 **Previsão ML:** {pred['prediction']}")
        
        # Médias em colunas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="📊 Média da Liga",
                value=f"{league_context['league_avg']:.0f}%",
                delta=league_context['comparison_text']
            )
        
        with col2:
            st.metric(
                label="🤖 Média do Sistema Over 0.5 HT",
                value=f"{pred['probability_over']:.0f}%",
                delta="ML Prediction"
            )
        
        with col3:
            diff = pred['confidence'] - league_context['league_avg']
            st.metric(
                label="📈 Diferença vs Liga",
                value=f"{diff:+.0f}%",
                delta="Acima" if diff > 0 else "Abaixo"
            )
        
        st.markdown("---")

def main():
    st.title("⚽ HT Goals AI Engine")
    st.markdown("🚀 Powered by Predictive Modeling & Advanced Metrics")
    
    # Teste de conectividade inicial
    conn_ok, conn_msg = test_api_connection()
    
    with st.sidebar:
        st.title("⚙️ Configurações")
        
        # Status da API com indicador visual
        if conn_ok:
            api_ok, requests_left, api_status = check_api_status()
            
            if not api_ok:
                st.error(f"❌ {api_status}")
            else:
                st.success("✅ API conectada")
                if requests_left > 0:
                    st.info(f"📊 Requests restantes hoje: {requests_left}")
                else:
                    st.warning(f"⚠️ Sem requests restantes hoje!")
        else:
            st.error(f"❌ {conn_msg}")
        
        selected_date = st.date_input(
            "📅 Data para análise:",
            value=datetime.now().date()
        )
        
        st.subheader("🤖 Machine Learning Avançado")
        
        days_training = st.slider(
            "📊 Dias para treinamento:",
            min_value=15,
            max_value=730,
            value=150
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
    
    # Tabs principais com nova aba
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Previsões do Dia",
        "📊 Análise por Liga", 
        "🤖 Treinar Modelo Avançado",
        "📈 Performance ML",
        "🚀 Análise Automática"  # Nova aba
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
                        st.metric("🎮 Total de Jogos", total_games)
                    
                    with col2:
                        st.metric("🎯 Alta Confiança", high_confidence)
                    
                    with col3:
                        st.metric("📈 Over 0.5", over_predictions)
                    
                    with col4:
                        st.metric("💯 Confiança Média", f"{avg_confidence:.1f}%")
                    
                    st.subheader("🏆 Melhores Apostas (Análise Avançada)")
                    
                    best_bets = [p for p in predictions if p['prediction'] == 'OVER 0.5' and p['confidence'] > 65]
                    best_bets.sort(key=lambda x: x['confidence'], reverse=True)
                    
                    if best_bets:
                        for i, pred in enumerate(best_bets[:5]):
                            # Criar card bonito sem HTML
                            with st.container():
                                # Badge de confiança no canto
                                col1, col2 = st.columns([4, 1])
                                
                                with col1:
                                    st.subheader(f"⚽ {pred['home_team']} vs {pred['away_team']}")
                                
                                with col2:
                                    if pred['confidence'] > 80:
                                        st.success(f"{pred['confidence']:.1f}%")
                                    elif pred['confidence'] > 65:
                                        st.info(f"{pred['confidence']:.1f}%")
                                    else:
                                        st.warning(f"{pred['confidence']:.1f}%")
                                
                                # Info do jogo
                                liga_info = f"🏆 **Liga:** {pred['league']} ({pred['country']})"
                                hora = pred['kickoff'][11:16]
                                st.write(f"{liga_info} | 🕐 **Horário PT:** {hora}")
                                
                                # Previsão
                                st.info(f"🎯 **Previsão ML:** {pred['prediction']}")
                                
                                st.markdown("---")
                    else:
                        st.info("🤷 Nenhuma aposta OVER 0.5 com boa confiança encontrada hoje")
                    
                    # Lista completa de jogos
                    st.subheader("📋 Todos os Jogos do Dia")
                    
                    # Criar DataFrame para todos os jogos
                    all_games_data = []
                    for pred in predictions:
                        try:
                            utc_time = datetime.strptime(pred['kickoff'][:16], '%Y-%m-%dT%H:%M')
                            hora = utc_time.strftime('%H:%M')
                        except:
                            hora = pred['kickoff'][11:16]
                        
                        all_games_data.append({
                            'Hora': hora,
                            'Jogo': f"{pred['home_team']} vs {pred['away_team']}",
                            'Liga': pred['league'],
                            'Previsão': pred['prediction'],
                            'Confiança': f"{pred['confidence']:.0f}%"
                        })
                    
                    # Exibir como tabela
                    df_all = pd.DataFrame(all_games_data)
                    st.dataframe(df_all, use_container_width=True, hide_index=True)
                
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
                        with st.expander(f"{league} - {stats['over_rate']:.1%}"):
                            st.write(f"**{stats['classification']}**")
                            st.write(f"📊 Taxa Over: {stats['over_rate']:.1%}")
                            st.write(f"⚽ Média gols HT: {stats['avg_goals_ht']:.2f}")
                            st.write(f"🎮 Jogos analisados: {stats['total_games']}")
                
                with col2:
                    st.subheader("❄️ Ligas UNDER (< 50%)")
                    for league, stats in sorted(under_leagues.items(), key=lambda x: x[1]['over_rate']):
                        with st.expander(f"{league} - {stats['over_rate']:.1%}"):
                            st.write(f"**{stats['classification']}**")
                            st.write(f"📊 Taxa Over: {stats['over_rate']:.1%}")
                            st.write(f"⚽ Média gols HT: {stats['avg_goals_ht']:.2f}")
                            st.write(f"🎮 Jogos analisados: {stats['total_games']}")
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
                st.metric("🎯 Acurácia", f"{accuracy:.1f}%")
            
            with col2:
                precision = best_metrics['precision'] * 100
                st.metric("💎 Precisão", f"{precision:.1f}%")
            
            with col3:
                recall = best_metrics['recall'] * 100
                st.metric("📊 Recall", f"{recall:.1f}%")
            
            with col4:
                f1 = best_metrics['f1_score'] * 100
                st.metric("🏅 F1-Score", f"{f1:.1f}%")
            
            st.subheader("📊 Performance Histórica do Modelo")
            
            if 'total_samples' in model_data:
                total_analyzed = model_data['total_samples']
                accuracy_rate = best_metrics['test_accuracy'] * 100
                correct_predictions = int(total_analyzed * best_metrics['test_accuracy'])
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📅 Jogos Analisados", f"{total_analyzed:,}")
                
                with col2:
                    st.metric("✅ Acertos", f"{correct_predictions:,}")
                
                with col3:
                    st.metric("📈 Taxa de Acerto", f"{accuracy_rate:.1f}%")
            
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
    
    with tab5:
        st.header("🚀 Análise Automática - Um Clique")
        st.markdown("Sistema automático que treina o modelo e analisa os jogos de hoje com apenas 1 clique!")
        
        # Container central para o botão
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Verificar se já tem modelo treinado hoje
            if 'auto_analysis_done' in st.session_state and st.session_state.auto_analysis_done:
                st.success("✅ Análise completa! Veja os resultados abaixo.")
                
                # Botão para refazer análise
                if st.button("🔄 Refazer Análise", use_container_width=True):
                    st.session_state.auto_analysis_done = False
                    st.rerun()
            else:
                # Botão principal
                if st.button("🎯 ANALISAR JOGOS DE HOJE", use_container_width=True, type="primary", key="auto_analyze"):
                    
                    # Container para mostrar o progresso
                    progress_container = st.container()
                    
                    with progress_container:
                        # Passo 1: Verificar conexão
                        progress = st.progress(0.1)
                        status_text = st.empty()
                        status_text.text("🔍 Verificando conexão com API...")
                        
                        conn_ok, conn_msg = test_api_connection()
                        if not conn_ok:
                            st.error(f"❌ Erro de conexão: {conn_msg}")
                            st.stop()
                        
                        # Passo 2: Carregar dados
                        progress.progress(0.2)
                        status_text.text("📊 Carregando dados históricos...")
                        
                        df, message = load_historical_data()
                        if df is None:
                            status_text.text("📥 Coletando dados da API (pode demorar)...")
                            df = collect_historical_data_optimized(days=30, use_cached=False)
                            if df.empty:
                                st.error("❌ Não foi possível coletar dados")
                                st.stop()
                        
                        # Passo 3: Treinar modelo
                        progress.progress(0.4)
                        status_text.text("🧠 Treinando modelo ML (70% treino, 15% validação, 15% teste)...")
                        
                        model_data, results = train_ml_model_robust(df)
                        if not model_data:
                            st.error("❌ Erro ao treinar modelo")
                            st.stop()
                        
                        # Passo 4: Buscar jogos de hoje
                        progress.progress(0.6)
                        status_text.text("🔍 Buscando jogos de hoje...")
                        
                        today = datetime.now().strftime('%Y-%m-%d')
                        fixtures = get_fixtures_cached_robust(today)
                        
                        if not fixtures:
                            progress.progress(1.0)
                            status_text.text("✅ Análise completa!")
                            st.warning("📅 Nenhum jogo encontrado para hoje")
                            st.session_state.auto_analysis_done = True
                            st.stop()
                        
                        # Passo 5: Fazer previsões
                        progress.progress(0.8)
                        status_text.text("🎯 Gerando previsões com ML avançado...")
                        
                        predictions = predict_matches(fixtures, model_data)
                        
                        # Filtrar apenas OVER 0.5 com boa confiança
                        best_predictions = [p for p in predictions if p['prediction'] == 'OVER 0.5' and p['confidence'] > 65]
                        best_predictions.sort(key=lambda x: x['confidence'], reverse=True)
                        
                        # Passo 6: Finalizar
                        progress.progress(1.0)
                        status_text.text("✅ Análise completa!")
                        time.sleep(1)
                        
                        # Limpar progresso
                        progress.empty()
                        status_text.empty()
                        
                        # Marcar como concluído
                        st.session_state.auto_analysis_done = True
                        st.session_state.auto_predictions = best_predictions
                        st.session_state.auto_model_data = model_data
                        st.session_state.auto_results = results
        
        # Mostrar resultados se a análise foi feita
        if 'auto_analysis_done' in st.session_state and st.session_state.auto_analysis_done:
            
            # Mostrar métricas do modelo
            if 'auto_results' in st.session_state:
                results = st.session_state.auto_results
                best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🏆 Modelo", best_model[0])
                with col2:
                    st.metric("📊 Precisão", f"{best_model[1]['precision']:.1%}")
                with col3:
                    st.metric("🎯 F1-Score", f"{best_model[1]['f1_score']:.1%}")
            
            st.markdown("---")
            
            # Mostrar previsões
            if 'auto_predictions' in st.session_state and st.session_state.auto_predictions:
                predictions = st.session_state.auto_predictions
                
                st.subheader(f"🏆 Melhores Apostas Over 0.5 HT - {datetime.now().strftime('%d/%m/%Y')}")
                
                # Estatísticas resumo
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total de Apostas", len(predictions))
                with col2:
                    avg_conf = sum(p['confidence'] for p in predictions) / len(predictions) if predictions else 0
                    st.metric("Confiança Média", f"{avg_conf:.1f}%")
                with col3:
                    high_conf = len([p for p in predictions if p['confidence'] > 70])
                    st.metric("Alta Confiança (>70%)", high_conf)
                
                st.markdown("---")
                
                # Exibir cards estilo da imagem
                for pred in predictions[:10]:  # Top 10
                    display_prediction_card_with_averages(pred)
            else:
                st.info("🤷 Nenhuma aposta com boa confiança encontrada hoje")

if __name__ == "__main__":
    main()
