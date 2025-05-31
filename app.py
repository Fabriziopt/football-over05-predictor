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
    page_title="⚽ Over 0.5 HT ML Predictor",
    page_icon="🤖",
    layout="wide"
)

# Inicializar session state para o modelo
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None

# Configuração da API Key
# Primeiro tenta ler do Streamlit secrets (produção)
# Se não encontrar, usa a key diretamente (desenvolvimento local)
try:
    API_KEY = st.secrets["API_KEY"]
except:
    # ATENÇÃO: Esta é apenas para teste local. Use st.secrets em produção!
    API_KEY = "474f15de5b22951077ccb71b8d75b95c"

# URL base da API-SPORTS
API_BASE_URL = "https://v3.football.api-sports.io"

# Diretório para salvar modelos
MODEL_DIR = "models"
try:
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
except Exception as e:
    st.error(f"Erro ao criar diretório de modelos: {e}")
    MODEL_DIR = "/tmp/models"  # Usar diretório temporário como fallback
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

# CSS Profissional
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .prediction-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .league-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .accuracy-high {
        color: #28a745;
        font-weight: bold;
    }
    .accuracy-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .accuracy-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def get_api_headers():
    """Retorna os headers corretos para API-SPORTS"""
    return {
        'x-apisports-key': API_KEY
    }

def check_api_status():
    """Verifica o status e limites da API"""
    headers = get_api_headers()
    
    try:
        response = requests.get(
            f'{API_BASE_URL}/status',
            headers=headers,
            timeout=10
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
                # Tratamento melhorado de erros
                if isinstance(data['errors'], list) and len(data['errors']) > 0:
                    error_msg = data['errors'][0]
                elif isinstance(data['errors'], dict):
                    error_msg = data['errors'].get('token', str(data['errors']))
                else:
                    error_msg = str(data['errors'])
                return False, 0, error_msg
        else:
            return False, 0, f"Status Code: {response.status_code}"
    except Exception as e:
        return False, 0, str(e)

# Cache para armazenar dados
@st.cache_data(ttl=3600)
def get_fixtures_cached(date_str):
    """Busca jogos com cache de 1 hora"""
    return get_fixtures(date_str)

def get_fixtures(date_str):
    """Busca jogos da API-Football"""
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
            
            # Verificar se há erros na resposta
            if 'errors' in data and data['errors']:
                st.error(f"Erro da API: {data['errors']}")
                return []
            
            fixtures = data.get('response', [])
            return fixtures
        else:
            st.error(f"Erro API: {response.status_code}")
            try:
                error_data = response.json()
                st.error(f"Detalhes: {error_data}")
            except:
                st.error(f"Resposta: {response.text}")
            return []
    except Exception as e:
        st.error(f"Erro de conexão: {str(e)}")
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
                
                # Converter para o formato esperado
                df['over_05'] = (df['ht_home'] + df['ht_away']) > 0
                return df, f"✅ {len(df)} jogos carregados do arquivo local"
            except Exception as e:
                continue
    
    return None, "❌ Nenhum arquivo de dados históricos encontrado"

def collect_historical_data(days=30, use_cached=True):
    """Coleta dados históricos para ML"""
    # Tentar usar dados em cache primeiro
    if use_cached:
        df, message = load_historical_data()
        if df is not None:
            st.info(message)
            # Filtrar apenas os últimos X dias se necessário
            if days < 730:
                cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                df = df[df['date'] >= cutoff_date]
                st.info(f"📊 Filtrado para {len(df)} jogos dos últimos {days} dias")
            return df
    
    # Se não houver cache, buscar da API (método antigo)
    all_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    headers = get_api_headers()
    
    for i in range(days):
        date = datetime.now() - timedelta(days=i+1)
        date_str = date.strftime('%Y-%m-%d')
        status_text.text(f"📊 Coletando dados ML: {date.strftime('%d/%m/%Y')} ({i+1}/{days})")
        
        fixtures = get_fixtures(date_str)
        
        for match in fixtures:
            if match['fixture']['status']['short'] == 'FT' and match.get('score', {}).get('halftime'):
                match_data = extract_match_features(match)
                if match_data:
                    all_data.append(match_data)
        
        progress_bar.progress((i+1)/days)
        time.sleep(0.3)  # Rate limiting
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(all_data)

def extract_match_features(match):
    """Extrai features para ML"""
    try:
        # Dados básicos
        home_team = match['teams']['home']['name']
        away_team = match['teams']['away']['name']
        league_id = match['league']['id']
        league_name = match['league']['name']
        country = match['league']['country']
        
        # Resultado HT
        ht_home = match['score']['halftime']['home']
        ht_away = match['score']['halftime']['away']
        over_05 = 1 if (ht_home + ht_away) > 0 else 0
        
        # Features do jogo
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
    except:
        return None

def prepare_ml_features(df):
    """Prepara features para o modelo ML com métricas avançadas e normalização de outliers"""
    # Garantir que temos as colunas necessárias
    if 'over_05' not in df.columns and 'ht_home' in df.columns and 'ht_away' in df.columns:
        df['over_05'] = (df['ht_home'] + df['ht_away']) > 0
    
    if 'ht_total_goals' not in df.columns and 'ht_home' in df.columns and 'ht_away' in df.columns:
        df['ht_total_goals'] = df['ht_home'] + df['ht_away']
    
    # Mapear nomes de colunas se necessário
    column_mapping = {
        'home_team_id': 'home_team_id',
        'away_team_id': 'away_team_id',
        'ht_home': 'ht_home_goals',
        'ht_away': 'ht_away_goals',
        'league_id': 'league_id'
    }
    
    # Criar colunas compatíveis se necessário
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # Estatísticas por time
    team_stats = {}
    
    # Primeiro, inicializar todos os times
    for idx, row in df.iterrows():
        home_team = row['home_team_id']
        away_team = row['away_team_id']
        
        if home_team not in team_stats:
            team_stats[home_team] = {
                'games': 0, 
                'over_05': 0, 
                'over_05_binary': 0,  # NOVO: conta apenas se teve Over (1) ou não (0)
                'goals_scored': 0, 
                'goals_conceded': 0,
                'goals_capped': 0,  # NOVO: gols com cap de 1
                'home_games': 0, 
                'home_over': 0, 
                'home_over_binary': 0,  # NOVO
                'home_goals': 0,
                'home_goals_capped': 0,  # NOVO
                'away_games': 0, 
                'away_over': 0,
                'away_over_binary': 0,  # NOVO
                'away_goals': 0,
                'away_goals_capped': 0,  # NOVO
                'goals_list': [],
                'over_list': [],
                'extreme_games': 0  # NOVO: conta jogos com mais de 2 gols HT
            }
        
        if away_team not in team_stats:
            team_stats[away_team] = {
                'games': 0, 
                'over_05': 0,
                'over_05_binary': 0,  # NOVO
                'goals_scored': 0, 
                'goals_conceded': 0,
                'goals_capped': 0,  # NOVO
                'home_games': 0, 
                'home_over': 0,
                'home_over_binary': 0,  # NOVO
                'home_goals': 0,
                'home_goals_capped': 0,  # NOVO
                'away_games': 0, 
                'away_over': 0,
                'away_over_binary': 0,  # NOVO
                'away_goals': 0,
                'away_goals_capped': 0,  # NOVO
                'goals_list': [],
                'over_list': [],
                'extreme_games': 0  # NOVO
            }
    
    # Calcular features
    features = []
    
    for idx, row in df.iterrows():
        home_id = row['home_team_id']
        away_id = row['away_team_id']
        
        # Features do time da casa
        home_stats = team_stats[home_id]
        
        # TAXAS BINÁRIAS (o que você sugeriu - conta apenas se foi Over ou não)
        home_over_rate_binary = home_stats['over_05_binary'] / max(home_stats['games'], 1)
        home_home_over_rate_binary = home_stats['home_over_binary'] / max(home_stats['home_games'], 1)
        
        # MÉDIAS COM CAP (limita impacto de jogos extremos)
        home_avg_goals_capped = home_stats['goals_capped'] / max(home_stats['games'], 1)
        home_home_avg_goals_capped = home_stats['home_goals_capped'] / max(home_stats['home_games'], 1)
        
        # Métricas tradicionais (para comparação)
        home_over_rate = home_stats['over_05'] / max(home_stats['games'], 1)
        home_avg_goals = home_stats['goals_scored'] / max(home_stats['games'], 1)
        home_home_over_rate = home_stats['home_over'] / max(home_stats['home_games'], 1)
        
        # Taxa de jogos extremos
        home_extreme_rate = home_stats['extreme_games'] / max(home_stats['games'], 1)
        
        # Features do time visitante
        away_stats = team_stats[away_id]
        
        # TAXAS BINÁRIAS
        away_over_rate_binary = away_stats['over_05_binary'] / max(away_stats['games'], 1)
        away_away_over_rate_binary = away_stats['away_over_binary'] / max(away_stats['away_games'], 1)
        
        # MÉDIAS COM CAP
        away_avg_goals_capped = away_stats['goals_capped'] / max(away_stats['games'], 1)
        away_away_avg_goals_capped = away_stats['away_goals_capped'] / max(away_stats['away_games'], 1)
        
        # Métricas tradicionais
        away_over_rate = away_stats['over_05'] / max(away_stats['games'], 1)
        away_avg_goals = away_stats['goals_scored'] / max(away_stats['games'], 1)
        away_away_over_rate = away_stats['away_over'] / max(away_stats['away_games'], 1)
        
        # Taxa de jogos extremos
        away_extreme_rate = away_stats['extreme_games'] / max(away_stats['games'], 1)
        
        # Features da liga
        league_games = df[df['league_id'] == row['league_id']]
        league_over_rate = league_games['over_05'].mean() if len(league_games) > 0 else 0.5
        league_over_rate_binary = (league_games['over_05'] > 0).mean() if len(league_games) > 0 else 0.5
        
        # FEATURES AVANÇADAS COM NOVA LÓGICA
        
        # 1. Combined Score Binário (usa taxa binária)
        home_strength_binary = home_over_rate_binary * home_avg_goals_capped
        away_strength_binary = away_over_rate_binary * away_avg_goals_capped
        combined_score_binary = home_strength_binary + away_strength_binary
        
        # 2. Eficiência de Over (quantos gols precisa em média para fazer Over)
        home_efficiency = home_avg_goals / home_over_rate_binary if home_over_rate_binary > 0 else 10
        away_efficiency = away_avg_goals / away_over_rate_binary if away_over_rate_binary > 0 else 10
        combined_efficiency = (home_efficiency + away_efficiency) / 2
        
        # 3. Consistência Binária (desvio das taxas de Over)
        if len(home_stats['over_list']) > 1:
            home_consistency = 1 - np.std(home_stats['over_list'])
        else:
            home_consistency = 0.5
            
        if len(away_stats['over_list']) > 1:
            away_consistency = 1 - np.std(away_stats['over_list'])
        else:
            away_consistency = 0.5
        
        # 4. Momentum Binário (últimos 5 jogos - Over ou não)
        home_recent = home_stats['over_list'][-5:] if len(home_stats['over_list']) >= 5 else home_stats['over_list']
        away_recent = away_stats['over_list'][-5:] if len(away_stats['over_list']) >= 5 else away_stats['over_list']
        
        home_momentum = sum([1 if x > 0 else 0 for x in home_recent]) / len(home_recent) if home_recent else home_over_rate_binary
        away_momentum = sum([1 if x > 0 else 0 for x in away_recent]) / len(away_recent) if away_recent else away_over_rate_binary
        
        # 5. Risco de Outlier (baseado em jogos extremos)
        outlier_risk = (home_extreme_rate + away_extreme_rate) / 2
        
        feature_row = {
            # Features binárias (principais)
            'home_over_rate_binary': home_over_rate_binary,
            'home_avg_goals_capped': home_avg_goals_capped,
            'home_home_over_rate_binary': home_home_over_rate_binary,
            'away_over_rate_binary': away_over_rate_binary,
            'away_avg_goals_capped': away_avg_goals_capped,
            'away_away_over_rate_binary': away_away_over_rate_binary,
            'league_over_rate_binary': league_over_rate_binary,
            'combined_over_rate_binary': (home_over_rate_binary + away_over_rate_binary) / 2,
            'combined_goals_capped': home_avg_goals_capped + away_avg_goals_capped,
            
            # Features tradicionais (para comparação)
            'home_over_rate': home_over_rate,
            'away_over_rate': away_over_rate,
            'combined_goals': home_avg_goals + away_avg_goals,
            
            # Features avançadas
            'combined_score_binary': combined_score_binary,
            'combined_efficiency': 1 / combined_efficiency if combined_efficiency > 0 else 0,
            'home_consistency': home_consistency,
            'away_consistency': away_consistency,
            'consistency_avg': (home_consistency + away_consistency) / 2,
            'home_momentum': home_momentum,
            'away_momentum': away_momentum,
            'momentum_sum': home_momentum + away_momentum,
            'outlier_risk': outlier_risk,
            'extreme_game_factor': 1 - outlier_risk,  # Penaliza times com muitos jogos extremos
            
            'target': row['over_05']
        }
        
        features.append(feature_row)
        
        # Atualizar stats após o jogo
        ht_home_goals = row.get('ht_home_goals', row.get('ht_home', 0))
        ht_away_goals = row.get('ht_away_goals', row.get('ht_away', 0))
        ht_total = ht_home_goals + ht_away_goals
        
        # Stats binários/capped
        team_stats[home_id]['games'] += 1
        team_stats[home_id]['over_05'] += row['over_05']
        team_stats[home_id]['over_05_binary'] += 1 if row['over_05'] > 0 else 0  # Conta apenas se foi Over
        team_stats[home_id]['goals_scored'] += ht_home_goals
        team_stats[home_id]['goals_capped'] += min(ht_home_goals, 1)  # Cap em 1 gol
        team_stats[home_id]['goals_conceded'] += ht_away_goals
        team_stats[home_id]['home_games'] += 1
        team_stats[home_id]['home_over'] += row['over_05']
        team_stats[home_id]['home_over_binary'] += 1 if row['over_05'] > 0 else 0
        team_stats[home_id]['home_goals'] += ht_home_goals
        team_stats[home_id]['home_goals_capped'] += min(ht_home_goals, 1)
        team_stats[home_id]['goals_list'].append(ht_home_goals)
        team_stats[home_id]['over_list'].append(row['over_05'])
        if ht_total > 2:  # Jogo extremo
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
        if ht_total > 2:  # Jogo extremo
            team_stats[away_id]['extreme_games'] += 1
    
    return pd.DataFrame(features), team_stats

def train_ml_model(df):
    """Treina o modelo de ML com análise por liga"""
    # Preparar features
    features_df, team_stats = prepare_ml_features(df)
    
    # Adicionar análise por liga
    league_analysis = analyze_leagues(df)
    
    # Separar features e target
    feature_cols = [col for col in features_df.columns if col != 'target']
    X = features_df[feature_cols]
    y = features_df['target']
    
    # Dividir dados: 70% treino, 15% validação, 15% teste
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)
    
    # Normalizar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Treinar múltiplos modelos
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, 
            max_depth=10,
            min_samples_split=5,
            random_state=42, 
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    }
    
    best_model = None
    best_score = 0
    results = {}
    
    # Treinar e validar cada modelo
    st.info("🧠 Treinando modelos de ML...")
    for name, model in models.items():
        # Treinar
        model.fit(X_train_scaled, y_train)
        
        # Validar
        val_pred = model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_pred)
        
        # Testar
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
        
        st.success(f"✅ {name}: F1-Score = {test_f1:.1%}")
        
        if test_f1 > best_score:
            best_score = test_f1
            best_model = model
    
    # Salvar melhor modelo
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'team_stats': team_stats,
        'league_analysis': league_analysis,  # Adicionar análise por liga
        'results': results,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(df),
        'training_days': len(df['date'].unique()) if 'date' in df.columns else 0
    }
    
    # Salvar no session state
    st.session_state.trained_model = model_data
    
    # Tentar salvar em arquivo também
    try:
        for directory in [MODEL_DIR, "/tmp/models"]:
            try:
                if not os.path.exists(directory):
                    os.makedirs(directory)
                model_path = os.path.join(directory, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
                joblib.dump(model_data, model_path)
                break
            except:
                pass
    except:
        pass
    
    return model_data, results

def load_latest_model():
    """Carrega o modelo mais recente"""
    try:
        # Tentar ambos os diretórios
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

def predict_matches(fixtures, model_data):
    """Faz previsões para os jogos do dia com análise por liga"""
    predictions = []
    
    if not model_data:
        return predictions
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = model_data['feature_cols']
    team_stats = model_data['team_stats']
    league_analysis = model_data.get('league_analysis', {})
    
    for fixture in fixtures:
        # Apenas jogos não iniciados
        if fixture['fixture']['status']['short'] not in ['NS', 'TBD']:
            continue
        
        try:
            home_id = fixture['teams']['home']['id']
            away_id = fixture['teams']['away']['id']
            league_name = fixture['league']['name']
            
            # Verificar se temos dados dos times
            if home_id not in team_stats or away_id not in team_stats:
                continue
            
            # Preparar features
            home_stats = team_stats[home_id]
            away_stats = team_stats[away_id]
            
            # Taxa da liga
            league_info = league_analysis.get(league_name, {})
            league_over_rate = league_info.get('over_rate', 0.5)
            
            features = {
                'home_over_rate': home_stats['over_05'] / max(home_stats['games'], 1),
                'home_avg_goals': home_stats['goals_scored'] / max(home_stats['games'], 1),
                'home_home_over_rate': home_stats['home_over'] / max(home_stats['home_games'], 1),
                'away_over_rate': away_stats['over_05'] / max(away_stats['games'], 1),
                'away_avg_goals': away_stats['goals_scored'] / max(away_stats['games'], 1),
                'away_away_over_rate': away_stats['away_over'] / max(away_stats['away_games'], 1),
                'league_over_rate': league_over_rate,
                'combined_over_rate': 0,
                'combined_goals': 0
            }
            
            features['combined_over_rate'] = (features['home_over_rate'] + features['away_over_rate']) / 2
            features['combined_goals'] = features['home_avg_goals'] + features['away_avg_goals']
            
            # Criar DataFrame com features
            X = pd.DataFrame([features])[feature_cols]
            X_scaled = scaler.transform(X)
            
            # Previsão
            pred_proba = model.predict_proba(X_scaled)[0]
            pred_class = model.predict(X_scaled)[0]
            confidence = max(pred_proba) * 100
            
            # Ajuste baseado na tendência da liga
            league_trend = league_info.get('trend', 'BALANCED')
            if league_trend == 'OVER' and pred_class == 1:
                confidence = min(confidence * 1.05, 95)  # Boost de 5% para ligas OVER
            elif league_trend == 'UNDER' and pred_class == 1:
                confidence = confidence * 0.95  # Redução de 5% para ligas UNDER
            
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
                'away_stats': away_stats
            }
            
            predictions.append(prediction)
            
        except:
            continue
    
    # Ordenar por confiança
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return predictions

def analyze_leagues(df):
    """Analisa tendências por liga"""
    league_analysis = {}
    
    for league_id in df['league_id'].unique():
        league_data = df[df['league_id'] == league_id]
        
        if len(league_data) >= 10:  # Mínimo 10 jogos
            over_rate = league_data['over_05'].mean()
            avg_goals = league_data['ht_total_goals'].mean()
            
            # Classificação
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
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Over 0.5 HT - Machine Learning Predictor</h1>
        <p>Sistema inteligente com análise por liga e evolução contínua</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configurações")
        
        # Verificar status da API
        api_ok, requests_left, api_status = check_api_status()
        
        if not api_ok:
            st.error("❌ Problema com a API")
            st.error(f"Erro: {api_status}")
        else:
            st.success(f"✅ API conectada")
            if requests_left > 0:
                st.info(f"📊 Requests restantes hoje: {requests_left}")
            else:
                st.warning(f"⚠️ Sem requests restantes hoje!")
                st.info("💡 A API reseta à meia-noite UTC")
        
        # Data selecionada
        selected_date = st.date_input(
            "📅 Data para análise:",
            value=datetime.now().date()
        )
        
        # Configurações ML
        st.subheader("🤖 Machine Learning")
        
        days_training = st.slider(
            "📊 Dias para treinamento:",
            min_value=15,
            max_value=730,  # Até 2 anos
            value=365  # 1 ano por padrão
        )
        
        use_cache = st.checkbox(
            "💾 Usar dados em cache",
            value=True,
            help="Usar dados históricos salvos localmente (muito mais rápido!)"
        )
        
        # Status do modelo
        model_data = st.session_state.trained_model if 'trained_model' in st.session_state else load_latest_model()
        if model_data:
            st.success("✅ Modelo carregado")
            st.info(f"📅 Treinado em: {model_data['training_date']}")
            st.info(f"📊 Amostras: {model_data['total_samples']}")
            
            # Mostrar melhor performance
            if 'results' in model_data:
                results = model_data['results']
                best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
                st.info(f"🏆 Melhor modelo: {best_model[0]}")
                st.info(f"📈 F1-Score: {best_model[1]['f1_score']:.1%}")
        else:
            st.warning("⚠️ Nenhum modelo encontrado")
    
    # Tabs principais
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Previsões do Dia",
        "📊 Análise por Liga",
        "🤖 Treinar Modelo",
        "📈 Performance ML"
    ])
    
    with tab1:
        st.header(f"🎯 Previsões para {selected_date.strftime('%d/%m/%Y')}")
        
        # Verificar se há modelo disponível - usar session_state primeiro
        if 'trained_model' in st.session_state:
            model_data = st.session_state.trained_model
        else:
            model_data = load_latest_model()
            if model_data:
                st.session_state.trained_model = model_data
        
        if not model_data:
            st.warning("⚠️ Treine um modelo primeiro na aba 'Treinar Modelo'")
        else:
            # Buscar jogos do dia
            date_str = selected_date.strftime('%Y-%m-%d')
            
            with st.spinner("🔍 Buscando jogos do dia..."):
                fixtures = get_fixtures_cached(date_str)
            
            if not fixtures:
                st.info("📅 Nenhum jogo encontrado para esta data")
            else:
                # Fazer previsões
                with st.spinner("🤖 Aplicando Machine Learning..."):
                    predictions = predict_matches(fixtures, model_data)
                
                if predictions:
                    # Métricas resumo
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_games = len(predictions)
                    high_confidence = len([p for p in predictions if p['confidence'] > 70])
                    over_predictions = len([p for p in predictions if p['prediction'] == 'OVER 0.5'])
                    avg_confidence = sum([p['confidence'] for p in predictions]) / len(predictions)
                    
                    with col1:
                        st.markdown("""
                        <div class="metric-card">
                            <h3>🎮 Total de Jogos</h3>
                            <h1>{}</h1>
                        </div>
                        """.format(total_games), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div class="metric-card">
                            <h3>🎯 Alta Confiança</h3>
                            <h1>{}</h1>
                        </div>
                        """.format(high_confidence), unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown("""
                        <div class="metric-card">
                            <h3>📈 Over 0.5</h3>
                            <h1>{}</h1>
                        </div>
                        """.format(over_predictions), unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown("""
                        <div class="metric-card">
                            <h3>💯 Confiança Média</h3>
                            <h1>{:.1f}%</h1>
                        </div>
                        """.format(avg_confidence), unsafe_allow_html=True)
                    
                    # Top previsões - ORDENADAS POR CONFIANÇA (MAIOR PARA MENOR)
                    st.subheader("🏆 Melhores Apostas do Dia")
                    
                    # Filtrar apenas OVER 0.5 com alta confiança e ordenar
                    best_bets = [p for p in predictions if p['prediction'] == 'OVER 0.5' and p['confidence'] > 65]
                    best_bets.sort(key=lambda x: x['confidence'], reverse=True)  # Maior confiança primeiro
                    
                    if best_bets:
                        for i, pred in enumerate(best_bets[:10]):
                            # Converter horário UTC para Portugal (UTC+0 no inverno, UTC+1 no verão)
                            try:
                                # Parse do horário UTC
                                utc_time = datetime.strptime(pred['kickoff'][:16], '%Y-%m-%dT%H:%M')
                                # Adicionar 0 horas (Portugal no inverno) ou 1 hora (verão)
                                # Por simplicidade, vamos usar UTC+0 (você pode ajustar conforme necessário)
                                pt_time = utc_time  # Portugal está em UTC+0 no inverno
                                hora_portugal = pt_time.strftime('%H:%M')
                            except:
                                hora_portugal = pred['kickoff'][11:16]
                            
                            confidence_class = "accuracy-high" if pred['confidence'] > 75 else "accuracy-medium"
                            
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h3>⚽ {pred['home_team']} vs {pred['away_team']}</h3>
                                <p><strong>🏆 Liga:</strong> {pred['league']} ({pred['country']})</p>
                                <p><strong>🕐 Horário PT:</strong> {hora_portugal}</p>
                                <hr style="opacity: 0.3;">
                                <p><strong>🎯 Previsão ML:</strong> {pred['prediction']}</p>
                                <p><strong>💯 Confiança:</strong> <span class="{confidence_class}">{pred['confidence']:.1f}%</span></p>
                                <p><strong>📊 Probabilidades:</strong> Over {pred['probability_over']:.1f}% | Under {pred['probability_under']:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("🤷 Nenhuma aposta OVER 0.5 com boa confiança encontrada hoje")
                    
                    # Todas as previsões
                    with st.expander("📋 Ver Todas as Previsões"):
                        # Filtrar apenas OVER 0.5 para a tabela também
                        over_predictions = [p for p in predictions if p['prediction'] == 'OVER 0.5']
                        
                        pred_data = []
                        for p in over_predictions:
                            try:
                                utc_time = datetime.strptime(p['kickoff'][:16], '%Y-%m-%dT%H:%M')
                                hora_pt = utc_time.strftime('%H:%M')
                            except:
                                hora_pt = p['kickoff'][11:16]
                            
                            pred_data.append({
                                'Hora PT': hora_pt,
                                'Casa': p['home_team'],
                                'Fora': p['away_team'],
                                'Liga': p['league'],
                                'Previsão': p['prediction'],
                                'Confiança': f"{p['confidence']:.1f}%",
                                '_confidence': p['confidence']  # Para ordenação
                            })
                        
                        if pred_data:
                            pred_df = pd.DataFrame(pred_data)
                            # Ordenar por confiança (decrescente) e remover coluna auxiliar
                            pred_df = pred_df.sort_values('_confidence', ascending=False).drop('_confidence', axis=1)
                            st.dataframe(pred_df, use_container_width=True)
                        else:
                            st.info("Nenhuma previsão OVER 0.5 encontrada")
                
                else:
                    st.info("🤷 Nenhuma previsão disponível (times sem dados históricos)")
    
    with tab2:
        st.header("📊 Análise de Ligas")
        
        # Verificar se há modelo carregado
        if 'trained_model' in st.session_state:
            model_data = st.session_state.trained_model
        else:
            model_data = load_latest_model()
        
        if model_data:
            # Usar o use_cache da sidebar
            use_cache = True  # Default
            if 'use_cache' in locals():
                use_cache = use_cache
                
            # Carregar dados históricos para análise
            df = collect_historical_data(days=15, use_cached=use_cache)
            
            if not df.empty:
                league_analysis = analyze_leagues(df)
                
                # Separar por tendência
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
        st.header("🤖 Treinar Modelo ML")
        
        st.info("""
        O modelo será treinado com:
        - **70%** dos dados para treinamento
        - **15%** para validação
        - **15%** para teste final
        """)
        
        # Botão de teste de API
        if st.button("🔌 Testar Conexão API", type="secondary"):
            with st.spinner("Testando conexão..."):
                headers = get_api_headers()
                
                try:
                    response = requests.get(
                        f'{API_BASE_URL}/status',
                        headers=headers,
                        timeout=10
                    )
                    
                    st.write(f"Status Code: {response.status_code}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        if 'response' in data:
                            st.success("✅ API conectada com sucesso!")
                            response_data = data['response']
                            
                            # Mostrar informações da conta
                            if 'account' in response_data:
                                st.info(f"👤 Conta: {response_data['account'].get('firstname', '')} {response_data['account'].get('lastname', '')}")
                            
                            if 'subscription' in response_data:
                                sub = response_data['subscription']
                                st.info(f"📦 Plano: {sub.get('plan', 'Unknown')}")
                                st.info(f"📅 Válido até: {sub.get('end', 'Unknown')}")
                            
                            if 'requests' in response_data:
                                req = response_data['requests']
                                used = req.get('current', 0)
                                limit = req.get('limit_day', 0)
                                remaining = limit - used
                                st.info(f"📊 Requests: {used}/{limit} (Restantes: {remaining})")
                        else:
                            st.error("❌ Resposta inválida da API")
                            st.json(data)
                    else:
                        st.error(f"❌ Erro: Status {response.status_code}")
                        try:
                            st.json(response.json())
                        except:
                            st.text(response.text)
                            
                except Exception as e:
                    st.error(f"❌ Erro de conexão: {str(e)}")
        
        if st.button("🚀 Iniciar Treinamento", type="primary"):
            # Coletar dados
            with st.spinner(f"📊 Coletando {days_training} dias de dados históricos..."):
                df = collect_historical_data(days=days_training, use_cached=use_cache)
            
            if df.empty:
                st.error("❌ Não foi possível coletar dados")
                
                # Teste manual
                st.info("🔍 Testando busca de dados...")
                test_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                test_fixtures = get_fixtures(test_date)
                
            else:
                st.success(f"✅ {len(df)} jogos coletados")
                
                # Treinar modelo
                with st.spinner("🧠 Treinando modelos de Machine Learning..."):
                    model_data, results = train_ml_model(df)
                
                st.success("✅ Modelo treinado com sucesso!")
                
                # Mostrar resultados
                st.subheader("📊 Resultados do Treinamento")
                
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
                
                # Análise de features importantes
                if hasattr(model_data['model'], 'feature_importances_'):
                    st.subheader("🎯 Features Mais Importantes")
                    
                    feature_importance = pd.DataFrame({
                        'feature': model_data['feature_cols'],
                        'importance': model_data['model'].feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    st.bar_chart(feature_importance.set_index('feature')['importance'])
    
    with tab4:
        st.header("📈 Performance do Modelo")
        
        # Verificar se há modelo carregado
        if 'trained_model' in st.session_state:
            model_data = st.session_state.trained_model
        else:
            model_data = load_latest_model()
        
        if model_data and 'results' in model_data:
            results = model_data['results']
            
            # Melhor modelo
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
            
            # Taxa de acerto histórica
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
            
            # Explicação das métricas
            with st.expander("📚 Entenda as Métricas"):
                st.write("""
                - **Acurácia**: Percentual total de acertos do modelo
                - **Precisão**: Quando o modelo prevê OVER 0.5, quantas vezes acerta
                - **Recall**: Dos jogos que foram OVER 0.5, quantos o modelo identificou
                - **F1-Score**: Média harmônica entre Precisão e Recall (métrica principal)
                - **Taxa de Acerto**: Percentual de previsões corretas no conjunto de teste
                """)
            
            # Informações do modelo
            st.subheader("ℹ️ Informações do Modelo")
            st.info(f"""
            - **Data de Treinamento**: {model_data['training_date']}
            - **Total de Jogos Analisados**: {model_data['total_samples']:,}
            - **Times no Banco de Dados**: {len(model_data['team_stats']):,}
            - **Algoritmo**: {best_model_name}
            """)
        else:
            st.info("🤖 Nenhum modelo treinado ainda")

if __name__ == "__main__":
    main()
