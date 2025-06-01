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

# Inicializar session state
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
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
except Exception as e:
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
    except requests.exceptions.Timeout:
        return False, "Timeout - conexão lenta"
    except requests.exceptions.ConnectionError:
        return False, "Erro de conexão - verifique internet"
    except Exception as e:
        return False, f"Erro: {str(e)}"

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
                    if 'errors' in data and data['errors']:
                        return []
                    fixtures = data.get('response', [])
                    return fixtures
                except Exception:
                    return []
            else:
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
def get_fixtures_cached(date_str):
    """Busca jogos com cache"""
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
                return df, f"✅ {len(df)} jogos carregados do cache local"
            except Exception:
                continue
    
    return None, "❌ Nenhum arquivo de dados históricos encontrado"

def collect_historical_data_smart(days=60, use_cached=True):
    """Coleta inteligente de dados históricos"""
    
    # 1. Tentar carregar do cache primeiro
    if use_cached:
        df, message = load_historical_data()
        if df is not None:
            # Filtrar apenas os dias necessários
            if days < 730:
                cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                if 'date' in df.columns:
                    df_filtered = df[df['date'] >= cutoff_date].copy()
                    return df_filtered
            return df
    
    # 2. Se não encontrou cache, coletar da API
    st.warning("⚠️ Coletando dados da API - pode ser lento...")
    
    # Amostragem inteligente para reduzir requests
    sample_days = []
    # Últimos 15 dias completos
    for i in range(15):
        sample_days.append(i + 1)
    # Restante: amostragem
    for i in range(15, days, 3):
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
                    try:
                        if match['fixture']['status']['short'] == 'FT' and match.get('score', {}).get('halftime'):
                            match_data = extract_match_features(match)
                            if match_data:
                                all_data.append(match_data)
                    except:
                        continue
        except:
            continue
        
        # Atualizar progresso
        progress = (idx + 1) / len(sample_days)
        progress_bar.progress(progress)
        
        if idx % 5 == 0:  # Pausa a cada 5 requests
            time.sleep(0.3)
    
    progress_bar.empty()
    
    # Salvar em cache
    if len(all_data) > 50:
        try:
            df_new = pd.DataFrame(all_data)
            cache_file = "data/historical_matches_cache.parquet"
            os.makedirs("data", exist_ok=True)
            df_new.to_parquet(cache_file)
        except:
            pass
    
    return pd.DataFrame(all_data)

def extract_match_features(match):
    """Extrai features básicas do jogo"""
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

def prepare_advanced_features(df):
    """Prepara features avançadas para ML"""
    
    # Garantir coluna over_05
    if 'over_05' not in df.columns:
        if 'ht_home_goals' in df.columns and 'ht_away_goals' in df.columns:
            df['over_05'] = (df['ht_home_goals'] + df['ht_away_goals']) > 0
        elif 'ht_home' in df.columns and 'ht_away' in df.columns:
            df['over_05'] = (df['ht_home'] + df['ht_away']) > 0
    
    if 'ht_total_goals' not in df.columns:
        if 'ht_home_goals' in df.columns and 'ht_away_goals' in df.columns:
            df['ht_total_goals'] = df['ht_home_goals'] + df['ht_away_goals']
    
    # Ordenar por data
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)
    
    # Inicializar estatísticas dos times
    team_stats = {}
    unique_teams = pd.concat([df['home_team_id'], df['away_team_id']]).unique()
    
    for team_id in unique_teams:
        team_stats[team_id] = {
            'games': 0, 'over_05': 0, 'goals_scored': 0, 'goals_conceded': 0,
            'home_games': 0, 'home_over': 0, 'away_games': 0, 'away_over': 0,
            'goals_list': [], 'over_list': []
        }
    
    features = []
    
    for idx, row in df.iterrows():
        home_id = row['home_team_id']
        away_id = row['away_team_id']
        
        # Obter estatísticas atuais
        home_stats = team_stats[home_id]
        away_stats = team_stats[away_id]
        
        # Calcular rates básicas
        home_over_rate = home_stats['over_05'] / max(home_stats['games'], 1)
        home_avg_goals = home_stats['goals_scored'] / max(home_stats['games'], 1)
        home_home_over_rate = home_stats['home_over'] / max(home_stats['home_games'], 1)
        
        away_over_rate = away_stats['over_05'] / max(away_stats['games'], 1)
        away_avg_goals = away_stats['goals_scored'] / max(away_stats['games'], 1)
        away_away_over_rate = away_stats['away_over'] / max(away_stats['away_games'], 1)
        
        # Features de liga
        league_id = row['league_id']
        league_mask = df['league_id'] == league_id
        league_data = df.loc[league_mask & (df.index < idx)]
        league_over_rate = league_data['over_05'].mean() if len(league_data) > 0 else 0.5
        
        # Features avançadas - Consistência
        if len(home_stats['goals_list']) > 1:
            home_goals_std = np.std(home_stats['goals_list'])
            home_goals_mean = np.mean(home_stats['goals_list'])
            home_consistency = 1 / (1 + home_goals_std / (home_goals_mean + 0.01))
        else:
            home_consistency = 0.5
        
        if len(away_stats['goals_list']) > 1:
            away_goals_std = np.std(away_stats['goals_list'])
            away_goals_mean = np.mean(away_stats['goals_list'])
            away_consistency = 1 / (1 + away_goals_std / (away_goals_mean + 0.01))
        else:
            away_consistency = 0.5
        
        # Features avançadas - Momentum
        home_recent = home_stats['over_list'][-5:] if len(home_stats['over_list']) >= 5 else home_stats['over_list']
        away_recent = away_stats['over_list'][-5:] if len(away_stats['over_list']) >= 5 else away_stats['over_list']
        
        home_momentum = sum(home_recent) / len(home_recent) if home_recent else home_over_rate
        away_momentum = sum(away_recent) / len(away_recent) if away_recent else away_over_rate
        
        # Criar features
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
            
            # Features avançadas
            'home_consistency': home_consistency,
            'away_consistency': away_consistency,
            'consistency_avg': (home_consistency + away_consistency) / 2,
            'consistency_diff': abs(home_consistency - away_consistency),
            'home_momentum': home_momentum,
            'away_momentum': away_momentum,
            'momentum_avg': (home_momentum + away_momentum) / 2,
            'momentum_diff': abs(home_momentum - away_momentum),
            'combined_strength': (home_over_rate * home_consistency + away_over_rate * away_consistency) / 2,
            
            # Target
            'target': row['over_05']
        }
        
        features.append(feature_row)
        
        # Atualizar estatísticas
        ht_home_goals = row.get('ht_home_goals', row.get('ht_home', 0))
        ht_away_goals = row.get('ht_away_goals', row.get('ht_away', 0))
        
        # Home team
        team_stats[home_id]['games'] += 1
        team_stats[home_id]['over_05'] += row['over_05']
        team_stats[home_id]['goals_scored'] += ht_home_goals
        team_stats[home_id]['goals_conceded'] += ht_away_goals
        team_stats[home_id]['home_games'] += 1
        team_stats[home_id]['home_over'] += row['over_05']
        team_stats[home_id]['goals_list'].append(ht_home_goals)
        team_stats[home_id]['over_list'].append(row['over_05'])
        
        # Away team
        team_stats[away_id]['games'] += 1
        team_stats[away_id]['over_05'] += row['over_05']
        team_stats[away_id]['goals_scored'] += ht_away_goals
        team_stats[away_id]['goals_conceded'] += ht_home_goals
        team_stats[away_id]['away_games'] += 1
        team_stats[away_id]['away_over'] += row['over_05']
        team_stats[away_id]['goals_list'].append(ht_away_goals)
        team_stats[away_id]['over_list'].append(row['over_05'])
        
        # Manter apenas últimos 10 jogos
        for team_id in [home_id, away_id]:
            if len(team_stats[team_id]['goals_list']) > 10:
                team_stats[team_id]['goals_list'] = team_stats[team_id]['goals_list'][-10:]
                team_stats[team_id]['over_list'] = team_stats[team_id]['over_list'][-10:]
    
    return pd.DataFrame(features), team_stats

def train_complete_model(df, progress_callback=None):
    """Treina modelo completo com todos os algoritmos"""
    
    if len(df) < 100:
        return None, "❌ Dados insuficientes (mínimo 100 jogos)"
    
    try:
        # Preparar features
        if progress_callback:
            progress_callback(0.1, "🧠 Preparando features avançadas...")
        
        features_df, team_stats = prepare_advanced_features(df)
        
        feature_cols = [col for col in features_df.columns if col != 'target']
        X = features_df[feature_cols]
        y = features_df['target']
        
        # Split dos dados
        if progress_callback:
            progress_callback(0.2, "📊 Dividindo dados (70% treino, 15% validação, 15% teste)...")
        
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)
        
        # Escalar dados
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Modelos para testar
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=5,
                min_samples_leaf=3, random_state=42, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=6,
                min_samples_split=5, random_state=42
            )
        }
        
        best_model = None
        best_score = 0
        results = {}
        
        # Treinar modelos
        for i, (name, model) in enumerate(models.items()):
            if progress_callback:
                progress_callback(0.3 + (i * 0.3), f"🚀 Treinando {name}...")
            
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
            
            results[name] = {
                'val_accuracy': val_acc,
                'test_accuracy': test_acc,
                'precision': test_prec,
                'recall': test_rec,
                'f1_score': test_f1
            }
            
            if test_f1 > best_score:
                best_score = test_f1
                best_model = model
        
        if progress_callback:
            progress_callback(0.9, "💾 Salvando modelo...")
        
        # Preparar dados do modelo
        model_data = {
            'model': best_model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'team_stats': team_stats,
            'results': results,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': len(df),
            'features_count': len(feature_cols)
        }
        
        # Salvar modelo
        try:
            model_path = os.path.join(MODEL_DIR, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
            joblib.dump(model_data, model_path)
        except:
            pass
        
        if progress_callback:
            progress_callback(1.0, "✅ Modelo treinado com sucesso!")
        
        return model_data, "✅ Modelo treinado com sucesso!"
        
    except Exception as e:
        return None, f"❌ Erro: {str(e)}"

def predict_matches_today(fixtures, model_data):
    """Faz previsões para os jogos"""
    predictions = []
    
    if not model_data:
        return predictions
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = model_data['feature_cols']
    team_stats = model_data['team_stats']
    
    for fixture in fixtures:
        if fixture['fixture']['status']['short'] not in ['NS', 'TBD']:
            continue
        
        try:
            home_id = fixture['teams']['home']['id']
            away_id = fixture['teams']['away']['id']
            
            if home_id not in team_stats or away_id not in team_stats:
                continue
            
            home_stats = team_stats[home_id]
            away_stats = team_stats[away_id]
            
            # Calcular features
            home_over_rate = home_stats['over_05'] / max(home_stats['games'], 1)
            home_avg_goals = home_stats['goals_scored'] / max(home_stats['games'], 1)
            away_over_rate = away_stats['over_05'] / max(away_stats['games'], 1)
            away_avg_goals = away_stats['goals_scored'] / max(away_stats['games'], 1)
            
            # Consistência
            if len(home_stats['goals_list']) > 1:
                home_consistency = 1 / (1 + np.std(home_stats['goals_list']) / (np.mean(home_stats['goals_list']) + 0.01))
            else:
                home_consistency = 0.5
                
            if len(away_stats['goals_list']) > 1:
                away_consistency = 1 / (1 + np.std(away_stats['goals_list']) / (np.mean(away_stats['goals_list']) + 0.01))
            else:
                away_consistency = 0.5
            
            # Momentum
            home_recent = home_stats['over_list'][-5:] if len(home_stats['over_list']) >= 5 else home_stats['over_list']
            away_recent = away_stats['over_list'][-5:] if len(away_stats['over_list']) >= 5 else away_stats['over_list']
            
            home_momentum = sum(home_recent) / len(home_recent) if home_recent else home_over_rate
            away_momentum = sum(away_recent) / len(away_recent) if away_recent else away_over_rate
            
            # Criar features para previsão
            features = {
                'home_over_rate': home_over_rate,
                'home_avg_goals': home_avg_goals,
                'home_home_over_rate': home_stats['home_over'] / max(home_stats['home_games'], 1),
                'away_over_rate': away_over_rate,
                'away_avg_goals': away_avg_goals,
                'away_away_over_rate': away_stats['away_over'] / max(away_stats['away_games'], 1),
                'league_over_rate': 0.5,  # Default
                'combined_over_rate': (home_over_rate + away_over_rate) / 2,
                'combined_goals': home_avg_goals + away_avg_goals,
                'home_consistency': home_consistency,
                'away_consistency': away_consistency,
                'consistency_avg': (home_consistency + away_consistency) / 2,
                'consistency_diff': abs(home_consistency - away_consistency),
                'home_momentum': home_momentum,
                'away_momentum': away_momentum,
                'momentum_avg': (home_momentum + away_momentum) / 2,
                'momentum_diff': abs(home_momentum - away_momentum),
                'combined_strength': (home_over_rate * home_consistency + away_over_rate * away_consistency) / 2
            }
            
            # Preencher features faltantes
            for col in feature_cols:
                if col not in features:
                    features[col] = 0.5
            
            # Fazer previsão
            X = pd.DataFrame([features])[feature_cols]
            X_scaled = scaler.transform(X)
            
            pred_proba = model.predict_proba(X_scaled)[0]
            pred_class = model.predict(X_scaled)[0]
            confidence = max(pred_proba) * 100
            
            prediction = {
                'home_team': fixture['teams']['home']['name'],
                'away_team': fixture['teams']['away']['name'],
                'league': fixture['league']['name'],
                'country': fixture['league']['country'],
                'kickoff': fixture['fixture']['date'],
                'prediction': 'OVER 0.5' if pred_class == 1 else 'UNDER 0.5',
                'confidence': confidence,
                'probability_over': pred_proba[1] * 100,
                'probability_under': pred_proba[0] * 100
            }
            
            predictions.append(prediction)
            
        except Exception:
            continue
    
    # Ordenar por confiança
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    return predictions

def display_prediction_card(pred):
    """Exibe card de previsão estilizado"""
    with st.container():
        # Header do card
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.subheader(f"⚽ {pred['home_team']} vs {pred['away_team']}")
        
        with col2:
            # Badge de confiança
            if pred['confidence'] > 75:
                st.success(f"**{pred['confidence']:.1f}%**")
            elif pred['confidence'] > 65:
                st.info(f"**{pred['confidence']:.1f}%**")
            else:
                st.warning(f"**{pred['confidence']:.1f}%**")
        
        # Informações do jogo
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"🏆 **Liga:** {pred['league']} ({pred['country']})")
        with col2:
            try:
                hora = pred['kickoff'][11:16]
                st.write(f"🕐 **Horário:** {hora}")
            except:
                st.write(f"🕐 **Horário:** --:--")
        
        # Previsão
        if pred['prediction'] == 'OVER 0.5':
            st.success(f"🎯 **Previsão:** {pred['prediction']} - {pred['confidence']:.1f}%")
        else:
            st.info(f"🎯 **Previsão:** {pred['prediction']} - {pred['confidence']:.1f}%")
        
        st.markdown("---")

def main():
    st.title("⚽ HT Goals AI Engine")
    st.markdown("🚀 Sistema Inteligente de Previsão Over 0.5 HT")
    
    # Sidebar com status
    with st.sidebar:
        st.title("⚙️ Status do Sistema")
        
        # Teste de conexão
        conn_ok, conn_msg = test_api_connection()
        if conn_ok:
            st.success("✅ API conectada")
        else:
            st.error(f"❌ {conn_msg}")
        
        # Status do modelo
        if st.session_state.model_trained and st.session_state.trained_model:
            model_data = st.session_state.trained_model
            st.success("✅ Modelo ativo")
            st.info(f"📅 Treinado: {model_data['training_date']}")
            st.info(f"📊 Jogos: {model_data['total_samples']:,}")
            st.info(f"🎯 Features: {model_data['features_count']}")
            
            # Mostrar melhor resultado
            if 'results' in model_data:
                results = model_data['results']
                best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
                st.success(f"🏆 {best_model[0]}")
                st.success(f"📈 F1: {best_model[1]['f1_score']:.1%}")
        else:
            st.warning("⚠️ Modelo não treinado")
        
        st.markdown("---")
        st.subheader("🧠 Configurações")
        
        # Configurações de treinamento
        days_training = st.slider(
            "📊 Dias para treinamento:",
            min_value=30,
            max_value=180,
            value=60,
            help="Mais dias = modelo mais robusto, mas demora mais"
        )
        
        use_cache = st.checkbox(
            "💾 Usar dados em cache",
            value=True,
            help="Recomendado: usar dados salvos localmente"
        )
    
    # Tabs principais - SIMPLIFICADAS
    tab1, tab2 = st.tabs(["🤖 Treinar Modelo", "🎯 Previsões do Dia"])
    
    with tab1:
        st.header("🤖 Treinamento Automático do Modelo")
        
        # Informações sobre o treinamento
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **📊 Configuração do Treinamento:**
            - **{days_training} dias** de dados históricos
            - **70%** para treinamento
            - **15%** para validação
            - **15%** para teste final
            """)
        
        with col2:
            st.success("""
            **🧠 Features Incluídas:**
            ✅ Taxa Over histórica
            ✅ Média de gols
            ✅ Performance casa/fora
            ✅ Consistência do time
            ✅ Momentum (últimos 5 jogos)
            ✅ Análise da liga
            """)
        
        if not conn_ok:
            st.warning("⚠️ API desconectada - apenas dados em cache disponíveis")
        
        # Botão principal de treinamento
        if not st.session_state.training_in_progress:
            if st.button("🚀 TREINAR MODELO COMPLETO", type="primary", use_container_width=True):
                st.session_state.training_in_progress = True
                
                # Container para progresso
                progress_container = st.container()
                
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(value, message):
                        progress_bar.progress(value)
                        status_text.text(message)
                    
                    # Etapa 1: Carregar dados
                    update_progress(0.05, "📥 Carregando dados históricos...")
                    df = collect_historical_data_smart(days=days_training, use_cached=use_cache)
                    
                    if df.empty:
                        st.error("❌ Não foi possível carregar dados suficientes")
                        st.session_state.training_in_progress = False
                        st.stop()
                    
                    st.success(f"✅ {len(df)} jogos carregados")
                    
                    # Etapa 2: Treinar modelo
                    model_data, message = train_complete_model(df, update_progress)
                    
                    if model_data:
                        st.session_state.trained_model = model_data
                        st.session_state.model_trained = True
                        
                        # Limpar progresso
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Mostrar resultados
                        st.success("🎉 MODELO TREINADO COM SUCESSO!")
                        
                        # Métricas do melhor modelo
                        results = model_data['results']
                        best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
                        best_metrics = best_model[1]
                        
                        st.subheader(f"🏆 Melhor Modelo: {best_model[0]}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🎯 Acurácia", f"{best_metrics['test_accuracy']:.1%}")
                        with col2:
                            st.metric("💎 Precisão", f"{best_metrics['precision']:.1%}")
                        with col3:
                            st.metric("📊 Recall", f"{best_metrics['recall']:.1%}")
                        with col4:
                            st.metric("🏅 F1-Score", f"{best_metrics['f1_score']:.1%}")
                        
                        st.balloons()
                        
                    else:
                        st.error(message)
                    
                    st.session_state.training_in_progress = False
        else:
            st.info("🔄 Treinamento em andamento...")
        
        # Mostrar informações do modelo atual
        if st.session_state.model_trained and st.session_state.trained_model:
            st.markdown("---")
            st.subheader("📊 Modelo Atual")
            
            model_data = st.session_state.trained_model
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📅 Data do Treinamento", model_data['training_date'])
            with col2:
                st.metric("🎮 Total de Jogos", f"{model_data['total_samples']:,}")
            with col3:
                st.metric("🎯 Features", model_data['features_count'])
            
            # Comparação de modelos
            if 'results' in model_data:
                st.subheader("🏆 Comparação de Algoritmos")
                
                results_df = pd.DataFrame(model_data['results']).T
                results_df = results_df.round(3)
                results_df.columns = ['Val. Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1-Score']
                
                st.dataframe(results_df, use_container_width=True)
    
    with tab2:
        st.header("🎯 Previsões do Dia")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ **Modelo não treinado!**")
            st.info("👈 Vá para a aba 'Treinar Modelo' primeiro")
            st.stop()
        
        # Seletor de data
        selected_date = st.date_input(
            "📅 Selecionar data:",
            value=datetime.now().date(),
            help="Escolha a data para ver as previsões"
        )
        
        date_str = selected_date.strftime('%Y-%m-%d')
        
        # Buscar jogos
        with st.spinner("🔍 Buscando jogos..."):
            fixtures = get_fixtures_cached(date_str)
        
        if not fixtures:
            st.info(f"📅 Nenhum jogo encontrado para {selected_date.strftime('%d/%m/%Y')}")
            if not conn_ok:
                st.error("❌ Verifique sua conexão com a internet")
        else:
            # Fazer previsões
            with st.spinner("🤖 Gerando previsões..."):
                predictions = predict_matches_today(fixtures, st.session_state.trained_model)
            
            if not predictions:
                st.info("🤷 Nenhuma previsão disponível (times sem histórico)")
            else:
                # Estatísticas resumo
                total_games = len(predictions)
                over_predictions = len([p for p in predictions if p['prediction'] == 'OVER 0.5'])
                high_confidence = len([p for p in predictions if p['confidence'] > 70])
                avg_confidence = sum([p['confidence'] for p in predictions]) / len(predictions)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎮 Total Jogos", total_games)
                with col2:
                    st.metric("📈 Over 0.5", over_predictions)
                with col3:
                    st.metric("🎯 Alta Confiança", high_confidence)
                with col4:
                    st.metric("💯 Confiança Média", f"{avg_confidence:.1f}%")
                
                st.markdown("---")
                
                # Melhores apostas
                best_bets = [p for p in predictions if p['prediction'] == 'OVER 0.5' and p['confidence'] > 60]
                best_bets.sort(key=lambda x: x['confidence'], reverse=True)
                
                if best_bets:
                    st.subheader("🏆 Melhores Apostas Over 0.5 HT")
                    
                    for pred in best_bets[:8]:  # Top 8
                        display_prediction_card(pred)
                else:
                    st.info("🤷 Nenhuma aposta Over 0.5 com boa confiança hoje")
                
                # Lista completa
                st.subheader("📋 Todas as Previsões")
                
                # Criar tabela resumo
                table_data = []
                for pred in predictions:
                    try:
                        hora = pred['kickoff'][11:16]
                    except:
                        hora = "--:--"
                    
                    table_data.append({
                        'Hora': hora,
                        'Jogo': f"{pred['home_team']} vs {pred['away_team']}",
                        'Liga': pred['league'],
                        'Previsão': pred['prediction'],
                        'Confiança': f"{pred['confidence']:.0f}%"
                    })
                
                df_table = pd.DataFrame(table_data)
                st.dataframe(df_table, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
