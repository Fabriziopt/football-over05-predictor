# Seletor de ligas
        st.subheader("🏆 Selecionar Ligas para Análise")
        
        # Obter lista de ligas disponíveis nos dados
        if 'df' not in st.session_state:
            st.session_state.df = None
            
        # Carregar dados para ver ligas disponíveis
        if st.session_state.df is None:
            with st.spinner("📊 Verificando ligas disponíveis..."):
                df_temp = collect_historical_data_smart(days=30, use_cached=True)
                if not df_temp.empty:
                    st.session_state.df = df_temp
        
        if st.session_state.df is not None and not st.session_state.df.empty:
            # Agrupar ligas por país
            leagues_by_country = {}
            league_info = st.session_state.df.groupby(['country', 'league_id', 'league_name']).size().reset_index(name='matches')
            
            for _, row in league_info.iterrows():
                country = row['country']
                if country not in leagues_by_country:
                    leagues_by_country[country] = []
                leagues_by_country[country].append({
                    'league_id': row['league_id'],
                    'league_name': row['league_name'],
                    'matches': row['matches']
                })
            
            # Criar tabs por país
            countries = sorted(leagues_by_country.keys())
            
            # Adicionar opção de selecionar todas
            select_all = st.checkbox("📋 Selecionar TODAS as ligas", value=False)
            
            selected_leagues = []
            
            if not select_all:
                # Tabs por país para seleção mais organizada
                st.markdown("### 🌍 Escolha as ligas por país:")
                
                # Criar colunas para países
                cols = st.columns(3)
                col_idx = 0
                
                for country in countries:
                    with cols[col_idx % 3]:
                        st.markdown(f"**{country}**")
                        for league in leagues_by_country[country]:
                            if league['matches'] >= min_matches:
                                if st.checkbox(
                                    f"{league['league_name']} ({league['matches']} jogos)",
                                    key=f"league_{league['league_id']}"
                                ):
                                    selected_leagues.append(league['league_id'])
                    col_idx += 1
                
                # Mostrar ligas populares como sugestão
                st.markdown("---")
                st.markdown("### 🌟 Ligas Populares (Sugestão)")
                
                popular_leagues = {
                    "Premier League": 39,
                    "La Liga": 140,
                    "Serie A": 135,
                    "Bundesliga": 78,
                    "Ligue 1": 61,
                    "Primeira Liga": 94,
                    "Eredivisie": 88,
                    "Championship": 40,
                    "Serie B": 136,
                    "La Liga 2": 141
                }
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("⚽ Selecionar Top 5 Europeias"):
                        for league_name, league_id in list(popular_leagues.items())[:5]:
                            st.session_state[f"league_{league_id}"] = True
                        st.rerun()
                
                with col2:
                    if st.button("🎯 Selecionar Ligas Portuguesas"):
                        # IDs das ligas portuguesas (você precisa verificar os IDs corretos)
                        portuguese_leagues = [94, 95, 96]  # Primeira Liga, Segunda Liga, etc
                        for league_id in portuguese_leagues:
                            st.session_state[f"league_{league_id}"] = True
                        st.rerun()
            
            # Mostrar resumo das ligas selecionadas
            if select_all:
                total_selected = len(league_info[league_info['matches'] >= min_matches])
                st.success(f"✅ Todas as {total_selected} ligas com {min_matches}+ jogos serão analisadas")
            else:
                if selected_leagues:
                    st.info(f"📋 {len(selected_leagues)} ligas selecionadas para análise")
                else:
                    st.warning("⚠️ Nenhuma liga selecionada! Selecione pelo menos uma liga.")
        
        st.markdown("---")import streamlit as st
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
                
                # Garantir que temos coluna date como datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                if 'ht_home' in df.columns and 'ht_away' in df.columns:
                    df['over_05'] = (df['ht_home'] + df['ht_away']) > 0
                elif 'ht_home_goals' in df.columns and 'ht_away_goals' in df.columns:
                    df['over_05'] = (df['ht_home_goals'] + df['ht_away_goals']) > 0
                
                return df, f"✅ {len(df)} jogos carregados do cache local"
            except Exception as e:
                continue
    
    return None, "❌ Nenhum arquivo de dados históricos encontrado"

def collect_historical_data_smart(days=730, use_cached=True):
    """Coleta inteligente de dados históricos"""
    
    df_from_cache = pd.DataFrame()
    
    # 1. Tentar carregar do cache primeiro
    if use_cached:
        df_cache, message = load_historical_data()
        if df_cache is not None and not df_cache.empty:
            if 'date' in df_cache.columns:
                df_cache['date'] = pd.to_datetime(df_cache['date'])
                current_date = datetime.now()
                cutoff_date = current_date - timedelta(days=days)
                df_from_cache = df_cache[df_cache['date'] >= cutoff_date].copy()
                
                if len(df_from_cache) > 0:
                    min_date = df_from_cache['date'].min()
                    max_date = df_from_cache['date'].max()
                    actual_days = (max_date - min_date).days
                    st.info(f"📊 Cache: {len(df_cache)} jogos totais → {len(df_from_cache)} jogos filtrados")
                    st.info(f"📅 Período: {min_date.strftime('%d/%m/%Y')} a {max_date.strftime('%d/%m/%Y')} ({actual_days} dias)")
                    
                    if actual_days >= (days * 0.8):
                        return df_from_cache
    
    # 2. Se não tem dados suficientes, buscar da API
    st.warning("⚠️ Coletando dados da API - pode demorar...")
    
    sample_days = []
    for i in range(min(30, days)):
        sample_days.append(i + 1)
    if days > 30:
        for i in range(30, min(90, days), 2):
            sample_days.append(i + 1)
    if days > 90:
        for i in range(90, min(270, days), 3):
            sample_days.append(i + 1)
    if days > 270:
        for i in range(270, min(365, days), 5):
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
                    try:
                        if match['fixture']['status']['short'] == 'FT' and match.get('score', {}).get('halftime'):
                            match_data = extract_match_features(match)
                            if match_data:
                                all_data.append(match_data)
                    except:
                        continue
        except:
            continue
        
        progress = (idx + 1) / len(sample_days)
        progress_bar.progress(progress)
        
        if idx % 5 == 0:
            time.sleep(0.3)
    
    progress_bar.empty()
    
    if all_data:
        df_api = pd.DataFrame(all_data)
        df_api['date'] = pd.to_datetime(df_api['date'])
        
        if not df_from_cache.empty:
            df_combined = pd.concat([df_from_cache, df_api], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['date', 'home_team_id', 'away_team_id'], keep='first')
        else:
            df_combined = df_api
        
        current_date = datetime.now()
        cutoff_date = current_date - timedelta(days=days)
        df_final = df_combined[df_combined['date'] >= cutoff_date].copy()
        
        try:
            cache_file = "data/historical_matches_cache.parquet"
            os.makedirs("data", exist_ok=True)
            df_combined.to_parquet(cache_file)
        except:
            pass
        
        return df_final
    
    return df_from_cache if not df_from_cache.empty else pd.DataFrame()

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

def prepare_league_specific_features(league_df):
    """Prepara features específicas para uma liga"""
    
    if 'over_05' not in league_df.columns:
        if 'ht_home_goals' in league_df.columns and 'ht_away_goals' in league_df.columns:
            league_df['over_05'] = (league_df['ht_home_goals'] + league_df['ht_away_goals']) > 0
    
    if 'date' in league_df.columns:
        league_df = league_df.sort_values('date').reset_index(drop=True)
    
    # Calcular estatísticas gerais da liga
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
                'games': len(team_all_matches),
                'over_05': team_all_matches['over_05'].sum(),
                'over_rate': team_all_matches['over_05'].mean(),
                'goals_scored': 0,
                'goals_conceded': 0,
                'home_games': len(team_home_matches),
                'home_over': team_home_matches['over_05'].sum() if len(team_home_matches) > 0 else 0,
                'home_over_rate': team_home_matches['over_05'].mean() if len(team_home_matches) > 0 else 0,
                'away_games': len(team_away_matches),
                'away_over': team_away_matches['over_05'].sum() if len(team_away_matches) > 0 else 0,
                'away_over_rate': team_away_matches['over_05'].mean() if len(team_away_matches) > 0 else 0,
                'goals_list': [],
                'over_list': []
            }
    
    # Preparar features para ML
    features = []
    
    for idx, row in league_df.iterrows():
        home_id = row['home_team_id']
        away_id = row['away_team_id']
        
        if home_id not in team_stats or away_id not in team_stats:
            continue
        
        home_stats = team_stats[home_id]
        away_stats = team_stats[away_id]
        
        # Calcular features até este ponto (não incluir o jogo atual)
        home_games_before = max(1, home_stats['games'] - 1)
        away_games_before = max(1, away_stats['games'] - 1)
        
        home_over_rate = home_stats['over_05'] / home_games_before
        away_over_rate = away_stats['over_05'] / away_games_before
        
        # Features do jogo
        feature_row = {
            'home_over_rate': home_over_rate,
            'away_over_rate': away_over_rate,
            'home_home_over_rate': home_stats['home_over_rate'],
            'away_away_over_rate': away_stats['away_over_rate'],
            'league_over_rate': league_over_rate,
            'combined_over_rate': (home_over_rate + away_over_rate) / 2,
            'over_rate_diff': abs(home_over_rate - away_over_rate),
            'home_games_played': home_games_before,
            'away_games_played': away_games_before,
            'target': row['over_05']
        }
        
        features.append(feature_row)
        
        # Atualizar estatísticas após o jogo
        ht_home_goals = row.get('ht_home_goals', 0)
        ht_away_goals = row.get('ht_away_goals', 0)
        
        team_stats[home_id]['goals_scored'] += ht_home_goals
        team_stats[home_id]['goals_conceded'] += ht_away_goals
        team_stats[home_id]['goals_list'].append(ht_home_goals)
        team_stats[home_id]['over_list'].append(row['over_05'])
        
        team_stats[away_id]['goals_scored'] += ht_away_goals
        team_stats[away_id]['goals_conceded'] += ht_home_goals
        team_stats[away_id]['goals_list'].append(ht_away_goals)
        team_stats[away_id]['over_list'].append(row['over_05'])
    
    return pd.DataFrame(features), team_stats, league_over_rate

def train_model_for_league(league_df, league_id, league_name, progress_callback=None):
    """Treina modelo específico para uma liga"""
    
    if len(league_df) < 50:
        return None, f"❌ Dados insuficientes para {league_name} (mínimo 50 jogos)"
    
    try:
        # Preparar features
        features_df, team_stats, league_over_rate = prepare_league_specific_features(league_df)
        
        if len(features_df) < 50:
            return None, f"❌ Features insuficientes para {league_name}"
        
        feature_cols = [col for col in features_df.columns if col != 'target']
        X = features_df[feature_cols]
        y = features_df['target']
        
        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Escalar dados
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Treinar modelo
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Avaliar modelo
        test_pred = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, test_pred)
        test_prec = precision_score(y_test, test_pred, zero_division=0)
        test_rec = recall_score(y_test, test_pred, zero_division=0)
        test_f1 = f1_score(y_test, test_pred, zero_division=0)
        
        # Preparar dados do modelo
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'team_stats': team_stats,
            'league_id': league_id,
            'league_name': league_name,
            'league_over_rate': league_over_rate,
            'total_matches': len(league_df),
            'metrics': {
                'accuracy': test_acc,
                'precision': test_prec,
                'recall': test_rec,
                'f1_score': test_f1
            }
        }
        
        return model_data, f"✅ Modelo treinado para {league_name}"
        
    except Exception as e:
        return None, f"❌ Erro ao treinar {league_name}: {str(e)}"

_bar.empty()
        status_text.empty()
    
    return league_models, successful_leagues

def get_top_over_teams_by_league(league_models, min_matches=10):
    """Identifica os times com maior taxa de Over 0.5 HT em cada liga"""
    
    top_teams_report = {}
    
    for league_id, model_data in league_models.items():
        league_name = model_data['league_name']
        team_stats = model_data['team_stats']
        league_over_rate = model_data['league_over_rate']
        
        # Filtrar times com jogos suficientes
        qualified_teams = []
        
        for team_id, stats in team_stats.items():
            if stats['games'] >= min_matches:
                qualified_teams.append({
                    'team_id': team_id,
                    'team_name': stats['team_name'],
                    'over_rate': stats['over_rate'],
                    'games': stats['games'],
                    'home_over_rate': stats['home_over_rate'],
                    'away_over_rate': stats['away_over_rate'],
                    'above_league_avg': stats['over_rate'] - league_over_rate
                })
        
        # Ordenar por taxa de Over 0.5
        qualified_teams.sort(key=lambda x: x['over_rate'], reverse=True)
        
        top_teams_report[league_name] = {
            'league_over_rate': league_over_rate,
            'top_teams': qualified_teams[:10],  # Top 10 times
            'bottom_teams': qualified_teams[-5:] if len(qualified_teams) > 5 else []  # Bottom 5
        }
    
    return top_teams_report

def predict_matches_with_league_models(fixtures, league_models):
    """Faz previsões usando modelos específicos de cada liga"""
    
    predictions = []
    
    for fixture in fixtures:
        if fixture['fixture']['status']['short'] not in ['NS', 'TBD']:
            continue
        
        league_id = fixture['league']['id']
        
        # Verificar se temos modelo para esta liga
        if league_id not in league_models:
            continue
        
        model_data = league_models[league_id]
        model = model_data['model']
        scaler = model_data['scaler']
        feature_cols = model_data['feature_cols']
        team_stats = model_data['team_stats']
        league_over_rate = model_data['league_over_rate']
        
        try:
            home_id = fixture['teams']['home']['id']
            away_id = fixture['teams']['away']['id']
            
            # Verificar se temos estatísticas dos times
            if home_id not in team_stats or away_id not in team_stats:
                continue
            
            home_stats = team_stats[home_id]
            away_stats = team_stats[away_id]
            
            # Criar features para previsão
            features = {
                'home_over_rate': home_stats['over_rate'],
                'away_over_rate': away_stats['over_rate'],
                'home_home_over_rate': home_stats['home_over_rate'],
                'away_away_over_rate': away_stats['away_over_rate'],
                'league_over_rate': league_over_rate,
                'combined_over_rate': (home_stats['over_rate'] + away_stats['over_rate']) / 2,
                'over_rate_diff': abs(home_stats['over_rate'] - away_stats['over_rate']),
                'home_games_played': home_stats['games'],
                'away_games_played': away_stats['games']
            }
            
            # Fazer previsão
            X = pd.DataFrame([features])[feature_cols]
            X_scaled = scaler.transform(X)
            
            pred_proba = model.predict_proba(X_scaled)[0]
            pred_class = model.predict(X_scaled)[0]
            confidence = max(pred_proba) * 100
            
            # Adicionar informações extras
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
                'model_metrics': model_data['metrics']
            }
            
            predictions.append(prediction)
            
        except Exception as e:
            continue
    
    # Ordenar por confiança
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    return predictions

def display_league_analysis(league_models):
    """Exibe análise detalhada das ligas"""
    
    st.header("📊 Análise por Liga")
    
    # Resumo geral
    total_leagues = len(league_models)
    total_matches = sum(model['total_matches'] for model in league_models.values())
    avg_over_rate = np.mean([model['league_over_rate'] for model in league_models.values()])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏆 Total de Ligas", total_leagues)
    with col2:
        st.metric("⚽ Total de Jogos", f"{total_matches:,}")
    with col3:
        st.metric("📈 Média Over 0.5 HT", f"{avg_over_rate:.1%}")
    
    st.markdown("---")
    
    # Análise detalhada por liga
    top_teams_report = get_top_over_teams_by_league(league_models)
    
    # Tabs para cada liga
    league_names = list(top_teams_report.keys())
    tabs = st.tabs(league_names[:10])  # Mostrar até 10 ligas
    
    for idx, (league_name, tab) in enumerate(zip(league_names[:10], tabs)):
        with tab:
            report = top_teams_report[league_name]
            league_rate = report['league_over_rate']
            
            st.subheader(f"🎯 {league_name}")
            st.info(f"**Taxa média da liga**: {league_rate:.1%} Over 0.5 HT")
            
            # Top Times Over
            st.markdown("### 🔥 Top Times Over 0.5 HT")
            
            if report['top_teams']:
                top_df = pd.DataFrame(report['top_teams'][:5])
                top_df['over_rate'] = (top_df['over_rate'] * 100).round(1)
                top_df['home_over_rate'] = (top_df['home_over_rate'] * 100).round(1)
                top_df['away_over_rate'] = (top_df['away_over_rate'] * 100).round(1)
                top_df['above_league_avg'] = (top_df['above_league_avg'] * 100).round(1)
                
                top_df = top_df.rename(columns={
                    'team_name': 'Time',
                    'over_rate': 'Over 0.5 HT %',
                    'games': 'Jogos',
                    'home_over_rate': 'Casa %',
                    'away_over_rate': 'Fora %',
                    'above_league_avg': 'Acima da Média %'
                })
                
                st.dataframe(top_df[['Time', 'Over 0.5 HT %', 'Jogos', 'Casa %', 'Fora %', 'Acima da Média %']], 
                           hide_index=True, use_container_width=True)
            
            # Times a evitar
            if report['bottom_teams']:
                st.markdown("### ❄️ Times Under (Evitar)")
                bottom_df = pd.DataFrame(report['bottom_teams'])
                bottom_df['over_rate'] = (bottom_df['over_rate'] * 100).round(1)
                
                bottom_df = bottom_df.rename(columns={
                    'team_name': 'Time',
                    'over_rate': 'Over 0.5 HT %',
                    'games': 'Jogos'
                })
                
                st.dataframe(bottom_df[['Time', 'Over 0.5 HT %', 'Jogos']], 
                           hide_index=True, use_container_width=True)

def display_prediction_card_enhanced(pred):
    """Exibe card de previsão com informações detalhadas"""
    
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
        
        # Informações do jogo com média da liga
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"🏆 **Liga:** {pred['league']} ({pred['country']})")
            try:
                hora = pred['kickoff'][11:16]
                st.write(f"🕐 **Horário:** {hora}")
            except:
                st.write(f"🕐 **Horário:** --:--")
        with col2:
            # Mostrar média da liga com destaque
            league_avg = pred['league_over_rate']
            st.write(f"📊 **Média da Liga:** {league_avg:.1f}%")
            
            # Indicador se está acima ou abaixo da média
            home_rate = pred['home_team_stats']['over_rate'] * 100
            away_rate = pred['away_team_stats']['over_rate'] * 100
            avg_teams = (home_rate + away_rate) / 2
            
            if avg_teams > league_avg * 1.1:  # 10% acima da média
                st.write("✅ **Acima da média da liga**")
            elif avg_teams < league_avg * 0.9:  # 10% abaixo da média
                st.write("⚠️ **Abaixo da média da liga**")
            else:
                st.write("➖ **Na média da liga**")
        
        # Estatísticas dos times
        st.markdown("### 📊 Estatísticas dos Times")
        col1, col2 = st.columns(2)
        with col1:
            home_rate = pred['home_team_stats']['over_rate'] * 100
            st.write(f"🏠 **{pred['home_team']}**")
            st.write(f"- Over 0.5 HT: **{home_rate:.1f}%**")
            st.write(f"- Jogos analisados: {pred['home_team_stats']['games']}")
            st.write(f"- Casa: {pred['home_team_stats']['home_over_rate']*100:.1f}%")
        with col2:
            away_rate = pred['away_team_stats']['over_rate'] * 100
            st.write(f"✈️ **{pred['away_team']}**")
            st.write(f"- Over 0.5 HT: **{away_rate:.1f}%**")
            st.write(f"- Jogos analisados: {pred['away_team_stats']['games']}")
            st.write(f"- Fora: {pred['away_team_stats']['away_over_rate']*100:.1f}%")
        
        # Previsão com análise
        st.markdown("### 🎯 Previsão do Modelo")
        if pred['prediction'] == 'OVER 0.5':
            st.success(f"**{pred['prediction']} HT** - Confiança: {pred['confidence']:.1f}%")
            st.write(f"Probabilidade Over: **{pred['probability_over']:.1f}%** | Under: {pred['probability_under']:.1f}%")
        else:
            st.info(f"**{pred['prediction']} HT** - Confiança: {pred['confidence']:.1f}%")
            st.write(f"Probabilidade Under: **{pred['probability_under']:.1f}%** | Over: {pred['probability_over']:.1f}%")
        
        # Critérios utilizados
        with st.expander("🔍 Ver critérios da análise"):
            st.write("**O modelo considera:**")
            st.write("- ✅ Taxa histórica de Over 0.5 HT de cada time")
            st.write("- ✅ Performance em casa vs fora")
            st.write("- ✅ Média de Over 0.5 HT da liga")
            st.write("- ✅ Número de jogos analisados (confiabilidade)")
            st.write("- ✅ Diferença entre as taxas dos times")
            st.write("- ✅ Combinação das taxas dos dois times")
            
            st.write("\n**Métricas do modelo desta liga:**")
            metrics = pred['model_metrics']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Acurácia", f"{metrics['accuracy']:.1%}")
            with col2:
                st.metric("Precisão", f"{metrics['precision']:.1%}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.1%}")
            with col4:
                st.metric("F1-Score", f"{metrics['f1_score']:.1%}")
        
        st.markdown("---")

def main():
    st.title("⚽ HT Goals AI Engine - Análise por Liga")
    st.markdown("🚀 Sistema Inteligente com Modelos Específicos por Liga")
    
    # Sidebar com status
    with st.sidebar:
        st.title("⚙️ Status do Sistema")
        
        # Teste de conexão
        conn_ok, conn_msg = test_api_connection()
        if conn_ok:
            st.success("✅ API conectada")
        else:
            st.error(f"❌ {conn_msg}")
        
        # Status dos modelos
        if st.session_state.models_trained and st.session_state.league_models:
            st.success("✅ Modelos ativos")
            st.info(f"🏆 {len(st.session_state.league_models)} ligas treinadas")
            
            # Listar ligas treinadas
            st.markdown("### 📋 Ligas Disponíveis:")
            for league_id, model_data in st.session_state.league_models.items():
                league_name = model_data['league_name']
                league_rate = model_data['league_over_rate']
                st.write(f"• {league_name}: {league_rate:.1%}")
        else:
            st.warning("⚠️ Modelos não treinados")
        
        st.markdown("---")
        st.subheader("🧠 Configurações")
        
        days_training = st.slider(
            "📊 Dias para treinamento:",
            min_value=30,
            max_value=730,
            value=730,
            step=30,
            help="Mais dias = mais dados por liga"
        )
        
        min_matches = st.slider(
            "🎮 Mínimo de jogos por liga:",
            min_value=30,
            max_value=200,
            value=50,
            help="Ligas com menos jogos serão ignoradas"
        )
        
        use_cache = st.checkbox(
            "💾 Usar dados em cache",
            value=True,
            help="Recomendado para economia de tempo"
        )
    
    # Tabs principais
    tab1, tab2, tab3 = st.tabs(["🤖 Treinar Modelos", "📊 Análise de Ligas", "🎯 Previsões do Dia"])
    
    with tab1:
        st.header("🤖 Treinamento de Modelos por Liga")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **📊 Configuração:**
            - **{days_training} dias** de dados históricos
            - **Mínimo {min_matches} jogos** por liga
            - **Um modelo específico** por liga
            """)
        
        with col2:
            st.success("""
            **🎯 Vantagens:**
            ✅ Aprende padrões específicos de cada liga
            ✅ Identifica "Super Times Over"
            ✅ Maior precisão nas previsões
            ✅ Análise detalhada por competição
            """)
        
        if not st.session_state.training_in_progress:
            if st.button("🚀 TREINAR MODELOS POR LIGA", type="primary", use_container_width=True):
                st.session_state.training_in_progress = True
                
                # Carregar dados
                with st.spinner("📥 Carregando dados históricos..."):
                    df = collect_historical_data_smart(days=days_training, use_cached=use_cache)
                
                if df.empty:
                    st.error("❌ Não foi possível carregar dados")
                    st.session_state.training_in_progress = False
                    st.stop()
                
                # Mostrar resumo dos dados
                st.success(f"✅ {len(df)} jogos carregados")
                
                # Treinar modelos por liga
                st.subheader("🏆 Treinando Modelos por Liga")
                
                league_models, successful = train_models_by_league(df, min_matches_per_league=min_matches)
                
                if league_models:
                    st.session_state.league_models = league_models
                    st.session_state.models_trained = True
                    
                    st.success(f"🎉 {successful} LIGAS TREINADAS COM SUCESSO!")
                    st.balloons()
                else:
                    st.error("❌ Nenhum modelo foi treinado com sucesso")
                
                st.session_state.training_in_progress = False
        else:
            st.info("🔄 Treinamento em andamento...")
    
    with tab2:
        st.header("📊 Análise Detalhada por Liga")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Treine os modelos primeiro!")
            st.stop()
        
        display_league_analysis(st.session_state.league_models)
    
    with tab3:
        st.header("🎯 Previsões do Dia")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Treine os modelos primeiro!")
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
        else:
            # Fazer previsões
            with st.spinner("🤖 Gerando previsões com modelos específicos..."):
                predictions = predict_matches_with_league_models(fixtures, st.session_state.league_models)
            
            if not predictions:
                st.info("🤷 Nenhuma previsão disponível (verifique se as ligas foram treinadas)")
            else:
                # Estatísticas
                total_games = len(predictions)
                over_predictions = len([p for p in predictions if p['prediction'] == 'OVER 0.5'])
                high_confidence = len([p for p in predictions if p['confidence'] > 70])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎮 Total Jogos", total_games)
                with col2:
                    st.metric("📈 Over 0.5", over_predictions)
                with col3:
                    st.metric("🎯 Alta Confiança", high_confidence)
                
                st.markdown("---")
                
                # Filtros
                st.subheader("🎯 Filtrar Previsões")
                
                col1, col2 = st.columns(2)
                with col1:
                    min_confidence = st.slider("Confiança mínima:", 50, 90, 65)
                with col2:
                    only_over = st.checkbox("Mostrar apenas OVER 0.5", value=True)
                
                # Filtrar previsões
                filtered_predictions = predictions
                if only_over:
                    filtered_predictions = [p for p in filtered_predictions if p['prediction'] == 'OVER 0.5']
                filtered_predictions = [p for p in filtered_predictions if p['confidence'] >= min_confidence]
                
                # Mostrar previsões
                st.subheader(f"🏆 {len(filtered_predictions)} Jogos Selecionados")
                
                # Opção de visualização
                view_mode = st.radio("Modo de visualização:", ["Cards Detalhados", "Tabela Resumida"], horizontal=True)
                
                if view_mode == "Cards Detalhados":
                    for pred in filtered_predictions[:10]:  # Top 10
                        display_prediction_card_enhanced(pred)
                else:
                    # Criar tabela resumida com média da liga
                    table_data = []
                    for pred in filtered_predictions:
                        try:
                            hora = pred['kickoff'][11:16]
                        except:
                            hora = "--:--"
                        
                        # Calcular se está acima da média
                        home_rate = pred['home_team_stats']['over_rate'] * 100
                        away_rate = pred['away_team_stats']['over_rate'] * 100
                        avg_teams = (home_rate + away_rate) / 2
                        league_avg = pred['league_over_rate']
                        
                        # Indicador visual
                        if avg_teams > league_avg * 1.1:
                            indicator = "✅"
                        elif avg_teams < league_avg * 0.9:
                            indicator = "⚠️"
                        else:
                            indicator = "➖"
                        
                        table_data.append({
                            'Hora': hora,
                            'Jogo': f"{pred['home_team']} vs {pred['away_team']}",
                            'Liga': pred['league'],
                            'Média Liga': f"{league_avg:.0f}%",
                            'Média Times': f"{avg_teams:.0f}%",
                            'Status': indicator,
                            'Previsão': pred['prediction'],
                            'Confiança': f"{pred['confidence']:.0f}%"
                        })
                    
                    df_table = pd.DataFrame(table_data)
                    
                    # Aplicar cores condicionais
                    def color_confidence(val):
                        num = int(val.strip('%'))
                        if num >= 75:
                            return 'background-color: #90EE90'
                        elif num >= 65:
                            return 'background-color: #87CEEB'
                        else:
                            return 'background-color: #FFE4B5'
                    
                    styled_df = df_table.style.applymap(
                        color_confidence, 
                        subset=['Confiança']
                    )
                    
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                    
                    # Legenda
                    st.markdown("### 📊 Legenda")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("✅ = Times acima da média da liga (>10%)")
                    with col2:
                        st.write("➖ = Times na média da liga (±10%)")
                    with col3:
                        st.write("⚠️ = Times abaixo da média da liga (<10%)")

if __name__ == "__main__":
    main()
