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
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None

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
        # Tenta salvar em arquivo para persistência entre sessões
        try:
            joblib.dump(league_models, os.path.join(MODEL_DIR, f"league_models_{step}.joblib"))
        except:
            pass
        return True
    except:
        return False

def load_training_progress():
    """Carrega progresso salvo"""
    try:
        # Primeiro tenta carregar da session_state
        if st.session_state.models_backup:
            st.session_state.league_models = st.session_state.models_backup.copy()
            st.session_state.models_trained = True
            return True
        
        # Se não encontrar, tenta carregar do arquivo
        try:
            for filename in ["league_models_final.joblib", "league_models_backup.joblib"]:
                filepath = os.path.join(MODEL_DIR, filename)
                if os.path.exists(filepath):
                    league_models = joblib.load(filepath)
                    st.session_state.league_models = league_models
                    st.session_state.models_backup = league_models.copy()
                    st.session_state.models_trained = True
                    return True
        except:
            pass
        
        return False
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
    # Se já temos dados carregados na session_state, usa-os
    if st.session_state.historical_data is not None:
        return st.session_state.historical_data, "✅ Dados carregados da sessão"
        
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
                
                # Guardar na session_state
                st.session_state.historical_data = df
                
                return df, f"✅ {len(df)} jogos carregados de {file_path}"
            except Exception as e:
                st.warning(f"⚠️ Erro ao carregar {file_path}: {str(e)}")
                continue
    
    return None, "❌ Nenhum arquivo encontrado"

def get_seasonal_data_period():
    """Calcula período ideal baseado na temporada - agora inclui múltiplas temporadas"""
    current_date = datetime.now()
    current_month = current_date.month
    current_year = current_date.year
    
    # Queremos ao menos 3 temporadas de dados para análise profunda
    # A maioria das ligas começa em agosto, então usamos isso como referência
    start_date = datetime(current_year - 3, 8, 1)  # 3 anos atrás
    days_back = (current_date - start_date).days
    
    # Mínimo de 3 anos
    days_back = max(days_back, 3*365)
    
    return days_back, start_date

def collect_historical_data_smart(days=None, use_cached=True, seasonal=True, include_all_seasons=True):
    """Coleta inteligente com opção para incluir múltiplas temporadas e tratamento robusto"""
    
    if include_all_seasons:
        days, start_date = get_seasonal_data_period()
        st.info(f"📅 Modo Múltiplas Temporadas: Buscando dados desde {start_date.strftime('%d/%m/%Y')} ({days} dias)")
    elif seasonal and days is None:
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
                    
                    if len(df_filtered) > 100:  # Mínimo de dados
                        st.success(f"✅ {len(df_filtered)} jogos carregados do cache")
                        return df_filtered
            except Exception as e:
                st.warning(f"⚠️ Erro ao processar cache: {str(e)}")
    
    # Buscar da API se necessário
    st.warning("⚠️ Coletando dados da API...")
    
    # Amostragem inteligente por temporadas
    sample_days = []
    
    # Últimos 30 dias - todos os dias
    for i in range(min(30, days)):
        sample_days.append(i + 1)
    
    # 30-90 dias - a cada 2 dias
    if days > 30:
        for i in range(30, min(90, days), 2):
            sample_days.append(i + 1)
    
    # 90-365 dias - a cada 3 dias
    if days > 90:
        for i in range(90, min(365, days), 3):
            sample_days.append(i + 1)
    
    # Mais de 1 ano - amostragem estratégica por temporada
    if days > 365:
        # Para cada temporada anterior, pegamos pontos estratégicos (meio da temporada, início, fim)
        for year_back in range(1, int(days/365) + 1):
            # Pontos médios da temporada (meses diferentes para garantir diversidade)
            for month in [9, 11, 2, 4]:  # Set, Nov, Fev, Abr - pontos estratégicos da temporada
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
                st.session_state.training_in_progress = False day_offset in [5, 15, 25]:  # Início, meio e fim do mês
                    try:
                        year = current_date.year - year_back
                        sample_date = datetime(year, month, day_offset)
                        days_diff = (current_date - sample_date).days
                        if days_diff > 0 and days_diff <= days:
                            sample_days.append(days_diff)
                    except:
                        continue
    
    # Remover duplicatas e ordenar
    sample_days = sorted(list(set(sample_days)))
    
    all_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    errors_count = 0
    max_errors = 20  # Máximo de erros permitidos - aumentado para coleta mais completa
    
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
    
    # Criar DataFrame e salvar na session_state para futuros usos
    df_result = pd.DataFrame(all_data)
    st.session_state.historical_data = df_result
    
    return df_result

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
            'over_05': 1 if (int(ht_home) + int(ht_away)) > 0 else 0,
            # Adicionar informação de temporada (baseada na data)
            'season': get_season_from_date(match['fixture']['date'][:10])
        }
        
        return features
    except Exception as e:
        return None

def get_season_from_date(date_str):
    """Extrai a temporada a partir da data"""
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        year = date.year
        month = date.month
        
        # Para meses a partir de julho, consideramos a temporada como ano/ano+1
        # Para meses até junho, consideramos a temporada como ano-1/ano
        if month >= 7:
            return f"{year}/{year+1}"
        else:
            return f"{year-1}/{year}"
    except:
        # Fallback para temporada atual
        current_year = datetime.now().year
        return f"{current_year-1}/{current_year}"

def calculate_poisson_probabilities(home_avg, away_avg, improved=True):
    """Calcula probabilidades usando distribuição de Poisson com validação e melhorias"""
    try:
        # Validar inputs
        if home_avg < 0 or away_avg < 0:
            return {'poisson_over_05': 0.5, 'expected_goals_ht': 0.5, 'home_lambda': 0.25, 'away_lambda': 0.25}
        
        # Lambda para cada time (média de gols esperados)
        home_lambda = max(home_avg / 2, 0.01)  # Mínimo 0.01
        away_lambda = max(away_avg / 2, 0.01)
        
        # MELHORIA: Ajuste dinâmico baseado em observações empíricas
        # Estudos mostram que o modelo Poisson simples tende a subestimar empates
        if improved:
            # Ajuste para correlação negativa entre gols (tendência de empate)
            correction_factor = 0.85  # Fator de correção empírico
            home_lambda = home_lambda * correction_factor
            away_lambda = away_lambda * correction_factor
        
        # Probabilidade de 0 gols para cada time
        prob_home_0 = poisson.pmf(0, home_lambda)
        prob_away_0 = poisson.pmf(0, away_lambda)
        
        # Probabilidade de 0-0 no HT
        prob_0_0 = prob_home_0 * prob_away_0
        
        # Probabilidade de Over 0.5 HT
        prob_over_05 = 1 - prob_0_0
        
        # Gols esperados no HT
        expected_goals_ht = home_lambda + away_lambda
        
        # MELHORIA: Cálculo de probabilidades específicas para apostas
        prob_exact_1 = (poisson.pmf(1, home_lambda) * prob_away_0) + (prob_home_0 * poisson.pmf(1, away_lambda))
        prob_2_plus = 1 - prob_0_0 - prob_exact_1
        
        return {
            'poisson_over_05': min(max(prob_over_05, 0), 1),
            'expected_goals_ht': max(expected_goals_ht, 0),
            'home_lambda': home_lambda,
            'away_lambda': away_lambda,
            'prob_0_0': prob_0_0,
            'prob_exact_1': prob_exact_1,
            'prob_2_plus': prob_2_plus
        }
    except:
        return {'poisson_over_05': 0.5, 'expected_goals_ht': 0.5, 'home_lambda': 0.25, 'away_lambda': 0.25}

def calculate_advanced_features(league_df, include_recent_form=True):
    """Calcula features avançadas com tratamento robusto de erros e análise de forma recente"""
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
            league_df['date'] = pd.to_datetime(league_df['date'], errors='coerce')
            league_df = league_df.sort_values('date').reset_index(drop=True)
        
        # Estatísticas da liga com fallbacks
        league_over_rate = league_df['over_05'].mean() if len(league_df) > 0 else 0.5
        league_avg_goals = league_df['ht_total_goals'].mean() if len(league_df) > 0 else 1.0
        
        # Analisar tendência da liga nas últimas temporadas
        league_trend = analyze_league_trend(league_df)
        
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
                
                # MELHORIA: Análise de forma recente (últimos 5 jogos)
                recent_form = {}
                if include_recent_form and len(all_matches) >= 3:
                    recent_matches = all_matches.sort_values('date', ascending=False).head(5)
                    recent_form = {
                        'recent_over_rate': recent_matches['over_05'].mean(),
                        'recent_goals_scored': (
                            recent_matches[recent_matches['home_team_id'] == team_id]['ht_home_goals'].sum() +
                            recent_matches[recent_matches['away_team_id'] == team_id]['ht_away_goals'].sum()
                        ) / len(recent_matches),
                        'recent_goals_conceded': (
                            recent_matches[recent_matches['home_team_id'] == team_id]['ht_away_goals'].sum() +
                            recent_matches[recent_matches['away_team_id'] == team_id]['ht_home_goals'].sum()
                        ) / len(recent_matches)
                    }
                
                # Análise com fallbacks
                home_goals_scored = home_matches['ht_home_goals'].mean() if len(home_matches) > 0 else league_avg_goals/2
                home_goals_conceded = home_matches['ht_away_goals'].mean() if len(home_matches) > 0 else league_avg_goals/2
                away_goals_scored = away_matches['ht_away_goals'].mean() if len(away_matches) > 0 else league_avg_goals/2
                away_goals_conceded = away_matches['ht_home_goals'].mean() if len(away_matches) > 0 else league_avg_goals/2
                
                # MELHORIA: Análise de tendência (melhorando ou piorando)
                team_trend = analyze_team_trend(all_matches, team_id)
                
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
                    'away_defense_strength': max(away_goals_conceded / (league_avg_goals/2 + 0.01), 0.1),
                    # Tendência da equipe
                    'overall_trend': team_trend.get('overall_trend', 0),
                    'home_trend': team_trend.get('home_trend', 0),
                    'away_trend': team_trend.get('away_trend', 0),
                    # Forma recente
                    **recent_form
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
                
                # Usar modelo Poisson melhorado
                poisson_calc = calculate_poisson_probabilities(home_expected * 2, away_expected * 2, improved=True)
                
                # MELHORIA: Análise Casa vs Fora específica
                home_away_dynamic = analyze_home_away_dynamic(home_stats, away_stats)
                
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
                    'prob_0_0': poisson_calc['prob_0_0'],
                    'prob_exact_1': poisson_calc['prob_exact_1'],
                    'prob_2_plus': poisson_calc['prob_2_plus'],
                    
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
                    
                    # MELHORIAS: Novas features
                    # Tendência das equipes
                    'home_trend': home_stats.get('home_trend', 0),
                    'away_trend': away_stats.get('away_trend', 0),
                    'combined_trend': (home_stats.get('home_trend', 0) + away_stats.get('away_trend', 0)) / 2,
                    
                    # Forma recente
                    'home_recent_over_rate': home_stats.get('recent_over_rate', home_stats['over_rate']),
                    'away_recent_over_rate': away_stats.get('recent_over_rate', away_stats['over_rate']),
                    'home_recent_goals': home_stats.get('recent_goals_scored', home_stats['home_goals_scored']),
                    'away_recent_goals': away_stats.get('recent_goals_scored', away_stats['away_goals_scored']),
                    
                    # Dinâmica Casa vs Fora
                    'home_dominance': home_away_dynamic.get('home_dominance', 1.0),
                    'away_threat': home_away_dynamic.get('away_threat', 1.0),
                    'matchup_balance': home_away_dynamic.get('matchup_balance', 0.5),
                    
                    # Tendência da liga
                    'league_trend': league_trend.get('overall_trend', 0),
                    
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

def analyze_league_trend(league_df):
    """Analisa tendência da liga ao longo do tempo"""
    try:
        if 'date' not in league_df.columns or len(league_df) < 30:
            return {'overall_trend': 0}
        
        # Ordenar por data
        df = league_df.sort_values('date')
        
        # Dividir em dois períodos para comparar
        half_point = len(df) // 2
        first_half = df.iloc[:half_point]
        second_half = df.iloc[half_point:]
        
        # Calcular taxas de over em cada período
        if len(first_half) > 0 and len(second_half) > 0:
            first_rate = first_half['over_05'].mean()
            second_rate = second_half['over_05'].mean()
            
            # Calcular tendência (-1 a 1, onde positivo indica aumento na taxa)
            trend = (second_rate - first_rate) * 2  # Normalizar para escala desejada
            
            return {
                'overall_trend': trend,
                'first_half_rate': first_rate,
                'second_half_rate': second_rate
            }
        
        return {'overall_trend': 0}
    except:
        return {'overall_trend': 0}

def analyze_team_trend(team_matches, team_id):
    """Analisa tendência da equipe ao longo do tempo"""
    try:
        if 'date' not in team_matches.columns or len(team_matches) < 10:
            return {'overall_trend': 0, 'home_trend': 0, 'away_trend': 0}
        
        # Ordenar por data
        df = team_matches.sort_values('date')
        
        # Dividir em dois períodos para comparar
        half_point = len(df) // 2
        first_half = df.iloc[:half_point]
        second_half = df.iloc[half_point:]
        
        # Geral
        overall_trend = 0
        if len(first_half) > 0 and len(second_half) > 0:
            first_rate = first_half['over_05'].mean()
            second_rate = second_half['over_05'].mean()
            overall_trend = (second_rate - first_rate) * 2
        
        # Casa
        home_trend = 0
        home_first = first_half[first_half['home_team_id'] == team_id]
        home_second = second_half[second_half['home_team_id'] == team_id]
        if len(home_first) > 0 and len(home_second) > 0:
            home_first_rate = home_first['over_05'].mean()
            home_second_rate = home_second['over_05'].mean()
            home_trend = (home_second_rate - home_first_rate) * 2
        
        # Fora
        away_trend = 0
        away_first = first_half[first_half['away_team_id'] == team_id]
        away_second = second_half[second_half['away_team_id'] == team_id]
        if len(away_first) > 0 and len(away_second) > 0:
            away_first_rate = away_first['over_05'].mean()
            away_second_rate = away_second['over_05'].mean()
            away_trend = (away_second_rate - away_first_rate) * 2
        
        return {
            'overall_trend': overall_trend,
            'home_trend': home_trend,
            'away_trend': away_trend
        }
    except:
        return {'overall_trend': 0, 'home_trend': 0, 'away_trend': 0}

def analyze_home_away_dynamic(home_stats, away_stats):
    """Analisa a dinâmica específica Casa vs Fora entre as duas equipes"""
    try:
        # Calcular dominância em casa vs ameaça fora
        home_dominance = (home_stats['home_over_rate'] / max(0.1, home_stats['away_over_rate']))
        away_threat = (away_stats['away_over_rate'] / max(0.1, away_stats['home_over_rate']))
        
        # Normalizar (valores acima de 1 indicam força na condição específica)
        home_dominance = max(0.5, min(2.0, home_dominance))
        away_threat = max(0.5, min(2.0, away_threat))
        
        # Equilíbrio do confronto (0 = dominância em casa, 1 = dominância fora)
        matchup_balance = away_threat / (home_dominance + away_threat)
        
        return {
            'home_dominance': home_dominance,
            'away_threat': away_threat,
            'matchup_balance': matchup_balance
        }
    except:
        return {'home_dominance': 1.0, 'away_threat': 1.0, 'matchup_balance': 0.5}

def train_complete_model_with_validation(league_df, league_id, league_name, min_matches=30):
    """Treina modelo com validação completa e tratamento robusto"""
    
    if len(league_df) < min_matches:
        return None, f"❌ {league_name}: {len(league_df)} jogos < {min_matches} mínimo"
    
    try:
        # Preparar features avançadas (incluindo análise de forma recente)
        features_df, team_stats, league_over_rate = calculate_advanced_features(league_df, include_recent_form=True)
        
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
        
        # MELHORIA: Usar Time Series Split para dados temporais
        # Isso é mais realista para dados de futebol que têm uma sequência temporal
        try:
            # Primeiramente tentar com TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=3, test_size=int(len(X) * 0.2))
            train_indices, test_indices = list(tscv.split(X))[2]  # Pegar a última divisão
            
            X_train_val, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train_val, y_test = y.iloc[train_indices], y.iloc[test_indices]
            
            # Agora dividir entre treino e validação
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
            )
        except:
            # Fallback para o método tradicional
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
            )
        
        # Escalar dados
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # MELHORIA: Modelos mais avançados com calibração de probabilidade
        # Modelos calibrados fornecem estimativas de probabilidade mais confiáveis
        models = {
            'rf': RandomForestClassifier(
                n_estimators=100, max_depth=8, random_state=42, 
                n_jobs=1, min_samples_split=5, min_samples_leaf=2
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
            ),
            'et': ExtraTreesClassifier(
                n_estimators=100, max_depth=8, random_state=42,
                n_jobs=1, min_samples_split=5, min_samples_leaf=2
            )
        }
        
        # Treinar e validar cada modelo
        best_model = None
        best_score = 0
        results = {}
        
        for name, model in models.items():
            try:
                # Treinar com calibração de probabilidade
                calibrated_model = CalibratedClassifierCV(model, cv=3, method='sigmoid')
                calibrated_model.fit(X_train_scaled, y_train)
                
                # Validar
                val_pred = calibrated_model.predict(X_val_scaled)
                val_acc = accuracy_score(y_val, val_pred)
                val_prec = precision_score(y_val, val_pred, zero_division=0)
                val_rec = recall_score(y_val, val_pred, zero_division=0)
                val_f1 = f1_score(y_val, val_pred, zero_division=0)
                
                results[name] = {
                    'val_accuracy': val_acc,
                    'val_precision': val_prec,
                    'val_recall': val_rec,
                    'val_f1': val_f1,
                    'model': calibrated_model
                }
                
                if val_f1 > best_score:
                    best_score = val_f1
                    best_model = calibrated_model
                    
            except Exception as e:
                st.warning(f"⚠️ Erro treinando {name}: {str(e)}")
                continue
        
        if best_model is None:
            return None, f"❌ {league_name}: Nenhum modelo funcionou"
        
        # MELHORIA: Ensemble dos modelos para decisões mais robustas
        try:
            # Criar ensemble dos melhores modelos
            working_models = [(name, model_data['model']) for name, model_data in results.items() 
                            if model_data['val_f1'] > 0.6]
            
            if len(working_models) >= 2:
                ensemble = VotingClassifier(working_models, voting='soft')
                ensemble.fit(X_train_scaled, y_train)
                
                # Verificar se o ensemble é melhor
                ensemble_pred = ensemble.predict(X_val_scaled)
                ensemble_f1 = f1_score(y_val, ensemble_pred, zero_division=0)
                
                if ensemble_f1 > best_score:
                    best_model = ensemble
                    best_score = ensemble_f1
        except:
            pass  # Se falhar, apenas usa o melhor modelo individual
        
        # Testar melhor modelo
        test_pred = best_model.predict(X_test_scaled)
        test_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, test_pred),
            'precision': precision_score(y_test, test_pred, zero_division=0),
            'recall': recall_score(y_test, test_pred, zero_division=0),
            'f1_score': f1_score(y_test, test_pred, zero_division=0)
        }
        
        # MELHORIA: Análise de threshold otimizado usando grid search
        best_threshold = 0.5
        best_f1 = test_metrics['f1_score']
        
        for threshold in np.arange(0.3, 0.8, 0.02):  # Grid mais fino
            try:
                pred_threshold = (test_pred_proba >= threshold).astype(int)
                f1_threshold = f1_score(y_test, pred_threshold, zero_division=0)
                precision_threshold = precision_score(y_test, pred_threshold, zero_division=0)
                
                # Balancear F1 e precisão para maximizar taxa de acerto
                combined_score = (f1_threshold * 0.7) + (precision_threshold * 0.3)
                
                if combined_score > best_f1:
                    best_f1 = combined_score
                    best_threshold = threshold
            except:
                continue
        
        # Retreinar no dataset completo
        X_scaled = scaler.fit_transform(X)
        best_model.fit(X_scaled, y)
        
        # Feature importance
        try:
            # Extrair importância das features
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = dict(zip(feature_cols, best_model.feature_importances_))
            elif hasattr(best_model, 'estimators_') and hasattr(best_model.estimators_[0], 'feature_importances_'):
                # Para VotingClassifier, pegamos a média das importâncias
                importances = np.zeros(len(feature_cols))
                for estimator in best_model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        importances += estimator.feature_importances_
                feature_importance = dict(zip(feature_cols, importances / len(best_model.estimators_)))
            else:
                # Fallback
                feature_importance = {feature: 1.0/len(feature_cols) for feature in feature_cols}
            
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        except:
            top_features = [('unknown', 1.0)]
        
        # MELHORIA: Análise de desempenho em diferentes cenários
        scenario_analysis = analyze_performance_scenarios(X_test_scaled, y_test, best_model, feature_cols)
        
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
            'top_features': top_features,
            'scenario_analysis': scenario_analysis,
            'model_type': type(best_model).__name__
        }
        
        # Adicionar análise de tendência da liga
        league_trend = analyze_league_trend(league_df)
        model_data['league_trend'] = league_trend
        
        return model_data, f"✅ {league_name}: Acc {test_metrics['accuracy']:.1%} | F1 {test_metrics['f1_score']:.1%}"
        
    except Exception as e:
        error_msg = f"❌ {league_name}: {str(e)}"
        st.session_state.training_errors.append(error_msg)
        return None, error_msg

def analyze_performance_scenarios(X_test_scaled, y_test, model, feature_cols):
    """Analisa o desempenho do modelo em diferentes cenários"""
    try:
        # Converter para DataFrame para análise
        X_test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
        X_test_df['target'] = y_test.values
        X_test_df['prediction'] = model.predict(X_test_scaled)
        X_test_df['probability'] = model.predict_proba(X_test_scaled)[:, 1]
        
        # Categorizar em diferentes cenários
        scenarios = {}
        
        # Alta confiança (>75%)
        high_conf = X_test_df[X_test_df['probability'] > 0.75]
        if len(high_conf) > 0:
            scenarios['high_confidence'] = {
                'count': len(high_conf),
                'accuracy': (high_conf['prediction'] == high_conf['target']).mean(),
                'avg_probability': high_conf['probability'].mean()
            }
        
        # Acima da média da liga
        if 'over_rate_vs_league' in X_test_df.columns:
            above_avg = X_test_df[X_test_df['over_rate_vs_league'] > 1.1]
            if len(above_avg) > 0:
                scenarios['above_league_avg'] = {
                    'count': len(above_avg),
                    'accuracy': (above_avg['prediction'] == above_avg['target']).mean(),
                    'avg_probability': above_avg['probability'].mean()
                }
        
        # Alta força de ataque
        if 'attack_index' in X_test_df.columns:
            high_attack = X_test_df[X_test_df['attack_index'] > 1.2]
            if len(high_attack) > 0:
                scenarios['high_attack'] = {
                    'count': len(high_attack),
                    'accuracy': (high_attack['prediction'] == high_attack['target']).mean(),
                    'avg_probability': high_attack['probability'].mean()
                }
        
        return scenarios
    except:
        return {}

def predict_with_strategy(fixtures, league_models, min_confidence=60, no_limit=True):
    """Faz previsões com estratégia inteligente e tratamento robusto - SEM LIMITE de apostas por dia"""
    
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
            
            # Calcular a dinâmica Casa vs Fora
            home_away_dynamic = analyze_home_away_dynamic(home_stats, away_stats)
            
            # Poisson calculation (modelo melhorado)
            home_expected = home_stats['home_attack_strength'] * away_stats['away_defense_strength'] * 0.5
            away_expected = away_stats['away_attack_strength'] * home_stats['home_defense_strength'] * 0.5
            
            poisson_calc = calculate_poisson_probabilities(home_expected * 2, away_expected * 2, improved=True)
            
            # Criar features básicas
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
                'prob_0_0': poisson_calc.get('prob_0_0', 1 - poisson_calc['poisson_over_05']),
                'prob_exact_1': poisson_calc.get('prob_exact_1', 0.3),
                'prob_2_plus': poisson_calc.get('prob_2_plus', 0.2),
                'combined_over_rate': (home_stats['home_over_rate'] + away_stats['away_over_rate']) / 2,
                'attack_index': (home_stats['home_attack_strength'] + away_stats['away_attack_strength']) / 2,
                'game_pace_index': (home_expected + away_expected),
                'over_rate_vs_league': ((home_stats['home_over_rate'] + away_stats['away_over_rate']) / 2) / max(league_over_rate, 0.01),
                'expected_vs_league': poisson_calc['expected_goals_ht'] / 0.5,
                'home_games_played': home_stats['home_games'],
                'away_games_played': away_stats['away_games'],
                'min_games': min(home_stats['home_games'], away_stats['away_games'])
            }
            
            # Adicionar features avançadas quando disponíveis
            if 'recent_over_rate' in home_stats:
                features['home_recent_over_rate'] = home_stats['recent_over_rate']
            if 'recent_over_rate' in away_stats:
                features['away_recent_over_rate'] = away_stats['recent_over_rate']
            if 'home_trend' in home_stats:
                features['home_trend'] = home_stats['home_trend']
            if 'away_trend' in away_stats:
                features['away_trend'] = away_stats['away_trend']
            
            # Adicionar dinâmica casa-fora
            features['home_dominance'] = home_away_dynamic.get('home_dominance', 1.0)
            features['away_threat'] = home_away_dynamic.get('away_threat', 1.0)
            features['matchup_balance'] = home_away_dynamic.get('matchup_balance', 0.5)
            
            # Adicionar tendência da liga se disponível
            if 'league_trend' in model_data:
                features['league_trend'] = model_data['league_trend'].get('overall_trend', 0)
            
            # Garantir que todas as features do modelo estão presentes
            for col in feature_cols:
                if col not in features:
                    features[col] = 0.0  # Valor neutro
            
            # Fazer previsão
            X = pd.DataFrame([features])[feature_cols]
            
            # Tratar missing features
            X = X.fillna(0.0)   # Tratar NaN
            
            X_scaled = scaler.transform(X)
            
            pred_proba = model.predict_proba(X_scaled)[0]
            confidence = pred_proba[1] * 100
            
            # Aplicar threshold otimizado
            pred_class = 1 if pred_proba[1] >= best_threshold else 0
            
            # Calcular indicadores de força
            game_vs_league_percent = (features['combined_over_rate'] / max(league_over_rate, 0.01) - 1) * 100
            
            # Calcular odds justas e análise de valor
            fair_odd = calculate_fair_odds(poisson_calc['poisson_over_05'] * 100)
            
            # Análise avançada do confronto
            matchup_analysis = {
                'home_dominance_score': features['home_dominance'] * 10,  # 0-10 escala
                'away_threat_score': features['away_threat'] * 10,  # 0-10 escala
                'matchup_balance': features['matchup_balance'],  # 0-1 onde 0.5 é equilibrado
                'expected_flow': 'Dominante em Casa' if features['matchup_balance'] < 0.4 else 
                               ('Equilíbrio' if features['matchup_balance'] <= 0.6 else 'Forte Fora')
            }
            
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
                'game_vs_league_percent': game_vs_league_percent,
                'game_vs_league_ratio': features['combined_over_rate'] / max(league_over_rate, 0.01),
                'model_metrics': model_data['test_metrics'],
                'top_features': model_data['top_features'],
                'fair_odds': fair_odd,
                'matchup_analysis': matchup_analysis,
                'fixture_id': fixture['fixture']['id']
            }
            
            # Se estamos no modo sem limite, incluímos todos acima da confiança mínima
            if confidence >= min_confidence:
                predictions.append(prediction)
            
        except Exception as e:
            continue
    
    # Ordenar por confiança
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return predictions

def calculate_fair_odds(poisson_probability_percentage):
    """Calcula a odd justa baseada na probabilidade Poisson"""
    try:
        # Converter Poisson para probabilidade (0-1)
        probability = poisson_probability_percentage / 100
        
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
                if pred['game_vs_league_percent'] > 20:
                    st.write("🔥 **+{:.0f}%**".format(pred['game_vs_league_percent']))
                elif pred['game_vs_league_percent'] < -20:
                    st.write("❄️ **{:.0f}%**".format(pred['game_vs_league_percent']))
                else:
                    st.write("➖ **{:+.0f}%**".format(pred['game_vs_league_percent']))
            
            # Análise detalhada
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"🏆 **Liga:** {pred['league']} ({pred['country']})")
                st.write(f"📊 **Média da Liga:** {pred['league_over_rate']:.1f}%")
                
            with col2:
                st.write(f"🏠 **{pred['home_team']}**")
                st.write(f"- Casa: {pred['home_team_stats']['home_over_rate']*100:.1f}%")
                st.write(f"- Força Ataque: {pred['away_team_stats']['away_attack_strength']:.2f} {pred['home_team_stats']['home_attack_strength']:.2f}")
                
            with col3:
                st.write(f"✈️ **{pred['away_team']}**")
                st.write(f"- Fora: {pred['away_team_stats']['away_over_rate']*100:.1f}%")
                st.write(f"- Força Ataque:
