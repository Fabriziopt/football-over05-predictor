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

# SUA API Key - API-Football oficial
API_KEY = "2aad0db0e5b88b3a080bdc85461a919"

# Diretório para salvar modelos
MODEL_DIR = "models"
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

# Cache para armazenar dados
@st.cache_data(ttl=3600)
def get_fixtures_cached(date_str):
    """Busca jogos com cache de 1 hora"""
    return get_fixtures(date_str)

def get_fixtures(date_str):
    """Busca jogos da API-Football"""
    headers = {
        'x-apisports-key': API_KEY
    }
    
    try:
        response = requests.get(
            'https://v3.football.api-sports.io/fixtures',
            headers=headers,
            params={'date': date_str},
            timeout=20
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get('response', [])
        else:
            st.error(f"Erro API: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Erro de conexão: {str(e)}")
        return []

def collect_historical_data(days=30):
    """Coleta dados históricos para ML"""
    all_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
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
    """Prepara features para o modelo ML"""
    # Estatísticas por time
    team_stats = {}
    
    for idx, row in df.iterrows():
        # Stats time da casa
        home_team = row['home_team_id']
        if home_team not in team_stats:
            team_stats[home_team] = {
                'games': 0, 'over_05': 0, 'goals_scored': 0, 'goals_conceded': 0,
                'home_games': 0, 'home_over': 0, 'home_goals': 0
            }
        
        # Stats time visitante
        away_team = row['away_team_id']
        if away_team not in team_stats:
            team_stats[away_team] = {
                'games': 0, 'over_05': 0, 'goals_scored': 0, 'goals_conceded': 0,
                'away_games': 0, 'away_over': 0, 'away_goals': 0
            }
    
    # Calcular features
    features = []
    
    for idx, row in df.iterrows():
        home_id = row['home_team_id']
        away_id = row['away_team_id']
        
        # Features do time da casa
        home_stats = team_stats.get(home_id, {})
        home_over_rate = home_stats.get('over_05', 0) / max(home_stats.get('games', 1), 1)
        home_avg_goals = home_stats.get('goals_scored', 0) / max(home_stats.get('games', 1), 1)
        home_home_over_rate = home_stats.get('home_over', 0) / max(home_stats.get('home_games', 1), 1)
        
        # Features do time visitante
        away_stats = team_stats.get(away_id, {})
        away_over_rate = away_stats.get('over_05', 0) / max(away_stats.get('games', 1), 1)
        away_avg_goals = away_stats.get('goals_scored', 0) / max(away_stats.get('games', 1), 1)
        away_away_over_rate = away_stats.get('away_over', 0) / max(away_stats.get('away_games', 1), 1)
        
        # Features da liga
        league_games = df[df['league_id'] == row['league_id']]
        league_over_rate = league_games['over_05'].mean() if len(league_games) > 0 else 0.5
        
        feature_row = {
            'home_over_rate': home_over_rate,
            'home_avg_goals': home_avg_goals,
            'home_home_over_rate': home_home_over_rate,
            'away_over_rate': away_over_rate,
            'away_avg_goals': away_avg_goals,
            'away_away_over_rate': away_away_over_rate,
            'league_over_rate': league_over_rate,
            'combined_over_rate': (home_over_rate + away_over_rate) / 2,
            'combined_goals': home_avg_goals + away_avg_goals,
            'target': row['over_05']
        }
        
        features.append(feature_row)
        
        # Atualizar stats após o jogo
        team_stats[home_id]['games'] += 1
        team_stats[home_id]['over_05'] += row['over_05']
        team_stats[home_id]['goals_scored'] += row['ht_home_goals']
        team_stats[home_id]['goals_conceded'] += row['ht_away_goals']
        team_stats[home_id]['home_games'] += 1
        team_stats[home_id]['home_over'] += row['over_05']
        team_stats[home_id]['home_goals'] += row['ht_home_goals']
        
        team_stats[away_id]['games'] += 1
        team_stats[away_id]['over_05'] += row['over_05']
        team_stats[away_id]['goals_scored'] += row['ht_away_goals']
        team_stats[away_id]['goals_conceded'] += row['ht_home_goals']
        team_stats[away_id]['away_games'] += 1
        team_stats[away_id]['away_over'] += row['over_05']
        team_stats[away_id]['away_goals'] += row['ht_away_goals']
    
    return pd.DataFrame(features), team_stats

def train_ml_model(df):
    """Treina o modelo de ML"""
    # Preparar features
    features_df, team_stats = prepare_ml_features(df)
    
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
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = 0
    results = {}
    
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
        
        if test_f1 > best_score:
            best_score = test_f1
            best_model = model
    
    # Salvar melhor modelo
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'team_stats': team_stats,
        'results': results,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(df)
    }
    
    # Salvar modelo
    model_path = os.path.join(MODEL_DIR, f"model_{datetime.now().strftime('%Y%m%d')}.pkl")
    joblib.dump(model_data, model_path)
    
    return model_data, results

def load_latest_model():
    """Carrega o modelo mais recente"""
    try:
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
        if model_files:
            latest_model = sorted(model_files)[-1]
            return joblib.load(os.path.join(MODEL_DIR, latest_model))
    except:
        pass
    return None

def predict_matches(fixtures, model_data):
    """Faz previsões para os jogos do dia"""
    predictions = []
    
    if not model_data:
        return predictions
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = model_data['feature_cols']
    team_stats = model_data['team_stats']
    
    for fixture in fixtures:
        # Apenas jogos não iniciados
        if fixture['fixture']['status']['short'] not in ['NS', 'TBD']:
            continue
        
        try:
            home_id = fixture['teams']['home']['id']
            away_id = fixture['teams']['away']['id']
            
            # Verificar se temos dados dos times
            if home_id not in team_stats or away_id not in team_stats:
                continue
            
            # Preparar features
            home_stats = team_stats[home_id]
            away_stats = team_stats[away_id]
            
            features = {
                'home_over_rate': home_stats['over_05'] / max(home_stats['games'], 1),
                'home_avg_goals': home_stats['goals_scored'] / max(home_stats['games'], 1),
                'home_home_over_rate': home_stats['home_over'] / max(home_stats['home_games'], 1),
                'away_over_rate': away_stats['over_05'] / max(away_stats['games'], 1),
                'away_avg_goals': away_stats['goals_scored'] / max(away_stats['games'], 1),
                'away_away_over_rate': away_stats['away_over'] / max(away_stats['away_games'], 1),
                'league_over_rate': 0.5,  # Média geral
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
            
            prediction = {
                'fixture': fixture,
                'home_team': fixture['teams']['home']['name'],
                'away_team': fixture['teams']['away']['name'],
                'league': fixture['league']['name'],
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
            max_value=60,
            value=30
        )
        
        # Status do modelo
        model_data = load_latest_model()
        if model_data:
            st.success("✅ Modelo carregado")
            st.info(f"📅 Treinado em: {model_data['training_date']}")
            st.info(f"📊 Amostras: {model_data['total_samples']}")
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
                    
                    # Top previsões
                    st.subheader("🏆 Melhores Apostas do Dia")
                    
                    for i, pred in enumerate(predictions[:10]):
                        if pred['confidence'] > 65:  # Apenas alta confiança
                            confidence_class = "accuracy-high" if pred['confidence'] > 75 else "accuracy-medium"
                            
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h3>⚽ {pred['home_team']} vs {pred['away_team']}</h3>
                                <p><strong>🏆 Liga:</strong> {pred['league']} ({pred['country']})</p>
                                <p><strong>⏰ Horário:</strong> {pred['kickoff'][11:16]}</p>
                                <hr style="opacity: 0.3;">
                                <p><strong>🎯 Previsão ML:</strong> {pred['prediction']}</p>
                                <p><strong>💯 Confiança:</strong> <span class="{confidence_class}">{pred['confidence']:.1f}%</span></p>
                                <p><strong>📊 Probabilidades:</strong> Over {pred['probability_over']:.1f}% | Under {pred['probability_under']:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Todas as previsões
                    with st.expander("📋 Ver Todas as Previsões"):
                        pred_df = pd.DataFrame([{
                            'Hora': p['kickoff'][11:16],
                            'Casa': p['home_team'],
                            'Fora': p['away_team'],
                            'Liga': p['league'],
                            'Previsão': p['prediction'],
                            'Confiança': f"{p['confidence']:.1f}%"
                        } for p in predictions])
                        
                        st.dataframe(pred_df, use_container_width=True)
                
                else:
                    st.info("🤷 Nenhuma previsão disponível (times sem dados históricos)")
    
    with tab2:
        st.header("📊 Análise de Ligas")
        
        if model_data:
            # Carregar dados históricos para análise
            df = collect_historical_data(days=15)
            
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
        
        if st.button("🚀 Iniciar Treinamento", type="primary"):
            # Coletar dados
            with st.spinner(f"📊 Coletando {days_training} dias de dados históricos..."):
                df = collect_historical_data(days=days_training)
            
            if df.empty:
                st.error("❌ Não foi possível coletar dados")
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
            
            # Explicação das métricas
            with st.expander("📚 Entenda as Métricas"):
                st.write("""
                - **Acurácia**: Percentual total de acertos
                - **Precisão**: Quando prevê OVER, quantas vezes acerta
                - **Recall**: Dos jogos que foram OVER, quantos o modelo identificou
                - **F1-Score**: Média harmônica entre Precisão e Recall (métrica principal)
                """)
            
            # Informações do modelo
            st.subheader("ℹ️ Informações do Modelo")
            st.info(f"""
            - **Data de Treinamento**: {model_data['training_date']}
            - **Total de Jogos Analisados**: {model_data['total_samples']}
            - **Times no Banco de Dados**: {len(model_data['team_stats'])}
            """)
        else:
            st.info("🤖 Nenhum modelo treinado ainda")

if __name__ == "__main__":
    main()
