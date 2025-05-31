import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import time

# Configuração da página
st.set_page_config(
    page_title="⚽ Over 0.5 HT Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2a5298;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .high-confidence {
        border-left: 5px solid #4CAF50;
        background: #f8fff8;
    }
    .medium-confidence {
        border-left: 5px solid #FF9800;
        background: #fff8f0;
    }
    .low-confidence {
        border-left: 5px solid #f44336;
        background: #fff0f0;
    }
</style>
""", unsafe_allow_html=True)

class FootballAPI:
    """Classe para gerenciar API do Football"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {'X-Auth-Token': api_key}
    
    def get_matches_by_date(self, date):
        """Busca jogos por data"""
        date_str = date.strftime('%Y-%m-%d')
        url = f"{self.base_url}/matches"
        params = {'dateFrom': date_str, 'dateTo': date_str}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                return response.json()['matches']
            else:
                st.error(f"Erro na API: {response.status_code}")
                return []
        except Exception as e:
            st.error(f"Erro de conexão: {e}")
            return []
    
    def get_historical_data(self, days_back=90):
        """Busca dados históricos"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        all_matches = []
        current_date = start_date
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while current_date <= end_date:
            status_text.text(f"Buscando dados: {current_date.strftime('%Y-%m-%d')}")
            
            matches = self.get_matches_by_date(current_date)
            all_matches.extend(matches)
            
            current_date += timedelta(days=1)
            progress = (current_date - start_date).days / (end_date - start_date).days
            progress_bar.progress(min(progress, 1.0))
            
            time.sleep(0.1)  # Rate limiting
        
        progress_bar.empty()
        status_text.empty()
        
        return all_matches

class MLPredictor:
    """Sistema de Machine Learning para previsões"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.performance = {}
        self.feature_names = []
    
    def process_historical_data(self, matches):
        """Processa dados históricos para treinamento"""
        processed_data = []
        
        for match in matches:
            if (match['status'] == 'FINISHED' and 
                match['score']['halfTime'] and 
                match['score']['fullTime']):
                
                match_data = {
                    'date': match['utcDate'][:10],
                    'competition': match['competition']['name'],
                    'home_team': match['homeTeam']['name'],
                    'away_team': match['awayTeam']['name'],
                    'home_team_id': match['homeTeam']['id'],
                    'away_team_id': match['awayTeam']['id'],
                    'ht_home': match['score']['halfTime']['home'] or 0,
                    'ht_away': match['score']['halfTime']['away'] or 0,
                    'ft_home': match['score']['fullTime']['home'] or 0,
                    'ft_away': match['score']['fullTime']['away'] or 0,
                    'over_05_ht': 1 if (match['score']['halfTime']['home'] + match['score']['halfTime']['away']) > 0.5 else 0
                }
                processed_data.append(match_data)
        
        return pd.DataFrame(processed_data)
    
    def create_features(self, df):
        """Cria features para ML"""
        if df.empty:
            return df
        
        # Converter data
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        
        # Estatísticas por equipe (últimos 10 jogos)
        team_stats = {}
        
        for team_id in pd.concat([df['home_team_id'], df['away_team_id']]).unique():
            team_matches = df[(df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)].tail(10)
            
            if len(team_matches) > 0:
                # Gols marcados no HT quando joga em casa
                home_ht_goals = team_matches[team_matches['home_team_id'] == team_id]['ht_home'].mean()
                # Gols marcados no HT quando joga fora
                away_ht_goals = team_matches[team_matches['away_team_id'] == team_id]['ht_away'].mean()
                # Taxa de Over 0.5 HT
                over_rate = team_matches['over_05_ht'].mean()
                
                team_stats[team_id] = {
                    'avg_ht_goals_home': home_ht_goals if not pd.isna(home_ht_goals) else 0.5,
                    'avg_ht_goals_away': away_ht_goals if not pd.isna(away_ht_goals) else 0.5,
                    'over_rate': over_rate if not pd.isna(over_rate) else 0.5
                }
        
        # Adicionar features ao dataframe
        df['home_avg_ht_goals'] = df['home_team_id'].map(lambda x: team_stats.get(x, {}).get('avg_ht_goals_home', 0.5))
        df['away_avg_ht_goals'] = df['away_team_id'].map(lambda x: team_stats.get(x, {}).get('avg_ht_goals_away', 0.5))
        df['home_over_rate'] = df['home_team_id'].map(lambda x: team_stats.get(x, {}).get('over_rate', 0.5))
        df['away_over_rate'] = df['away_team_id'].map(lambda x: team_stats.get(x, {}).get('over_rate', 0.5))
        
        # Estatísticas da liga
        league_stats = df.groupby('competition').agg({
            'over_05_ht': 'mean',
            'ht_home': 'mean',
            'ht_away': 'mean'
        }).reset_index()
        league_stats.columns = ['competition', 'league_over_rate', 'league_avg_home', 'league_avg_away']
        
        df = df.merge(league_stats, on='competition', how='left')
        
        # Features finais
        feature_columns = [
            'day_of_week', 'month', 'home_avg_ht_goals', 'away_avg_ht_goals',
            'home_over_rate', 'away_over_rate', 'league_over_rate', 
            'league_avg_home', 'league_avg_away'
        ]
        
        # Preencher NaN
        for col in feature_columns:
            df[col] = df[col].fillna(df[col].mean() if not df[col].empty else 0.5)
        
        self.feature_names = feature_columns
        return df
    
    def train_models(self, df):
        """Treina modelos por liga"""
        if df.empty:
            st.error("Sem dados para treinamento!")
            return
        
        # Criar features
        df_features = self.create_features(df)
        
        leagues = df_features['competition'].unique()
        
        for league in leagues:
            league_data = df_features[df_features['competition'] == league]
            
            if len(league_data) < 30:  # Mínimo de jogos
                continue
            
            # Preparar dados
            X = league_data[self.feature_names]
            y = league_data['over_05_ht']
            
            if len(X) < 10:
                continue
            
            # Split temporal
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Modelo
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Avaliação
            if len(X_test) > 0:
                predictions = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, predictions)
            else:
                accuracy = 0.6
            
            # Salvar modelo
            self.models[league] = model
            self.scalers[league] = scaler
            self.performance[league] = {
                'accuracy': accuracy,
                'total_games': len(league_data),
                'over_rate': y.mean()
            }
    
    def predict_matches(self, matches_today, historical_df):
        """Faz previsões para jogos de hoje"""
        if not matches_today:
            return pd.DataFrame()
        
        predictions = []
        
        # Processar jogos de hoje
        today_df = pd.DataFrame([{
            'date': match['utcDate'][:10],
            'competition': match['competition']['name'],
            'home_team': match['homeTeam']['name'],
            'away_team': match['awayTeam']['name'],
            'home_team_id': match['homeTeam']['id'],
            'away_team_id': match['awayTeam']['id'],
            'match_id': match['id']
        } for match in matches_today])
        
        # Criar features para hoje
        combined_df = pd.concat([historical_df, today_df], ignore_index=True)
        combined_df = self.create_features(combined_df)
        
        # Pegar apenas jogos de hoje
        today_features = combined_df.tail(len(today_df))
        
        for idx, match in today_features.iterrows():
            league = match['competition']
            
            if league in self.models:
                model = self.models[league]
                scaler = self.scalers[league]
                
                # Preparar features
                features = match[self.feature_names].values.reshape(1, -1)
                features_scaled = scaler.transform(features)
                
                # Previsão
                probability = model.predict_proba(features_scaled)[0][1]
                prediction = 1 if probability > 0.6 else 0
                
                # Confiança
                if probability > 0.75 or probability < 0.25:
                    confidence = "Alta"
                elif probability > 0.65 or probability < 0.35:
                    confidence = "Média"
                else:
                    confidence = "Baixa"
                
                predictions.append({
                    'home_team': match['home_team'],
                    'away_team': match['away_team'],
                    'competition': league,
                    'probability': probability,
                    'prediction': "Over 0.5" if prediction else "Under 0.5",
                    'confidence': confidence,
                    'model_accuracy': self.performance[league]['accuracy']
                })
        
        return pd.DataFrame(predictions)

def initialize_session_state():
    """Inicializa estado da sessão"""
    if 'api_handler' not in st.session_state:
        st.session_state.api_handler = None
    if 'predictor' not in st.session_state:
        st.session_state.predictor = MLPredictor()
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = pd.DataFrame()
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False

def main():
    """Função principal"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚽ Sistema Over 0.5 HT com Machine Learning</h1>
        <p>Previsões inteligentes para primeiro tempo - Análise por Liga</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Inicializar
    initialize_session_state()
    
    # Sidebar
    st.sidebar.header("🔧 Configurações")
    
    # API Key
    api_key = st.sidebar.text_input(
        "🔑 API Key Football-Data.org:",
        type="password",
        help="Cole sua API key aqui"
    )
    
    if api_key:
        st.session_state.api_handler = FootballAPI(api_key)
        st.sidebar.success("✅ API conectada!")
    
    # Data
    selected_date = st.sidebar.date_input(
        "📅 Data para análise:",
        value=datetime.now().date()
    )
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "🤖 Treinar ML", "🔮 Previsões", "📊 Estatísticas"])
    
    with tab1:
        show_dashboard()
    
    with tab2:
        show_training_tab()
    
    with tab3:
        show_predictions_tab(selected_date)
    
    with tab4:
        show_statistics_tab()

def show_dashboard():
    """Dashboard principal"""
    st.header("📊 Dashboard Geral")
    
    if st.session_state.models_trained:
        performance = st.session_state.predictor.performance
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🏆 Ligas Analisadas", len(performance))
        
        with col2:
            avg_accuracy = np.mean([p['accuracy'] for p in performance.values()])
            st.metric("🎯 Precisão Média", f"{avg_accuracy:.1%}")
        
        with col3:
            total_games = sum([p['total_games'] for p in performance.values()])
            st.metric("⚽ Total de Jogos", total_games)
        
        # Gráfico de performance por liga
        if performance:
            df_perf = pd.DataFrame([
                {'Liga': league, 'Precisão': data['accuracy'], 'Jogos': data['total_games']}
                for league, data in performance.items()
            ])
            
            fig = px.bar(df_perf, x='Liga', y='Precisão', 
                        title="Precisão do Modelo por Liga",
                        color='Precisão', color_continuous_scale='viridis')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("🤖 Treine os modelos primeiro na aba 'Treinar ML'")

def show_training_tab():
    """Aba de treinamento"""
    st.header("🤖 Treinamento dos Modelos")
    
    if not st.session_state.api_handler:
        st.error("⚠️ Configure a API Key primeiro!")
        return
    
    st.info("📈 O sistema irá baixar dados históricos e treinar modelos separados para cada liga")
    
    col1, col2 = st.columns(2)
    
    with col1:
        days_back = st.number_input("📅 Dias de histórico", 30, 180, 90)
    
    with col2:
        st.write("")
        st.write("")
        if st.button("🚀 Iniciar Treinamento", type="primary"):
            with st.spinner("Baixando dados históricos..."):
                matches = st.session_state.api_handler.get_historical_data(days_back)
                
                if matches:
                    df = st.session_state.predictor.process_historical_data(matches)
                    st.session_state.historical_data = df
                    
                    st.success(f"✅ {len(df)} jogos processados!")
                    
                    with st.spinner("Treinando modelos..."):
                        st.session_state.predictor.train_models(df)
                        st.session_state.models_trained = True
                    
                    st.success("🎉 Modelos treinados com sucesso!")
                    st.rerun()
                else:
                    st.error("❌ Erro ao baixar dados!")
    
    # Mostrar progresso se já treinou
    if st.session_state.models_trained:
        st.subheader("📈 Resultado do Treinamento")
        
        performance = st.session_state.predictor.performance
        for league, perf in performance.items():
            with st.expander(f"🏆 {league}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Precisão", f"{perf['accuracy']:.1%}")
                with col2:
                    st.metric("Total Jogos", perf['total_games'])
                with col3:
                    st.metric("Taxa Over 0.5", f"{perf['over_rate']:.1%}")

def show_predictions_tab(selected_date):
    """Aba de previsões"""
    st.header(f"🔮 Previsões para {selected_date}")
    
    if not st.session_state.api_handler:
        st.error("⚠️ Configure a API Key primeiro!")
        return
    
    if not st.session_state.models_trained:
        st.error("🤖 Treine os modelos primeiro!")
        return
    
    if st.button("🔍 Buscar Jogos do Dia", type="primary"):
        with st.spinner("Buscando jogos..."):
            matches_today = st.session_state.api_handler.get_matches_by_date(selected_date)
            
            if matches_today:
                # Filtrar apenas jogos não finalizados
                upcoming_matches = [m for m in matches_today if m['status'] in ['SCHEDULED', 'TIMED']]
                
                if upcoming_matches:
                    predictions = st.session_state.predictor.predict_matches(
                        upcoming_matches, 
                        st.session_state.historical_data
                    )
                    
                    if not predictions.empty:
                        st.subheader(f"🎯 {len(predictions)} Jogos Analisados")
                        
                        # Filtrar apenas Over 0.5 com alta confiança
                        best_bets = predictions[
                            (predictions['prediction'] == 'Over 0.5') & 
                            (predictions['confidence'].isin(['Alta', 'Média']))
                        ].sort_values('probability', ascending=False)
                        
                        if not best_bets.empty:
                            st.subheader("🌟 Melhores Apostas do Dia")
                            
                            for _, bet in best_bets.iterrows():
                                confidence_class = bet['confidence'].lower().replace('é', 'e') + '-confidence'
                                
                                st.markdown(f"""
                                <div class="prediction-card {confidence_class}">
                                    <h4>⚽ {bet['home_team']} vs {bet['away_team']}</h4>
                                    <p><strong>Liga:</strong> {bet['competition']}</p>
                                    <p><strong>Previsão:</strong> {bet['prediction']} ({bet['probability']:.1%})</p>
                                    <p><strong>Confiança:</strong> {bet['confidence']}</p>
                                    <p><strong>Precisão do Modelo:</strong> {bet['model_accuracy']:.1%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("🤔 Nenhuma aposta de alta confiança encontrada hoje")
                        
                        # Mostrar todas as previsões
                        with st.expander("📋 Todas as Previsões"):
                            st.dataframe(predictions, use_container_width=True)
                    else:
                        st.info("📊 Nenhuma previsão disponível para hoje")
                else:
                    st.info("⏰ Nenhum jogo programado para hoje")
            else:
                st.warning("❌ Nenhum jogo encontrado para esta data")

def show_statistics_tab():
    """Aba de estatísticas"""
    st.header("📊 Análises Detalhadas")
    
    if st.session_state.historical_data.empty:
        st.info("📈 Treine os modelos primeiro para ver estatísticas")
        return
    
    df = st.session_state.historical_data
    
    # Estatísticas gerais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Jogos", len(df))
    with col2:
        st.metric("Taxa Over 0.5 HT", f"{df['over_05_ht'].mean():.1%}")
    with col3:
        st.metric("Média Gols HT", f"{(df['ht_home'] + df['ht_away']).mean():.2f}")
    with col4:
        st.metric("Ligas Analisadas", df['competition'].nunique())
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Taxa Over por liga
        league_stats = df.groupby('competition')['over_05_ht'].agg(['mean', 'count']).reset_index()
        league_stats = league_stats[league_stats['count'] >= 10]  # Mínimo 10 jogos
        
        fig = px.bar(league_stats, x='competition', y='mean',
                    title="Taxa Over 0.5 HT por Liga",
                    labels={'mean': 'Taxa Over 0.5', 'competition': 'Liga'})
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribuição de gols no HT
        df['total_ht_goals'] = df['ht_home'] + df['ht_away']
        fig = px.histogram(df, x='total_ht_goals', nbins=6,
                          title="Distribuição de Gols no 1º Tempo")
        st.plotly_chart(fig, use_container_width=True)
    
    # Análise temporal
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.isocalendar().week
    weekly_stats = df.groupby('week')['over_05_ht'].mean().reset_index()
    
    fig = px.line(weekly_stats, x='week', y='over_05_ht',
                  title="Tendência Semanal - Taxa Over 0.5 HT")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
