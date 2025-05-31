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

# CSS Neumorphism Personalizado
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Configuração global neumorphism */
    .stApp {
        background: #e0e5ec;
        font-family: 'Inter', sans-serif;
        color: #2d3748;
    }
    
    /* Header principal neumorphism */
    .main-header {
        background: #e0e5ec;
        padding: 3rem 2rem;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 15px 15px 30px #c8d0e7, -15px -15px 30px #f8ffff;
        color: #2d3748;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.05), transparent);
        animation: shine 4s infinite;
    }
    
    @keyframes shine {
        0% { left: -100%; }
        50%, 100% { left: 100%; }
    }
    
    .main-header h1 {
        color: #667eea;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(102, 126, 234, 0.1);
    }
    
    .main-header p {
        color: #718096;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    /* Cards neumorphism modernos */
    .metric-card {
        background: #e0e5ec;
        padding: 2rem 1.5rem;
        border-radius: 20px;
        box-shadow: 9px 9px 18px #c8d0e7, -9px -9px 18px #f8ffff;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        color: #2d3748;
        border: none;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, transparent 0%, rgba(102, 126, 234, 0.02) 50%, transparent 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-card:hover {
        box-shadow: inset 9px 9px 18px #c8d0e7, inset -9px -9px 18px #f8ffff;
        transform: scale(0.98);
    }
    
    .metric-card h1 {
        color: #667eea;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0.5rem 0;
        text-shadow: 1px 1px 2px rgba(102, 126, 234, 0.1);
    }
    
    .metric-card h3 {
        color: #4a5568;
        font-size: 1rem;
        font-weight: 600;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Cards de previsão estilo neumorphism limpo */
    .prediction-card {
        background: #e0e5ec;
        color: #2d3748;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 9px 9px 18px #c8d0e7, -9px -9px 18px #f8ffff;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: none;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(to bottom, #667eea, #38ef7d);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .prediction-card:hover::before {
        opacity: 1;
    }
    
    .prediction-card:hover {
        box-shadow: inset 9px 9px 18px #c8d0e7, inset -9px -9px 18px #f8ffff;
        transform: translateY(-2px);
    }
    
    /* Estilo para o cabeçalho da previsão */
    .prediction-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(200, 208, 231, 0.3);
    }
    
    .team-info h3 {
        color: #2d3748;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }
    
    .league-info {
        color: #667eea;
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    .time-info {
        color: #718096;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    /* Confidence badge neumorphism */
    .confidence-badge {
        background: #e0e5ec;
        padding: 0.8rem 1.5rem;
        border-radius: 15px;
        font-weight: 700;
        font-size: 1.1rem;
        box-shadow: inset 3px 3px 6px #c8d0e7, inset -3px -3px 6px #f8ffff;
        color: #667eea;
        min-width: 80px;
        text-align: center;
    }
    
    /* Seção de análise limpa */
    .analysis-section {
        background: rgba(102, 126, 234, 0.02);
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 1rem;
        box-shadow: inset 2px 2px 4px rgba(200, 208, 231, 0.3);
    }
    
    .analysis-title {
        color: #667eea;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .analysis-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 1rem;
    }
    
    .analysis-item {
        background: #e0e5ec;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 3px 3px 6px #c8d0e7, -3px -3px 6px #f8ffff;
        text-align: center;
    }
    
    .analysis-item-label {
        color: #718096;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        margin-bottom: 0.3rem;
    }
    
    .analysis-item-value {
        color: #2d3748;
        font-size: 1.1rem;
        font-weight: 700;
    }
    
    /* Seção de probabilidades */
    .probability-section {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(200, 208, 231, 0.3);
    }
    
    .prediction-result {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2d3748;
    }
    
    .probability-text {
        color: #718096;
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    /* Lista de jogos estilo neumorphism */
    .games-list {
        background: #e0e5ec;
        border-radius: 20px;
        padding: 1.5rem;
        margin-top: 2rem;
        box-shadow: 9px 9px 18px #c8d0e7, -9px -9px 18px #f8ffff;
    }
    
    .games-list h3 {
        color: #667eea;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .game-item {
        background: #e0e5ec;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        box-shadow: 3px 3px 6px #c8d0e7, -3px -3px 6px #f8ffff;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.3s ease;
    }
    
    .game-item:hover {
        box-shadow: inset 3px 3px 6px #c8d0e7, inset -3px -3px 6px #f8ffff;
        transform: scale(0.99);
    }
    
    .game-teams {
        font-weight: 600;
        color: #2d3748;
        font-size: 0.95rem;
    }
    
    .game-confidence {
        background: #e0e5ec;
        padding: 0.4rem 0.8rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.9rem;
        box-shadow: inset 2px 2px 4px #c8d0e7, inset -2px -2px 4px #f8ffff;
        color: #667eea;
    }
    
    /* Sidebar neumorphism */
    .css-1d391kg {
        background: #e0e5ec;
        box-shadow: inset 3px 3px 6px #c8d0e7, inset -3px -3px 6px #f8ffff;
        border-right: none;
    }
    
    /* Botões neumorphism */
    .stButton > button {
        background: #e0e5ec;
        color: #667eea;
        border: none;
        border-radius: 15px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        box-shadow: 6px 6px 12px #c8d0e7, -6px -6px 12px #f8ffff;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        box-shadow: inset 6px 6px 12px #c8d0e7, inset -6px -6px 12px #f8ffff;
        transform: scale(0.98);
        color: #5a6fd8;
    }
    
    /* Tabs neumorphism */
    .stTabs [data-baseweb="tab-list"] {
        background: #e0e5ec;
        border-radius: 15px;
        box-shadow: inset 3px 3px 6px #c8d0e7, inset -3px -3px 6px #f8ffff;
        border: none;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #2d3748;
        border-radius: 10px;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: #e0e5ec;
        box-shadow: 3px 3px 6px #c8d0e7, -3px -3px 6px #f8ffff;
        color: #667eea;
    }
    
    /* Cards de liga */
    .league-card {
        background: #e0e5ec;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 6px 6px 12px #c8d0e7, -6px -6px 12px #f8ffff;
        border: none;
        transition: all 0.3s ease;
    }
    
    .league-card:hover {
        box-shadow: inset 6px 6px 12px #c8d0e7, inset -6px -6px 12px #f8ffff;
    }
    
    .league-card h4 {
        color: #667eea;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Classes de precisão */
    .accuracy-high {
        color: #38ef7d;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(56, 239, 125, 0.2);
    }
    
    .accuracy-medium {
        color: #f093fb;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(240, 147, 251, 0.2);
    }
    
    .accuracy-low {
        color: #ff6b6b;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(255, 107, 107, 0.2);
    }
    
    /* Responsividade */
    @media (max-width: 768px) {
        .main-header {
            padding: 2rem 1rem;
            border-radius: 15px;
        }
        
        .main-header h1 {
            font-size: 2rem;
        }
        
        .metric-card, .prediction-card {
            border-radius: 15px;
            padding: 1.5rem;
        }
        
        .prediction-header {
            flex-direction: column;
            gap: 1rem;
            text-align: center;
        }
        
        .analysis-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .game-item {
            flex-direction: column;
            gap: 0.5rem;
            text-align: center;
        }
    }
</style>
""", unsafe_allow_html=True)

# Todas as funções existentes permanecem iguais...
# (get_api_headers, check_api_status, get_fixtures, etc. - mantém tudo igual)

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
            
            if 'errors' in data and data['errors']:
                st.error(f"Erro da API: {data['errors']}")
                return []
            
            fixtures = data.get('response', [])
            return fixtures
        else:
            st.error(f"Erro API: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Erro de conexão: {str(e)}")
        return []

# [INCLUIR TODAS AS OUTRAS FUNÇÕES EXISTENTES - load_historical_data, extract_match_features, etc.]
# Mantenha todas as funções de ML iguais...

# FUNÇÃO NOVA: Exibir card de previsão limpo
def display_prediction_card_clean(pred, index):
    """Exibe card de previsão no estilo limpo neumorphism"""
    try:
        utc_time = datetime.strptime(pred['kickoff'][:16], '%Y-%m-%dT%H:%M')
        hora_portugal = utc_time.strftime('%H:%M')
    except:
        hora_portugal = pred['kickoff'][11:16]
    
    confidence_class = "accuracy-high" if pred['confidence'] > 75 else "accuracy-medium"
    
    # Análise avançada limpa
    advanced_info = ""
    if 'advanced_features' in pred:
        adv = pred['advanced_features']
        advanced_info = f"""
        <div class="analysis-section">
            <div class="analysis-title">
                🧠 Análise Avançada
            </div>
            <div class="analysis-grid">
                <div class="analysis-item">
                    <div class="analysis-item-label">Consistência Casa</div>
                    <div class="analysis-item-value">{adv.get('home_consistency', 0):.2f}</div>
                </div>
                <div class="analysis-item">
                    <div class="analysis-item-label">Consistência Fora</div>
                    <div class="analysis-item-value">{adv.get('away_consistency', 0):.2f}</div>
                </div>
                <div class="analysis-item">
                    <div class="analysis-item-label">Combined Score</div>
                    <div class="analysis-item-value">{adv.get('combined_score', 0):.3f}</div>
                </div>
                <div class="analysis-item">
                    <div class="analysis-item-label">Momentum Casa</div>
                    <div class="analysis-item-value">{adv.get('home_momentum', 0):.0%}</div>
                </div>
                <div class="analysis-item">
                    <div class="analysis-item-label">Momentum Fora</div>
                    <div class="analysis-item-value">{adv.get('away_momentum', 0):.0%}</div>
                </div>
                <div class="analysis-item">
                    <div class="analysis-item-label">Taxa de Acerto</div>
                    <div class="analysis-item-value">{pred['confidence']:.1f}%</div>
                </div>
            </div>
        </div>
        """
    
    st.markdown(f"""
    <div class="prediction-card">
        <div class="prediction-header">
            <div class="team-info">
                <h3>⚽ {pred['home_team']} vs {pred['away_team']}</h3>
                <div class="league-info">🏆 {pred['league']} ({pred['country']})</div>
                <div class="time-info">🕐 {hora_portugal}</div>
            </div>
            <div class="confidence-badge">{pred['confidence']:.1f}%</div>
        </div>
        
        {advanced_info}
        
        <div class="probability-section">
            <div class="prediction-result">
                🎯 Previsão: {pred['prediction']}
            </div>
            <div class="probability-text">
                Over {pred['probability_over']:.1f}% | Under {pred['probability_under']:.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# FUNÇÃO NOVA: Lista completa de jogos
def display_all_games_list(predictions):
    """Exibe lista de todos os jogos com confiança"""
    if not predictions:
        return
    
    st.markdown("""
    <div class="games-list">
        <h3>📋 Todos os Jogos do Dia</h3>
    """, unsafe_allow_html=True)
    
    for pred in predictions:
        try:
            utc_time = datetime.strptime(pred['kickoff'][:16], '%Y-%m-%dT%H:%M')
            hora = utc_time.strftime('%H:%M')
        except:
            hora = pred['kickoff'][11:16]
        
        prediction_icon = "🟢" if pred['prediction'] == 'OVER 0.5' else "🔴"
        
        st.markdown(f"""
        <div class="game-item">
            <div class="game-teams">
                {prediction_icon} {pred['home_team']} vs {pred['away_team']} ({hora})
            </div>
            <div class="game-confidence">
                {pred['confidence']:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# [INCLUIR TODAS AS OUTRAS FUNÇÕES DE ML - prepare_ml_features, train_ml_model, etc.]
# Mantenha todas iguais...

def main():
    # Header principal MODIFICADO
    st.markdown("""
    <div class="main-header">
        <h1>⚽ HT Goals AI Engine</h1>
        <p>🚀 Powered by Predictive Modeling & Advanced Metrics</p>
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
        
        # Data selecionada
        selected_date = st.date_input(
            "📅 Data para análise:",
            value=datetime.now().date()
        )
        
        # Configurações ML
        st.subheader("🤖 Machine Learning Avançado")
        
        days_training = st.slider(
            "📊 Dias para treinamento:",
            min_value=15,
            max_value=730,
            value=365
        )
        
        use_cache = st.checkbox(
            "💾 Usar dados em cache",
            value=True,
            help="Usar dados históricos salvos localmente"
        )
        
        # Mostrar features do modelo
        st.subheader("🧠 Features Avançadas")
        st.info("""
        ✅ **Coeficiente de Variação**
        ✅ **Combined Score**
        ✅ **Momentum Analysis**
        ✅ **Outlier Detection**
        ✅ **League Consistency**
        """)
        
        # Status do modelo (mantém igual)
        # [INCLUIR CÓDIGO DO STATUS DO MODELO...]
    
    # Tabs principais
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Previsões do Dia",
        "📊 Análise por Liga", 
        "🤖 Treinar Modelo Avançado",
        "📈 Performance ML"
    ])
    
    with tab1:
        st.header(f"🎯 Previsões para {selected_date.strftime('%d/%m/%Y')}")
        
        # [INCLUIR CÓDIGO DE VERIFICAÇÃO DO MODELO...]
        
        # Após obter as previsões, usar as novas funções:
        if predictions:
            # Métricas resumo (mantém igual)
            col1, col2, col3, col4 = st.columns(4)
            
            total_games = len(predictions)
            high_confidence = len([p for p in predictions if p['confidence'] > 70])
            over_predictions = len([p for p in predictions if p['prediction'] == 'OVER 0.5'])
            avg_confidence = sum([p['confidence'] for p in predictions]) / len(predictions)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎮 Total de Jogos</h3>
                    <h1>{total_games}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎯 Alta Confiança</h3>
                    <h1>{high_confidence}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📈 Over 0.5</h3>
                    <h1>{over_predictions}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💯 Confiança Média</h3>
                    <h1>{avg_confidence:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            # Top previsões com nova função
            st.subheader("🏆 Melhores Apostas (Análise Avançada)")
            
            best_bets = [p for p in predictions if p['prediction'] == 'OVER 0.5' and p['confidence'] > 65]
            best_bets.sort(key=lambda x: x['confidence'], reverse=True)
            
            if best_bets:
                for i, pred in enumerate(best_bets[:5]):  # Mostrar top 5
                    display_prediction_card_clean(pred, i)
            else:
                st.info("🤷 Nenhuma aposta OVER 0.5 com boa confiança encontrada hoje")
            
            # NOVA SEÇÃO: Lista completa de jogos
            display_all_games_list(predictions)
    
    # [INCLUIR AS OUTRAS TABS IGUAIS...]

if __name__ == "__main__":
    main()
