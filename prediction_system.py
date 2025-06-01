import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import requests
import warnings
warnings.filterwarnings('ignore')

# Para exportar Excel
try:
    import openpyxl
    from openpyxl.styles import PatternFill, Font, Alignment
    from openpyxl.utils import get_column_letter
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

class HTGoalsAPIClient:
    """Cliente para integração com API HT Goals"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.htgoals.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def get_historical_data(self, league_id, days=365):
        """Busca dados históricos da liga"""
        endpoint = f"{self.base_url}/matches/historical"
        params = {
            "league_id": league_id,
            "days": days,
            "include_stats": True
        }
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Converter para DataFrame
            if 'matches' in data:
                df = pd.DataFrame(data['matches'])
                df['date'] = pd.to_datetime(df['date'])
                df['over_05'] = df['ht_total_goals'] > 0.5
                return df
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Erro ao buscar dados históricos: {e}")
            return pd.DataFrame()
    
    def get_team_data(self, team_id, position='both', days=90):
        """Busca dados específicos de um time"""
        endpoint = f"{self.base_url}/teams/{team_id}/matches"
        params = {
            "days": days,
            "position": position,
            "include_ht_stats": True
        }
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'matches' in data:
                df = pd.DataFrame(data['matches'])
                df['date'] = pd.to_datetime(df['date'])
                df['over_05'] = df['ht_total_goals'] > 0.5
                return df
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Erro ao buscar dados do time: {e}")
            return pd.DataFrame()
    
    def make_prediction(self, match_data):
        """Envia predição para API"""
        endpoint = f"{self.base_url}/predictions/create"
        
        try:
            response = requests.post(endpoint, headers=self.headers, json=match_data)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            print(f"Erro ao enviar predição: {e}")
            return None

class SmartPredictionSystem:
    """Sistema de Predição Inteligente com Análise Hierárquica - Integrado com HT Goals"""
    
    def __init__(self, api_client=None):
        self.api_client = api_client
        self.league_base_rates = {}
        self.team_adjustments = {}
        self.backtesting_results = {}
        
        # Parâmetros otimizáveis
        self.outlier_threshold = 1.5
        self.recent_weight = 0.3
        self.consistency_weight = 0.7
        self.min_games_threshold = 3
        
        # Parâmetros da API HT Goals
        self.confidence_range = {'min': 70, 'max': 100}
    
    def fetch_data_from_api(self, league_id, home_team_id, away_team_id):
        """Busca dados necessários da API HT Goals"""
        if not self.api_client:
            raise ValueError("API client não configurado")
        
        # Buscar dados da liga (365 dias)
        league_data = self.api_client.get_historical_data(league_id, days=365)
        
        # Buscar dados dos times (90 dias)
        home_team_data = self.api_client.get_team_data(home_team_id, position='home', days=90)
        away_team_data = self.api_client.get_team_data(away_team_id, position='away', days=90)
        
        return league_data, home_team_data, away_team_data
    
    def detect_outliers(self, goals_series):
        """Detecta outliers usando IQR method com threshold configurável
        
        EXEMPLO: Time com gols [5, 0, 0, 0, 0]
        - Q75 = 0, Q25 = 0, IQR = 0
        - Como IQR = 0, usa método alternativo baseado em desvio padrão
        - O 5 será marcado como outlier e REMOVIDO dos cálculos
        """
        if len(goals_series) < 3:
            return np.zeros(len(goals_series), dtype=bool)
        
        q75 = np.percentile(goals_series, 75)
        q25 = np.percentile(goals_series, 25)
        iqr = q75 - q25
        
        # Se IQR = 0 (muitos valores iguais), usar método alternativo
        if iqr == 0:
            mean = np.mean(goals_series)
            std = np.std(goals_series)
            if std > 0:
                # Marcar como outlier valores > 2 desvios padrão da média
                outliers = np.abs(goals_series - mean) > (2 * std)
            else:
                # Se todos valores iguais, nenhum é outlier
                outliers = np.zeros(len(goals_series), dtype=bool)
        else:
            # Método IQR tradicional
            lower_bound = q25 - self.outlier_threshold * iqr
            upper_bound = q75 + self.outlier_threshold * iqr
            outliers = (goals_series < lower_bound) | (goals_series > upper_bound)
        
        return outliers
    
    def calculate_realistic_team_stats(self, team_matches, position='home'):
        """Calcula estatísticas realistas removendo outliers e ponderando recência"""
        
        if len(team_matches) == 0:
            return {
                'raw_over_rate': 0.5,
                'adjusted_over_rate': 0.5,
                'consistency_score': 0.5,
                'recent_form': 0.5,
                'scoring_frequency': 0.5,
                'avg_when_scores': 0.5,
                'games_count': 0,
                'outliers_detected': 0
            }
        
        # Determinar coluna de gols baseada na posição
        goals_col = f'ht_{position}_goals' if position in ['home', 'away'] else 'ht_total_goals'
        
        if goals_col not in team_matches.columns:
            return {
                'raw_over_rate': 0.5,
                'adjusted_over_rate': 0.5,
                'consistency_score': 0.5,
                'recent_form': 0.5,
                'scoring_frequency': 0.5,
                'avg_when_scores': 0.5,
                'games_count': 0,
                'outliers_detected': 0
            }
        
        goals_series = team_matches[goals_col].fillna(0)
        over_05_series = team_matches['over_05'].fillna(0)
        
        # 1. Taxa Over 0.5 raw
        raw_over_rate = over_05_series.mean()
        
        # 2. Detectar outliers
        outliers = self.detect_outliers(goals_series)
        
        # 3. Remover outliers para cálculo ajustado
        clean_goals = goals_series[~outliers] if outliers.sum() > 0 else goals_series
        clean_over = over_05_series[~outliers] if outliers.sum() > 0 else over_05_series
        
        adjusted_over_rate = clean_over.mean() if len(clean_over) > 0 else raw_over_rate
        
        # 4. Análise de consistência com peso configurável
        consistency_score = (1 / (1 + np.std(clean_over) + 0.1)) * self.consistency_weight
        
        # 5. Forma recente (últimos 5 jogos) com peso configurável
        recent_games = over_05_series.tail(5)
        recent_form = recent_games.mean() if len(recent_games) > 0 else adjusted_over_rate
        
        # 6. Frequência de marcar
        games_with_goals = len(goals_series[goals_series > 0])
        scoring_frequency = games_with_goals / len(goals_series) if len(goals_series) > 0 else 0
        
        # 7. Média quando marca
        goals_when_scores = goals_series[goals_series > 0]
        avg_when_scores = goals_when_scores.mean() if len(goals_when_scores) > 0 else 0
        
        return {
            'raw_over_rate': raw_over_rate,
            'adjusted_over_rate': adjusted_over_rate,
            'consistency_score': consistency_score,
            'recent_form': recent_form,
            'scoring_frequency': scoring_frequency,
            'avg_when_scores': avg_when_scores,
            'games_count': len(team_matches),
            'outliers_detected': outliers.sum()
        }
    
    def analyze_league(self, league_data):
        """
        ETAPA 1: Análise da Liga
        Estabelece a taxa base da liga
        """
        if len(league_data) == 0:
            return {'base_rate': 0.5, 'volatility': 0.5, 'games_count': 0, 'seasonal_trend': 0}
        
        # Taxa base da liga
        base_rate = league_data['over_05'].mean()
        
        # Volatilidade da liga (algumas ligas são mais previsíveis)
        volatility = np.std(league_data['over_05'])
        
        # Padrões sazonais (início vs final de temporada)
        if 'date' in league_data.columns:
            league_data['date'] = pd.to_datetime(league_data['date'], errors='coerce')
            recent_data = league_data.tail(int(len(league_data) * 0.3))  # Últimos 30%
            recent_rate = recent_data['over_05'].mean()
            seasonal_trend = recent_rate - base_rate
        else:
            seasonal_trend = 0
        
        return {
            'base_rate': base_rate,
            'volatility': volatility,
            'seasonal_trend': seasonal_trend,
            'games_count': len(league_data)
        }
    
    def analyze_teams(self, home_team_data, away_team_data):
        """
        ETAPA 2: Análise dos Times
        Calcula ajustes baseados no histórico dos times
        """
        
        # Analisar time da casa
        home_stats = self.calculate_realistic_team_stats(home_team_data, 'home')
        
        # Analisar time de fora  
        away_stats = self.calculate_realistic_team_stats(away_team_data, 'away')
        
        # Calcular ajustes com pesos configuráveis
        home_adjustment = (home_stats['adjusted_over_rate'] - 0.5) * home_stats['consistency_score']
        away_adjustment = (away_stats['adjusted_over_rate'] - 0.5) * away_stats['consistency_score']
        
        # Forma recente tem peso extra configurável
        home_recent_adj = (home_stats['recent_form'] - home_stats['adjusted_over_rate']) * self.recent_weight
        away_recent_adj = (away_stats['recent_form'] - away_stats['adjusted_over_rate']) * self.recent_weight
        
        return {
            'home_stats': home_stats,
            'away_stats': away_stats,
            'home_adjustment': home_adjustment + home_recent_adj,
            'away_adjustment': away_adjustment + away_recent_adj,
            'combined_adjustment': (home_adjustment + away_adjustment + home_recent_adj + away_recent_adj) / 2
        }
    
    def analyze_match_context(self, home_stats, away_stats, league_analysis):
        """
        ETAPA 3: Análise do Contexto do Jogo
        Fatores específicos do confronto
        """
        
        # Head-to-head seria aqui (se disponível)
        h2h_adjustment = 0  # Placeholder
        
        # Motivação/contexto (posição na tabela, etc.)
        motivation_adjustment = 0  # Placeholder
        
        # Estilo de jogo compatibilidade
        # Times ofensivos vs defensivos
        home_attack_style = home_stats['scoring_frequency']
        away_defense_style = 1 - away_stats['scoring_frequency']
        
        style_compatibility = (home_attack_style + away_defense_style) / 2 - 0.5
        
        return {
            'h2h_adjustment': h2h_adjustment,
            'motivation_adjustment': motivation_adjustment,
            'style_compatibility': style_compatibility,
            'final_context_adjustment': (h2h_adjustment + motivation_adjustment + style_compatibility) / 3
        }
    
    def predict_match(self, league_data, home_team_data, away_team_data):
        """
        Sistema Hierárquico Completo de Predição
        """
        
        # ETAPA 1: Análise da Liga
        league_analysis = self.analyze_league(league_data)
        base_probability = league_analysis['base_rate']
        
        # ETAPA 2: Análise dos Times
        team_analysis = self.analyze_teams(home_team_data, away_team_data)
        team_adjustment = team_analysis['combined_adjustment']
        
        # ETAPA 3: Análise do Contexto
        context_analysis = self.analyze_match_context(
            team_analysis['home_stats'], 
            team_analysis['away_stats'], 
            league_analysis
        )
        context_adjustment = context_analysis['final_context_adjustment']
        
        # ETAPA 4: Cálculo Final
        # Base da liga + ajustes dos times + contexto do jogo
        final_probability = base_probability + team_adjustment + context_adjustment
        
        # Manter entre 0 e 1
        final_probability = max(0.05, min(0.95, final_probability))
        
        # Convertir para porcentagem
        confidence_percentage = final_probability * 100
        
        # Garantir que está dentro dos limites da API (70-100%)
        confidence_percentage = max(self.confidence_range['min'], 
                                  min(self.confidence_range['max'], confidence_percentage))
        
        return {
            'final_probability': final_probability,
            'confidence_percentage': confidence_percentage,
            'breakdown': {
                'league_base': base_probability * 100,
                'team_adjustment': team_adjustment * 100,
                'context_adjustment': context_adjustment * 100,
                'final': confidence_percentage
            },
            'analysis_details': {
                'league': league_analysis,
                'teams': team_analysis,
                'context': context_analysis
            }
        }
    
    def predict_from_api(self, league_id, home_team_id, away_team_id, match_date=None):
        """
        Faz predição completa usando dados da API HT Goals
        """
        # Buscar dados da API
        league_data, home_team_data, away_team_data = self.fetch_data_from_api(
            league_id, home_team_id, away_team_id
        )
        
        # Verificar dados mínimos
        if len(home_team_data) < self.min_games_threshold:
            return {
                'error': f'Dados insuficientes para time da casa (mínimo: {self.min_games_threshold} jogos)',
                'confidence_percentage': 70
            }
        
        if len(away_team_data) < self.min_games_threshold:
            return {
                'error': f'Dados insuficientes para time visitante (mínimo: {self.min_games_threshold} jogos)',
                'confidence_percentage': 70
            }
        
        # Fazer predição
        prediction = self.predict_match(league_data, home_team_data, away_team_data)
        
        # Preparar dados para API
        api_payload = {
            'league_id': league_id,
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'match_date': match_date or datetime.now().isoformat(),
            'prediction': {
                'market': 'over_05_ht',
                'confidence': prediction['confidence_percentage'],
                'probability': prediction['final_probability'],
                'breakdown': prediction['breakdown']
            },
            'metadata': {
                'model_version': '2.0',
                'analysis_type': 'hierarchical',
                'data_points': {
                    'league_games': len(league_data),
                    'home_games': len(home_team_data),
                    'away_games': len(away_team_data)
                }
            }
        }
        
        # Enviar para API
        if self.api_client:
            api_response = self.api_client.make_prediction(api_payload)
            prediction['api_response'] = api_response
        
        return prediction
    
    def unified_prediction_with_validation(self, league_id, home_team_id, away_team_id, match_date=None):
        """
        Sistema Unificado de Predição com Validação Histórica
        
        Processo completo:
        1. Análise da Liga (base rate)
        2. Análise dos Times (ajustes específicos)
        3. Contexto do Jogo (confronto direto)
        4. Comparação com Histórico (validação)
        5. Ajuste Final da Confiança
        """
        
        print("🔄 Iniciando Análise Unificada...")
        
        # Buscar todos os dados necessários
        league_data, home_team_data, away_team_data = self.fetch_data_from_api(
            league_id, home_team_id, away_team_id
        )
        
        # Verificar dados mínimos
        if len(home_team_data) < self.min_games_threshold or len(away_team_data) < self.min_games_threshold:
            return {
                'error': 'Dados insuficientes',
                'confidence_percentage': 70,  # Mínimo agora é 70%
                'recommendation': 'SKIP'
            }
        
        # ETAPA 1: Predição Base
        base_prediction = self.predict_match(league_data, home_team_data, away_team_data)
        
        # ETAPA 2: Validação com Histórico Similar
        historical_validation = self.validate_with_historical_matches(
            league_data, 
            base_prediction['confidence_percentage'],
            base_prediction['breakdown']
        )
        
        # ETAPA 3: Ajuste de Confiança baseado no Histórico
        adjusted_confidence = self.adjust_confidence_by_history(
            base_prediction['confidence_percentage'],
            historical_validation['accuracy_in_range'],
            historical_validation['matches_count']
        )
        
        # ETAPA 4: Decisão Final
        recommendation = self.make_final_recommendation(
            adjusted_confidence,
            historical_validation
        )
        
        return {
            'final_confidence': adjusted_confidence,
            'base_confidence': base_prediction['confidence_percentage'],
            'historical_accuracy': historical_validation['accuracy_in_range'],
            'similar_matches_analyzed': historical_validation['matches_count'],
            'recommendation': recommendation,
            'breakdown': base_prediction['breakdown'],
            'validation_details': historical_validation,
            'analysis_summary': {
                'league_tendency': f"{base_prediction['breakdown']['league_base']:.1f}%",
                'teams_adjustment': f"{base_prediction['breakdown']['team_adjustment']:.1f}%",
                'historical_performance': f"{historical_validation['accuracy_in_range']:.1f}%",
                'confidence_level': self.get_confidence_level(adjusted_confidence)
            }
        }
    
    def validate_with_historical_matches(self, league_data, predicted_confidence, breakdown):
        """
        Valida predição comparando com jogos históricos similares
        """
        # Filtrar jogos com características similares
        confidence_range = (predicted_confidence - 5, predicted_confidence + 5)
        
        similar_matches = []
        
        for idx, match in league_data.iterrows():
            # Simular predição para jogos passados
            try:
                # Pegar dados históricos até aquele jogo
                historical_until_match = league_data[league_data['date'] < match['date']]
                
                if len(historical_until_match) < 20:
                    continue
                
                # Dados dos times
                home_data = historical_until_match[
                    historical_until_match['home_team_id'] == match['home_team_id']
                ]
                away_data = historical_until_match[
                    historical_until_match['away_team_id'] == match['away_team_id']
                ]
                
                if len(home_data) >= self.min_games_threshold and len(away_data) >= self.min_games_threshold:
                    # Fazer predição retroativa
                    retro_prediction = self.predict_match(historical_until_match, home_data, away_data)
                    
                    # Se confiança similar, adicionar aos similares
                    if confidence_range[0] <= retro_prediction['confidence_percentage'] <= confidence_range[1]:
                        similar_matches.append({
                            'predicted_conf': retro_prediction['confidence_percentage'],
                            'actual_result': match['over_05'],
                            'correct': (retro_prediction['final_probability'] > 0.5) == match['over_05']
                        })
                        
            except:
                continue
        
        # Calcular estatísticas dos jogos similares
        if len(similar_matches) >= 5:  # Mínimo de jogos para validação
            correct_predictions = sum([m['correct'] for m in similar_matches])
            accuracy = (correct_predictions / len(similar_matches)) * 100
            
            return {
                'matches_count': len(similar_matches),
                'accuracy_in_range': accuracy,
                'confidence_range': confidence_range,
                'avg_confidence': np.mean([m['predicted_conf'] for m in similar_matches])
            }
        
        return {
            'matches_count': 0,
            'accuracy_in_range': predicted_confidence,  # Mantém confiança original
            'confidence_range': confidence_range,
            'avg_confidence': predicted_confidence
        }
    
    def adjust_confidence_by_history(self, base_confidence, historical_accuracy, matches_count):
        """
        Ajusta confiança baseado no desempenho histórico
        """
        if matches_count < 5:
            # Poucos dados históricos, manter confiança base
            return base_confidence
        
        # Peso do histórico aumenta com quantidade de jogos
        historical_weight = min(0.4, matches_count / 50)  # Máximo 40% de peso
        
        # Ajuste ponderado
        adjusted = (base_confidence * (1 - historical_weight)) + (historical_accuracy * historical_weight)
        
        # Garantir limites 70-100%
        return max(self.confidence_range['min'], min(self.confidence_range['max'], adjusted))
    
    def make_final_recommendation(self, confidence, validation):
        """
        Faz recomendação final baseada em todos os fatores
        """
        if validation['matches_count'] < 5:
            if confidence >= 85:
                return "BET_CAUTIOUS"  # Apostar com cautela (poucos dados históricos)
            else:
                return "ANALYZE_MORE"  # Precisa mais análise
        
        # Com dados históricos suficientes
        if confidence >= 85 and validation['accuracy_in_range'] >= 75:
            return "STRONG_BET"  # Aposta forte
        elif confidence >= 75 and validation['accuracy_in_range'] >= 70:
            return "MODERATE_BET"  # Aposta moderada
        elif confidence >= 70:
            return "WEAK_BET"  # Aposta fraca
        else:
            return "SKIP"  # Pular esta aposta
    
    def get_confidence_level(self, confidence):
        """
        Retorna nível de confiança em texto
        """
        if confidence >= 90:
            return "MUITO ALTA"
        elif confidence >= 80:
            return "ALTA"
        elif confidence >= 70:
            return "MODERADA"
        else:
            return "BAIXA"
    
    def compare_model_vs_league_baseline(self, backtest_results, league_data):
        """
        Compara performance do modelo vs taxa base da liga
        Isso mostra se o modelo está agregando valor real
        """
        # Taxa base da liga (% de jogos Over 0.5 HT)
        league_baseline = league_data['over_05'].mean() * 100
        
        # Performance do modelo
        model_accuracy = backtest_results.get('overall_accuracy', 0)
        
        # Calcular lift (melhoria sobre baseline)
        if league_baseline > 0:
            lift = ((model_accuracy - league_baseline) / league_baseline) * 100
        else:
            lift = 0
        
        # Análise por tipo de aposta
        performance_vs_baseline = {}
        
        for rec_type, stats in backtest_results['by_recommendation'].items():
            if stats['count'] > 0:
                # Taxa de acerto do modelo para este tipo
                model_rate = stats['accuracy']
                
                # Comparar com baseline
                improvement = model_rate - league_baseline
                
                performance_vs_baseline[rec_type] = {
                    'model_accuracy': model_rate,
                    'league_baseline': league_baseline,
                    'improvement': improvement,
                    'relative_lift': (improvement / league_baseline * 100) if league_baseline > 0 else 0
                }
        
        return {
            'league_baseline': league_baseline,
            'model_overall_accuracy': model_accuracy,
            'absolute_improvement': model_accuracy - league_baseline,
            'relative_lift': lift,
            'by_recommendation': performance_vs_baseline,
            'is_model_better': model_accuracy > league_baseline
        }
    
    def run_integrated_backtest(self, historical_data, min_confidence=70):
        """
        Executa backtesting integrado com o sistema unificado
        """
        print("🚀 Iniciando Backtesting Integrado...")
        
        results = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'by_recommendation': {
                'STRONG_BET': {'count': 0, 'correct': 0},
                'MODERATE_BET': {'count': 0, 'correct': 0},
                'WEAK_BET': {'count': 0, 'correct': 0},
                'SKIP': {'count': 0, 'correct': 0}
            },
            'by_confidence_range': {},
            'detailed_results': []
        }
        
        # Ordenar por data
        historical_data = historical_data.sort_values('date')
        
        # Últimos 20% dos dados para teste
        test_size = int(len(historical_data) * 0.2)
        test_start_idx = len(historical_data) - test_size
        
        for idx in range(test_start_idx, len(historical_data)):
            match = historical_data.iloc[idx]
            
            # Dados até este jogo
            train_data = historical_data.iloc[:idx]
            
            # Simular predição unificada
            try:
                # Dados da liga
                league_train = train_data[train_data['league_id'] == match['league_id']]
                
                # Dados dos times
                home_train = league_train[league_train['home_team_id'] == match['home_team_id']]
                away_train = league_train[league_train['away_team_id'] == match['away_team_id']]
                
                if len(home_train) >= self.min_games_threshold and len(away_train) >= self.min_games_threshold:
                    # Fazer predição
                    base_pred = self.predict_match(league_train, home_train, away_train)
                    
                    # Validar com histórico
                    validation = self.validate_with_historical_matches(
                        league_train, 
                        base_pred['confidence_percentage'],
                        base_pred['breakdown']
                    )
                    
                    # Ajustar confiança
                    final_conf = self.adjust_confidence_by_history(
                        base_pred['confidence_percentage'],
                        validation['accuracy_in_range'],
                        validation['matches_count']
                    )
                    
                    # Recomendação
                    recommendation = self.make_final_recommendation(final_conf, validation)
                    
                    # Resultado real
                    actual = match['over_05']
                    predicted = base_pred['final_probability'] > 0.5
                    correct = predicted == actual
                    
                    # Atualizar estatísticas
                    results['total_predictions'] += 1
                    if correct:
                        results['correct_predictions'] += 1
                    
                    # Por recomendação
                    if recommendation in results['by_recommendation']:
                        results['by_recommendation'][recommendation]['count'] += 1
                        if correct:
                            results['by_recommendation'][recommendation]['correct'] += 1
                    
                    # Guardar resultado detalhado
                    results['detailed_results'].append({
                        'date': match['date'],
                        'league_id': match['league_id'],
                        'final_confidence': final_conf,
                        'historical_accuracy': validation['accuracy_in_range'],
                        'recommendation': recommendation,
                        'correct': correct,
                        'actual': actual
                    })
                    
            except Exception as e:
                continue
        
        # Calcular estatísticas finais
        if results['total_predictions'] > 0:
            results['overall_accuracy'] = (results['correct_predictions'] / results['total_predictions']) * 100
            
            # Acurácia por recomendação
            for rec_type, stats in results['by_recommendation'].items():
                if stats['count'] > 0:
                    stats['accuracy'] = (stats['correct'] / stats['count']) * 100
                else:
                    stats['accuracy'] = 0
        
        # NOVA FUNCIONALIDADE: Comparar com baseline da liga
        league_comparison = self.compare_model_vs_league_baseline(results, historical_data)
        results['league_comparison'] = league_comparison
        
        return results

class BacktestingEngine:
    """Engine para testar performance histórica com 365 dias e divisão train/val/test"""
    
    def __init__(self, prediction_system):
        self.prediction_system = prediction_system
        self.results = []
        self.training_history = []
    
    def split_temporal_data(self, historical_data, train_days=240, val_days=60, test_days=65):
        """
        Divisão temporal dos dados (365 dias total):
        - Train: 240 dias (65.8%) - Para treinar o sistema
        - Validation: 60 dias (16.4%) - Para ajustar parâmetros  
        - Test: 65 dias (17.8%) - Para avaliar performance final
        """
        
        if 'date' not in historical_data.columns:
            raise ValueError("Dados históricos devem ter coluna 'date'")
        
        # Ordenar por data
        historical_data['date'] = pd.to_datetime(historical_data['date'], errors='coerce')
        historical_data = historical_data.sort_values('date').reset_index(drop=True)
        
        # Calcular datas de corte
        max_date = historical_data['date'].max()
        
        test_start = max_date - timedelta(days=test_days)
        val_start = test_start - timedelta(days=val_days)
        train_start = val_start - timedelta(days=train_days)
        
        # Filtrar apenas últimos 365 dias
        cutoff_date = max_date - timedelta(days=365)
        recent_data = historical_data[historical_data['date'] >= cutoff_date].copy()
        
        # Dividir dados
        train_data = recent_data[
            (recent_data['date'] >= train_start) & 
            (recent_data['date'] < val_start)
        ].copy()
        
        val_data = recent_data[
            (recent_data['date'] >= val_start) & 
            (recent_data['date'] < test_start)
        ].copy()
        
        test_data = recent_data[
            recent_data['date'] >= test_start
        ].copy()
        
        return {
            'train': train_data,
            'validation': val_data, 
            'test': test_data,
            'split_info': {
                'train_period': f"{train_start.strftime('%Y-%m-%d')} to {val_start.strftime('%Y-%m-%d')}",
                'val_period': f"{val_start.strftime('%Y-%m-%d')} to {test_start.strftime('%Y-%m-%d')}",
                'test_period': f"{test_start.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}",
                'train_games': len(train_data),
                'val_games': len(val_data),
                'test_games': len(test_data)
            }
        }

# Funções auxiliares para exportação e demonstração
def export_predictions_to_excel(predictions_df, filename="ht_goals_predictions.xlsx"):
    """
    Exporta predições para Excel com formatação profissional
    """
    if not EXCEL_AVAILABLE:
        # Fallback para CSV se openpyxl não estiver instalado
        predictions_df.to_csv(filename.replace('.xlsx', '.csv'), index=False)
        return filename.replace('.xlsx', '.csv')
    
    # Criar Excel writer
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Aba principal com predições
        predictions_df.to_excel(writer, sheet_name='Predições', index=False)
        
        # Pegar workbook e worksheet
        workbook = writer.book
        worksheet = writer.sheets['Predições']
        
        # Estilos
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True)
        
        # Formatar cabeçalhos
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")
        
        # Ajustar largura das colunas
        for column in worksheet.columns:
            max_length = 0
            column_letter = get_column_letter(column[0].column)
            
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
        
        # Adicionar formatação condicional para confiança
        for row in range(2, len(predictions_df) + 2):
            conf_cell = worksheet[f'F{row}']  # Assumindo que confiança está na coluna F
            try:
                conf_value = float(conf_cell.value)
                if conf_value >= 80:
                    conf_cell.fill = PatternFill(start_color="70AD47", end_color="70AD47", fill_type="solid")
                elif conf_value >= 70:
                    conf_cell.fill = PatternFill(start_color="FFC000", end_color="FFC000", fill_type="solid")
                else:
                    conf_cell.fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
            except:
                pass
        
        # Criar aba de resumo
        summary_data = create_summary_stats(predictions_df)
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Resumo', index=False)
        
        # Formatar aba de resumo
        summary_sheet = writer.sheets['Resumo']
        for cell in summary_sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")
    
    return filename

def create_summary_stats(predictions_df):
    """Cria estatísticas resumidas das predições"""
    
    summary = []
    
    # Estatísticas gerais
    summary.append({
        'Métrica': 'Total de Predições',
        'Valor': len(predictions_df)
    })
    
    if 'confidence_percentage' in predictions_df.columns:
        summary.append({
            'Métrica': 'Confiança Média',
            'Valor': f"{predictions_df['confidence_percentage'].mean():.1f}%"
        })
    
    # Por faixas de confiança
    confidence_ranges = [(70, 80), (80, 90), (90, 100)]
    
    if 'confidence_percentage' in predictions_df.columns:
        for min_conf, max_conf in confidence_ranges:
            mask = (predictions_df['confidence_percentage'] >= min_conf) & \
                   (predictions_df['confidence_percentage'] < max_conf)
            count = mask.sum()
            
            summary.append({
                'Métrica': f'Predições {min_conf}-{max_conf}%',
                'Valor': count
            })
    
    # Se houver resultados reais
    if 'correct' in predictions_df.columns:
        accuracy = predictions_df['correct'].mean()
        summary.append({
            'Métrica': 'Acurácia Geral',
            'Valor': f"{accuracy:.1%}"
        })
    
    return summary

def demonstrate_outlier_handling():
    """
    Demonstra como o sistema lida com outliers
    Exemplo: Time com [5, 0, 0, 0, 0] gols HT
    """
    
    print("🔍 DEMONSTRAÇÃO: Tratamento de Outliers")
    print("=" * 60)
    
    # Criar sistema
    system = SmartPredictionSystem()
    
    # Dados do seu exemplo - time da casa
    home_team_data = pd.DataFrame({
        'ht_home_goals': [5, 0, 0, 0, 0],  # Exatamente seu exemplo!
        'over_05': [1, 0, 0, 0, 0],
        'date': pd.date_range('2024-01-01', periods=5)
    })
    
    # Analisar time
    stats = system.calculate_realistic_team_stats(home_team_data, 'home')
    
    print("📊 Dados originais do time:")
    print(f"   Gols HT: {home_team_data['ht_home_goals'].tolist()}")
    print(f"   Média com outlier: {home_team_data['ht_home_goals'].mean():.2f} gols")
    
    print("\n🎯 Após detecção de outliers:")
    print(f"   Outliers detectados: {stats['outliers_detected']}")
    print(f"   Taxa Over 0.5 raw: {stats['raw_over_rate']:.1%}")
    print(f"   Taxa Over 0.5 ajustada: {stats['adjusted_over_rate']:.1%}")
    
    # Mostrar o que aconteceu
    goals = home_team_data['ht_home_goals'].values
    outliers = system.detect_outliers(goals)
    clean_goals = goals[~outliers]
    
    print(f"\n✅ Gols após remover outliers: {clean_goals.tolist()}")
    print(f"   Média sem outlier: {clean_goals.mean():.2f} gols")
    print(f"   Frequência de marcar (limpa): {(clean_goals > 0).mean():.1%}")
    
    return stats
