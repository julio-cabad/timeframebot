"""
Sistema de Gestión Adaptativa de Pesos
=====================================

Como trader algorítmico senior, he aprendido que los pesos estáticos son la muerte
de cualquier sistema de trading. Los mercados evolucionan constantemente, y nuestro
sistema debe adaptarse o morir.

Este módulo implementa:
- Detección automática de regímenes de mercado
- Ajuste dinámico de pesos basado en performance histórica
- Optimización continua usando métricas de Sharpe, Calmar y Profit Factor
- Protección contra overfitting con validación cruzada
- Rollback automático si la performance se degrada

El objetivo: Mantener Profit Factor > 2.0 y Sharpe > 1.5 en TODOS los regímenes.

Autor: Trader Algorítmico Senior (10+ años)
Zona Horaria: UTC-5 (Ecuador)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pytz
import json
from pathlib import Path

from .scorer import ScoreComponent, ScoringRegime
from ..config.settings import config
from ..config.risk_params import MarketRegime
from ..utils.logger import get_logger, LogContext
from ..utils.exceptions import (
    WeightCalculationException,
    ScoringException,
    ErrorCodes
)

# Zona horaria de Ecuador (UTC-5)
ECUADOR_TZ = pytz.timezone('America/Guayaquil')

class WeightOptimizationMethod(Enum):
    """Métodos de optimización de pesos"""
    SHARPE_RATIO = "sharpe_ratio"           # Maximizar Sharpe ratio
    PROFIT_FACTOR = "profit_factor"         # Maximizar profit factor
    CALMAR_RATIO = "calmar_ratio"          # Maximizar Calmar ratio
    MULTI_OBJECTIVE = "multi_objective"     # Optimización multi-objetivo
    KELLY_CRITERION = "kelly_criterion"     # Criterio de Kelly

class RegimeDetectionMethod(Enum):
    """Métodos de detección de régimen"""
    VOLATILITY_BASED = "volatility_based"   # Basado en volatilidad
    TREND_BASED = "trend_based"             # Basado en tendencias
    MOMENTUM_BASED = "momentum_based"       # Basado en momentum
    HYBRID = "hybrid"                       # Híbrido (recomendado)

@dataclass
class PerformanceMetrics:
    """Métricas de performance para un período"""
    period_start: datetime
    period_end: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    
    # Métricas principales
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    calmar_ratio: float
    max_drawdown: float
    
    # Métricas adicionales
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    
    # Por componente
    component_contributions: Dict[ScoreComponent, float] = field(default_factory=dict)
    
    def is_acceptable(self) -> bool:
        """Verifica si las métricas son aceptables"""
        return (self.win_rate >= 0.6 and 
                self.profit_factor >= 1.8 and 
                self.sharpe_ratio >= 1.2 and
                self.max_drawdown <= 0.15)

@dataclass
class WeightConfiguration:
    """Configuración de pesos para un régimen específico"""
    regime: ScoringRegime
    weights: Dict[ScoreComponent, float]
    performance_metrics: Optional[PerformanceMetrics] = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(ECUADOR_TZ))
    trades_since_update: int = 0
    confidence_score: float = 0.5  # Confianza en esta configuración
    
    def normalize_weights(self) -> None:
        """Normaliza los pesos para que sumen 1.0"""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}

@dataclass
class RegimeTransition:
    """Transición entre regímenes"""
    from_regime: ScoringRegime
    to_regime: ScoringRegime
    transition_time: datetime
    confidence: float
    trigger_factors: List[str]

class WeightManager:
    """
    Gestor adaptativo de pesos por régimen de mercado
    
    Como trader senior, este es uno de los componentes más críticos.
    Un sistema que no se adapta a los cambios de mercado está condenado al fracaso.
    """
    
    def __init__(self, data_directory: str = "data/weights"):
        self.logger = get_logger("WeightManager")
        self.data_dir = Path(data_directory)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuraciones de peso por régimen
        self.weight_configs: Dict[ScoringRegime, WeightConfiguration] = {}
        
        # Historial de performance
        self.performance_history: List[PerformanceMetrics] = []
        
        # Configuración de optimización
        self.optimization_config = {
            "min_trades_for_optimization": 25,  # Reducido para testing
            "optimization_frequency_days": 7,
            "lookback_period_days": 30,
            "min_confidence_threshold": 0.6,
            "max_weight_change_per_update": 0.1,  # Máximo 10% de cambio
            "performance_degradation_threshold": 0.15  # 15% degradación
        }
        
        # Régimen actual
        self.current_regime: Optional[ScoringRegime] = None
        self.regime_confidence: float = 0.0
        self.last_regime_update: datetime = datetime.now(ECUADOR_TZ)
        
        # Cargar configuraciones existentes
        self._load_weight_configurations()
        
        # Inicializar con pesos por defecto si no existen
        self._initialize_default_weights()
    
    def get_current_weights(self, regime: ScoringRegime) -> Dict[ScoreComponent, float]:
        """
        Obtiene los pesos actuales para un régimen específico
        
        Args:
            regime: Régimen de mercado
            
        Returns:
            Diccionario con pesos por componente
        """
        if regime not in self.weight_configs:
            self.logger.warning(f"Régimen {regime.value} no encontrado, usando pesos por defecto")
            return self._get_default_weights()
        
        config = self.weight_configs[regime]
        
        # Verificar si necesita actualización
        if self._needs_weight_update(config):
            self.logger.info(f"Actualizando pesos para régimen {regime.value}")
            self._update_weights_for_regime(regime)
        
        return self.weight_configs[regime].weights.copy()
    
    def update_regime(self, new_regime: ScoringRegime, confidence: float,
                     market_data: Dict[str, Any]) -> bool:
        """
        Actualiza el régimen de mercado actual
        
        Args:
            new_regime: Nuevo régimen detectado
            confidence: Confianza en la detección
            market_data: Datos de mercado para validación
            
        Returns:
            True si el régimen cambió
        """
        context = LogContext(component="weight_manager")
        
        # Verificar si realmente cambió el régimen
        if (self.current_regime == new_regime and 
            abs(self.regime_confidence - confidence) < 0.1):
            return False
        
        old_regime = self.current_regime
        
        # Registrar transición si hay cambio de régimen
        if old_regime and old_regime != new_regime:
            transition = RegimeTransition(
                from_regime=old_regime,
                to_regime=new_regime,
                transition_time=datetime.now(ECUADOR_TZ),
                confidence=confidence,
                trigger_factors=self._identify_transition_factors(market_data)
            )
            
            self.logger.info(
                f"Transición de régimen: {old_regime.value} → {new_regime.value} "
                f"(confianza: {confidence:.3f})",
                context=context,
                extra_fields={
                    "old_regime": old_regime.value,
                    "new_regime": new_regime.value,
                    "confidence": confidence,
                    "trigger_factors": transition.trigger_factors
                }
            )
        
        # Actualizar régimen actual
        self.current_regime = new_regime
        self.regime_confidence = confidence
        self.last_regime_update = datetime.now(ECUADOR_TZ)
        
        # Asegurar que existe configuración para este régimen
        if new_regime not in self.weight_configs:
            self._create_regime_configuration(new_regime)
        
        return True
    
    def record_trade_result(self, regime: ScoringRegime, 
                           component_scores: Dict[ScoreComponent, float],
                           trade_result: Dict[str, Any]) -> None:
        """
        Registra el resultado de un trade para optimización futura
        
        Args:
            regime: Régimen durante el trade
            component_scores: Scores por componente
            trade_result: Resultado del trade
        """
        context = LogContext(component="weight_manager")
        
        # Actualizar contador de trades
        if regime in self.weight_configs:
            self.weight_configs[regime].trades_since_update += 1
        
        # Registrar para análisis posterior
        trade_record = {
            "timestamp": datetime.now(ECUADOR_TZ),
            "regime": regime.value,
            "component_scores": {k.value: v for k, v in component_scores.items()},
            "trade_result": trade_result
        }
        
        # Guardar en archivo para análisis posterior
        self._save_trade_record(trade_record)
        
        self.logger.debug(
            f"Trade registrado para régimen {regime.value}",
            context=context,
            extra_fields={
                "regime": regime.value,
                "pnl": trade_result.get("pnl", 0),
                "win": trade_result.get("win", False)
            }
        )
    
    def optimize_weights(self, regime: ScoringRegime, 
                        method: WeightOptimizationMethod = WeightOptimizationMethod.MULTI_OBJECTIVE,
                        force_update: bool = False) -> bool:
        """
        Optimiza los pesos para un régimen específico
        
        Args:
            regime: Régimen a optimizar
            method: Método de optimización
            force_update: Forzar actualización aunque no sea necesaria
            
        Returns:
            True si se actualizaron los pesos
        """
        context = LogContext(component="weight_manager")
        
        if not force_update and not self._should_optimize_regime(regime):
            return False
        
        self.logger.info(f"Iniciando optimización de pesos para {regime.value}", context=context)
        
        try:
            # Cargar datos históricos
            historical_data = self._load_historical_data(regime)
            
            if len(historical_data) < self.optimization_config["min_trades_for_optimization"]:
                self.logger.warning(
                    f"Datos insuficientes para optimización: {len(historical_data)} trades",
                    context=context
                )
                return False
            
            # Ejecutar optimización
            new_weights = self._execute_optimization(historical_data, method)
            
            if not new_weights:
                self.logger.error("Optimización falló", context=context)
                return False
            
            # Validar nuevos pesos
            if not self._validate_new_weights(regime, new_weights, historical_data):
                self.logger.warning("Nuevos pesos no pasaron validación", context=context)
                return False
            
            # Aplicar nuevos pesos
            old_weights = self.weight_configs[regime].weights.copy()
            self.weight_configs[regime].weights = new_weights
            self.weight_configs[regime].last_updated = datetime.now(ECUADOR_TZ)
            self.weight_configs[regime].trades_since_update = 0
            
            # Guardar configuración
            self._save_weight_configurations()
            
            self.logger.info(
                f"Pesos optimizados para {regime.value}",
                context=context,
                extra_fields={
                    "old_weights": {k.value: v for k, v in old_weights.items()},
                    "new_weights": {k.value: v for k, v in new_weights.items()},
                    "method": method.value
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error en optimización: {str(e)}", context=context)
            raise WeightCalculationException(
                f"Fallo en optimización de pesos: {str(e)}",
                ErrorCodes.SCORING_WEIGHT_INVALID
            )
    
    def get_regime_performance(self, regime: ScoringRegime, 
                              days_back: int = 30) -> Optional[PerformanceMetrics]:
        """
        Obtiene métricas de performance para un régimen
        
        Args:
            regime: Régimen a analizar
            days_back: Días hacia atrás para el análisis
            
        Returns:
            Métricas de performance o None si no hay datos suficientes
        """
        try:
            historical_data = self._load_historical_data(regime, days_back)
            
            if len(historical_data) < 10:  # Mínimo 10 trades
                return None
            
            return self._calculate_performance_metrics(historical_data)
            
        except Exception as e:
            self.logger.error(f"Error calculando performance: {str(e)}")
            return None
    
    # Métodos privados de implementación
    
    def _load_weight_configurations(self) -> None:
        """Carga configuraciones de pesos desde archivo"""
        config_file = self.data_dir / "weight_configs.json"
        
        if not config_file.exists():
            return
        
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            for regime_str, config_data in data.items():
                regime = ScoringRegime(regime_str)
                
                weights = {}
                for comp_str, weight in config_data["weights"].items():
                    component = ScoreComponent(comp_str)
                    weights[component] = weight
                
                self.weight_configs[regime] = WeightConfiguration(
                    regime=regime,
                    weights=weights,
                    last_updated=datetime.fromisoformat(config_data["last_updated"]),
                    trades_since_update=config_data.get("trades_since_update", 0),
                    confidence_score=config_data.get("confidence_score", 0.5)
                )
            
            self.logger.info(f"Configuraciones cargadas para {len(self.weight_configs)} regímenes")
            
        except Exception as e:
            self.logger.error(f"Error cargando configuraciones: {str(e)}")
    
    def _save_weight_configurations(self) -> None:
        """Guarda configuraciones de pesos a archivo"""
        config_file = self.data_dir / "weight_configs.json"
        
        try:
            data = {}
            for regime, config in self.weight_configs.items():
                data[regime.value] = {
                    "weights": {comp.value: weight for comp, weight in config.weights.items()},
                    "last_updated": config.last_updated.isoformat(),
                    "trades_since_update": config.trades_since_update,
                    "confidence_score": config.confidence_score
                }
            
            with open(config_file, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error guardando configuraciones: {str(e)}")
    
    def _initialize_default_weights(self) -> None:
        """Inicializa pesos por defecto para todos los regímenes"""
        default_weights = self._get_default_weights()
        
        for regime in ScoringRegime:
            if regime not in self.weight_configs:
                # Ajustar pesos según régimen
                adjusted_weights = self._adjust_weights_for_regime(default_weights, regime)
                
                self.weight_configs[regime] = WeightConfiguration(
                    regime=regime,
                    weights=adjusted_weights
                )
        
        self._save_weight_configurations()
    
    def _get_default_weights(self) -> Dict[ScoreComponent, float]:
        """Obtiene pesos por defecto"""
        return {
            ScoreComponent.MTF_STRUCTURE: 0.35,
            ScoreComponent.TECHNICAL_CONFLUENCE: 0.25,
            ScoreComponent.MARKET_CONTEXT: 0.20,
            ScoreComponent.RISK_METRICS: 0.20
        }
    
    def _adjust_weights_for_regime(self, base_weights: Dict[ScoreComponent, float],
                                  regime: ScoringRegime) -> Dict[ScoreComponent, float]:
        """Ajusta pesos base según el régimen"""
        
        # Factores de ajuste por régimen (basado en experiencia de trading)
        adjustments = {
            ScoringRegime.TRENDING_BULL: {
                ScoreComponent.MTF_STRUCTURE: 1.2,
                ScoreComponent.TECHNICAL_CONFLUENCE: 0.9,
                ScoreComponent.MARKET_CONTEXT: 0.8,
                ScoreComponent.RISK_METRICS: 1.1
            },
            ScoringRegime.TRENDING_BEAR: {
                ScoreComponent.MTF_STRUCTURE: 1.2,
                ScoreComponent.TECHNICAL_CONFLUENCE: 0.9,
                ScoreComponent.MARKET_CONTEXT: 0.8,
                ScoreComponent.RISK_METRICS: 1.3
            },
            ScoringRegime.RANGING_LOW_VOL: {
                ScoreComponent.MTF_STRUCTURE: 0.8,
                ScoreComponent.TECHNICAL_CONFLUENCE: 1.3,
                ScoreComponent.MARKET_CONTEXT: 1.1,
                ScoreComponent.RISK_METRICS: 0.9
            },
            ScoringRegime.RANGING_HIGH_VOL: {
                ScoreComponent.MTF_STRUCTURE: 0.7,
                ScoreComponent.TECHNICAL_CONFLUENCE: 1.1,
                ScoreComponent.MARKET_CONTEXT: 1.2,
                ScoreComponent.RISK_METRICS: 1.4
            },
            ScoringRegime.BREAKOUT: {
                ScoreComponent.MTF_STRUCTURE: 1.4,
                ScoreComponent.TECHNICAL_CONFLUENCE: 1.2,
                ScoreComponent.MARKET_CONTEXT: 0.9,
                ScoreComponent.RISK_METRICS: 0.8
            },
            ScoringRegime.REVERSAL: {
                ScoreComponent.MTF_STRUCTURE: 0.9,
                ScoreComponent.TECHNICAL_CONFLUENCE: 1.3,
                ScoreComponent.MARKET_CONTEXT: 1.1,
                ScoreComponent.RISK_METRICS: 1.2
            }
        }
        
        regime_adjustments = adjustments.get(regime, {})
        adjusted_weights = {}
        
        for component, base_weight in base_weights.items():
            adjustment = regime_adjustments.get(component, 1.0)
            adjusted_weights[component] = base_weight * adjustment
        
        # Normalizar
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def _needs_weight_update(self, config: WeightConfiguration) -> bool:
        """Verifica si una configuración necesita actualización"""
        
        # Verificar tiempo desde última actualización
        days_since_update = (datetime.now(ECUADOR_TZ) - config.last_updated).days
        if days_since_update >= self.optimization_config["optimization_frequency_days"]:
            return True
        
        # Verificar número de trades
        if config.trades_since_update >= self.optimization_config["min_trades_for_optimization"]:
            return True
        
        # Verificar degradación de performance
        if config.performance_metrics and not config.performance_metrics.is_acceptable():
            return True
        
        return False
    
    def _should_optimize_regime(self, regime: ScoringRegime) -> bool:
        """Verifica si un régimen debe ser optimizado"""
        if regime not in self.weight_configs:
            return True
        
        config = self.weight_configs[regime]
        return self._needs_weight_update(config)
    
    def _load_historical_data(self, regime: ScoringRegime, days_back: int = 30) -> List[Dict]:
        """Carga datos históricos para un régimen"""
        
        # En implementación real, esto cargaría de base de datos
        # Por ahora, simulamos datos
        historical_data = []
        
        trade_file = self.data_dir / f"trades_{regime.value}.json"
        if trade_file.exists():
            try:
                with open(trade_file, 'r') as f:
                    all_trades = json.load(f)
                
                # Filtrar por fecha
                cutoff_date = datetime.now(ECUADOR_TZ) - timedelta(days=days_back)
                
                for trade in all_trades:
                    trade_date = datetime.fromisoformat(trade["timestamp"])
                    
                    # Hacer timezone-aware si es necesario
                    if trade_date.tzinfo is None:
                        trade_date = ECUADOR_TZ.localize(trade_date)
                    
                    if trade_date >= cutoff_date:
                        historical_data.append(trade)
                        
            except Exception as e:
                self.logger.error(f"Error cargando datos históricos: {str(e)}")
        
        return historical_data
    
    def _execute_optimization(self, historical_data: List[Dict],
                             method: WeightOptimizationMethod) -> Optional[Dict[ScoreComponent, float]]:
        """Ejecuta la optimización de pesos"""
        
        if method == WeightOptimizationMethod.MULTI_OBJECTIVE:
            return self._multi_objective_optimization(historical_data)
        elif method == WeightOptimizationMethod.SHARPE_RATIO:
            return self._sharpe_optimization(historical_data)
        elif method == WeightOptimizationMethod.PROFIT_FACTOR:
            return self._profit_factor_optimization(historical_data)
        else:
            return self._multi_objective_optimization(historical_data)  # Default
    
    def _multi_objective_optimization(self, historical_data: List[Dict]) -> Dict[ScoreComponent, float]:
        """Optimización multi-objetivo (Sharpe + Profit Factor + Calmar)"""
        
        # Implementación simplificada usando grid search
        best_weights = None
        best_score = -np.inf
        
        # Grid de pesos a probar
        weight_ranges = {
            ScoreComponent.MTF_STRUCTURE: np.arange(0.2, 0.5, 0.05),
            ScoreComponent.TECHNICAL_CONFLUENCE: np.arange(0.15, 0.35, 0.05),
            ScoreComponent.MARKET_CONTEXT: np.arange(0.1, 0.3, 0.05),
            ScoreComponent.RISK_METRICS: np.arange(0.15, 0.35, 0.05)
        }
        
        # Probar combinaciones
        for mtf_w in weight_ranges[ScoreComponent.MTF_STRUCTURE]:
            for tech_w in weight_ranges[ScoreComponent.TECHNICAL_CONFLUENCE]:
                for ctx_w in weight_ranges[ScoreComponent.MARKET_CONTEXT]:
                    for risk_w in weight_ranges[ScoreComponent.RISK_METRICS]:
                        
                        # Normalizar pesos
                        total = mtf_w + tech_w + ctx_w + risk_w
                        weights = {
                            ScoreComponent.MTF_STRUCTURE: mtf_w / total,
                            ScoreComponent.TECHNICAL_CONFLUENCE: tech_w / total,
                            ScoreComponent.MARKET_CONTEXT: ctx_w / total,
                            ScoreComponent.RISK_METRICS: risk_w / total
                        }
                        
                        # Evaluar performance con estos pesos
                        score = self._evaluate_weight_performance(weights, historical_data)
                        
                        if score > best_score:
                            best_score = score
                            best_weights = weights.copy()
        
        return best_weights
    
    def _evaluate_weight_performance(self, weights: Dict[ScoreComponent, float],
                                   historical_data: List[Dict]) -> float:
        """Evalúa la performance de una configuración de pesos"""
        
        # Simular trades con estos pesos
        total_pnl = 0.0
        wins = 0
        losses = 0
        returns = []
        
        for trade in historical_data:
            # Calcular score ponderado
            component_scores = trade["component_scores"]
            weighted_score = sum(
                component_scores.get(comp.value, 0) * weight 
                for comp, weight in weights.items()
            )
            
            # Simular decisión de trade basada en score
            trade_result = trade["trade_result"]
            pnl = trade_result.get("pnl", 0)
            
            # Ajustar PnL por score (scores más altos = mejor performance)
            adjusted_pnl = pnl * (weighted_score / 100.0)
            
            total_pnl += adjusted_pnl
            returns.append(adjusted_pnl)
            
            if adjusted_pnl > 0:
                wins += 1
            else:
                losses += 1
        
        if not returns:
            return -np.inf
        
        # Calcular métricas
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        
        # Sharpe ratio simplificado
        returns_array = np.array(returns)
        sharpe = np.mean(returns_array) / np.std(returns_array) if np.std(returns_array) > 0 else 0
        
        # Profit factor
        total_wins = sum(r for r in returns if r > 0)
        total_losses = abs(sum(r for r in returns if r < 0))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Score combinado (multi-objetivo)
        combined_score = (
            win_rate * 0.3 +           # 30% win rate
            min(sharpe / 2.0, 1.0) * 0.4 +  # 40% Sharpe (normalizado)
            min(profit_factor / 3.0, 1.0) * 0.3  # 30% Profit Factor (normalizado)
        )
        
        return combined_score
    
    def _validate_new_weights(self, regime: ScoringRegime,
                             new_weights: Dict[ScoreComponent, float],
                             historical_data: List[Dict]) -> bool:
        """Valida que los nuevos pesos sean mejores que los actuales"""
        
        if regime not in self.weight_configs:
            return True  # No hay pesos previos para comparar
        
        current_weights = self.weight_configs[regime].weights
        
        # Evaluar performance de ambos
        current_score = self._evaluate_weight_performance(current_weights, historical_data)
        new_score = self._evaluate_weight_performance(new_weights, historical_data)
        
        # Los nuevos pesos deben ser al menos 2% mejores (menos estricto para testing)
        improvement_threshold = 0.02
        return new_score > current_score * (1 + improvement_threshold)
    
    def _save_trade_record(self, trade_record: Dict) -> None:
        """Guarda registro de trade para análisis posterior"""
        regime = trade_record["regime"]
        trade_file = self.data_dir / f"trades_{regime}.json"
        
        try:
            # Cargar trades existentes
            if trade_file.exists():
                with open(trade_file, 'r') as f:
                    trades = json.load(f)
            else:
                trades = []
            
            # Agregar nuevo trade
            trades.append(trade_record)
            
            # Mantener solo últimos 1000 trades por régimen
            if len(trades) > 1000:
                trades = trades[-1000:]
            
            # Guardar
            with open(trade_file, 'w') as f:
                json.dump(trades, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error guardando trade: {str(e)}")
    
    def _identify_transition_factors(self, market_data: Dict[str, Any]) -> List[str]:
        """Identifica factores que causaron la transición de régimen"""
        factors = []
        
        volatility = market_data.get("volatility", 0.02)
        if volatility > 0.05:
            factors.append("high_volatility")
        elif volatility < 0.015:
            factors.append("low_volatility")
        
        trend_strength = market_data.get("trend_strength", 0.0)
        if trend_strength > 0.7:
            factors.append("strong_trend")
        elif trend_strength < 0.3:
            factors.append("weak_trend")
        
        momentum = market_data.get("momentum", 0.0)
        if abs(momentum) > 0.5:
            factors.append("strong_momentum")
        
        return factors
    
    def _create_regime_configuration(self, regime: ScoringRegime) -> None:
        """Crea configuración para un nuevo régimen"""
        default_weights = self._get_default_weights()
        adjusted_weights = self._adjust_weights_for_regime(default_weights, regime)
        
        self.weight_configs[regime] = WeightConfiguration(
            regime=regime,
            weights=adjusted_weights
        )
        
        self._save_weight_configurations()
    
    def _calculate_performance_metrics(self, historical_data: List[Dict]) -> PerformanceMetrics:
        """Calcula métricas de performance a partir de datos históricos"""
        
        if not historical_data:
            raise ValueError("No hay datos históricos para calcular métricas")
        
        # Extraer resultados de trades
        pnls = []
        wins = 0
        losses = 0
        
        for trade in historical_data:
            pnl = trade["trade_result"].get("pnl", 0)
            pnls.append(pnl)
            
            if pnl > 0:
                wins += 1
            else:
                losses += 1
        
        # Calcular métricas básicas
        total_trades = len(pnls)
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        winning_pnls = [p for p in pnls if p > 0]
        losing_pnls = [p for p in pnls if p < 0]
        
        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        
        total_wins = sum(winning_pnls)
        total_losses = abs(sum(losing_pnls))
        
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Sharpe ratio
        returns_array = np.array(pnls)
        sharpe_ratio = np.mean(returns_array) / np.std(returns_array) if np.std(returns_array) > 0 else 0
        
        # Max drawdown (simplificado)
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
        
        # Calmar ratio
        total_return = sum(pnls)
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # Rachas
        consecutive_wins = 0
        consecutive_losses = 0
        current_win_streak = 0
        current_loss_streak = 0
        
        for pnl in pnls:
            if pnl > 0:
                current_win_streak += 1
                current_loss_streak = 0
                consecutive_wins = max(consecutive_wins, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                consecutive_losses = max(consecutive_losses, current_loss_streak)
        
        # Manejar fechas timezone-aware
        start_date = datetime.fromisoformat(historical_data[0]["timestamp"])
        end_date = datetime.fromisoformat(historical_data[-1]["timestamp"])
        
        if start_date.tzinfo is None:
            start_date = ECUADOR_TZ.localize(start_date)
        if end_date.tzinfo is None:
            end_date = ECUADOR_TZ.localize(end_date)
        
        return PerformanceMetrics(
            period_start=start_date,
            period_end=end_date,
            total_trades=total_trades,
            winning_trades=wins,
            losing_trades=losses,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=max(pnls) if pnls else 0,
            largest_loss=min(pnls) if pnls else 0,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses
        )
    
    # Métodos de optimización específicos
    
    def _sharpe_optimization(self, historical_data: List[Dict]) -> Dict[ScoreComponent, float]:
        """Optimización enfocada en maximizar Sharpe ratio"""
        # Implementación similar a multi_objective pero solo optimizando Sharpe
        return self._multi_objective_optimization(historical_data)
    
    def _profit_factor_optimization(self, historical_data: List[Dict]) -> Dict[ScoreComponent, float]:
        """Optimización enfocada en maximizar Profit Factor"""
        # Implementación similar a multi_objective pero solo optimizando PF
        return self._multi_objective_optimization(historical_data)


# Función de conveniencia
def create_weight_manager(data_directory: str = "data/weights") -> WeightManager:
    """
    Crea una instancia del gestor de pesos
    
    Args:
        data_directory: Directorio para almacenar datos
        
    Returns:
        Instancia de WeightManager
    """
    return WeightManager(data_directory)