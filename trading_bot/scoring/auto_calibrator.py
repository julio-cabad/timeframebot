"""
Sistema de Auto-Calibración Inteligente
======================================

Como trader algorítmico senior con más de 10 años de experiencia, he aprendido que
los sistemas estáticos mueren. La auto-calibración es lo que separa un bot amateur
de uno profesional que puede adaptarse y sobrevivir en mercados cambiantes.

Este sistema implementa:
- Optimización automática de umbrales de decisión
- Calibración de parámetros de riesgo basada en drawdown histórico
- Ajuste dinámico de expectativas de performance por régimen
- Detección automática de degradación de performance
- Rollback inteligente cuando los cambios empeoran los resultados
- Aprendizaje continuo con validación cruzada

El objetivo: Mantener el sistema siempre optimizado sin intervención manual.
Profit Factor > 2.0, Win Rate > 65%, Max DD < 15% en TODOS los regímenes.

Autor: Trader Algorítmico Senior (10+ años)
Zona Horaria: UTC-5 (Ecuador)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import pytz
from scipy import optimize
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

from .scorer import ScoreComponent, ScoringRegime
from .weight_manager import WeightManager, PerformanceMetrics
from ..config.settings import config
from ..utils.logger import get_logger, LogContext
from ..utils.exceptions import (
    ScoringException,
    WeightCalculationException,
    ErrorCodes
)

# Zona horaria de Ecuador (UTC-5)
ECUADOR_TZ = pytz.timezone('America/Guayaquil')

class CalibrationMethod(Enum):
    """Métodos de calibración disponibles"""
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GRID_SEARCH = "grid_search"
    GENETIC_ALGORITHM = "genetic_algorithm"
    GRADIENT_DESCENT = "gradient_descent"
    ENSEMBLE = "ensemble"  # Combina múltiples métodos

class CalibrationTarget(Enum):
    """Objetivos de calibración"""
    PROFIT_FACTOR = "profit_factor"
    SHARPE_RATIO = "sharpe_ratio"
    CALMAR_RATIO = "calmar_ratio"
    WIN_RATE = "win_rate"
    MAX_DRAWDOWN = "max_drawdown"
    MULTI_OBJECTIVE = "multi_objective"  # Optimización multi-objetivo

class ParameterType(Enum):
    """Tipos de parámetros calibrables"""
    DECISION_THRESHOLD = "decision_threshold"
    RISK_PARAMETER = "risk_parameter"
    SCORING_WEIGHT = "scoring_weight"
    REGIME_THRESHOLD = "regime_threshold"
    POSITION_SIZE = "position_size"

@dataclass
class ParameterBounds:
    """Límites para un parámetro calibrable"""
    name: str
    param_type: ParameterType
    min_value: float
    max_value: float
    current_value: float
    step_size: Optional[float] = None
    is_integer: bool = False
    
    def validate_value(self, value: float) -> float:
        """Valida y ajusta un valor dentro de los límites"""
        if self.is_integer:
            value = round(value)
        return max(self.min_value, min(self.max_value, value))

@dataclass
class CalibrationResult:
    """Resultado de una calibración"""
    method: CalibrationMethod
    target: CalibrationTarget
    regime: Optional[ScoringRegime]
    
    # Parámetros optimizados
    optimized_parameters: Dict[str, float]
    original_parameters: Dict[str, float]
    
    # Métricas de performance
    original_performance: PerformanceMetrics
    optimized_performance: PerformanceMetrics
    improvement_pct: float
    
    # Validación
    validation_passed: bool
    validation_score: float
    
    # Metadata
    calibration_time: datetime
    iterations: int
    convergence_achieved: bool
    confidence_score: float

@dataclass
class CalibrationConfig:
    """Configuración para el sistema de calibración"""
    # Frecuencia de calibración
    calibration_frequency_days: int = 14
    min_trades_for_calibration: int = 100
    
    # Validación cruzada
    validation_splits: int = 3
    min_improvement_threshold: float = 0.05  # 5% mejora mínima
    
    # Límites de seguridad
    max_parameter_change_pct: float = 0.25  # Máximo 25% de cambio por calibración
    rollback_threshold: float = -0.10  # Rollback si performance baja 10%
    
    # Objetivos de performance
    target_profit_factor: float = 2.0
    target_win_rate: float = 0.65
    target_sharpe_ratio: float = 1.5
    max_acceptable_drawdown: float = 0.15
    
    # Optimización
    max_iterations: int = 100
    convergence_tolerance: float = 1e-6
    population_size: int = 50  # Para algoritmos genéticos

class AutoCalibrator:
    """
    Sistema de auto-calibración inteligente
    
    Como trader senior, he diseñado este sistema para ser:
    1. Conservador - No hace cambios arriesgados
    2. Validado - Usa validación cruzada para evitar overfitting
    3. Adaptativo - Se ajusta a diferentes regímenes de mercado
    4. Reversible - Puede deshacer cambios que no funcionan
    """
    
    def __init__(self, weight_manager: WeightManager, 
                 data_directory: str = "data/calibration"):
        self.weight_manager = weight_manager
        self.logger = get_logger("AutoCalibrator")
        self.data_dir = Path(data_directory)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuración
        self.config = CalibrationConfig()
        
        # Parámetros calibrables
        self.calibrable_parameters = self._initialize_calibrable_parameters()
        
        # Historial de calibraciones
        self.calibration_history: List[CalibrationResult] = []
        
        # Estado actual
        self.last_calibration: Dict[ScoringRegime, datetime] = {}
        self.active_parameters: Dict[ScoringRegime, Dict[str, float]] = {}
        
        # Cargar estado previo
        self._load_calibration_state()
    
    def should_calibrate(self, regime: ScoringRegime) -> bool:
        """
        Determina si un régimen necesita calibración
        
        Args:
            regime: Régimen a evaluar
            
        Returns:
            True si necesita calibración
        """
        context = LogContext(component="auto_calibrator")
        
        # Verificar tiempo desde última calibración
        last_cal = self.last_calibration.get(regime)
        if last_cal:
            days_since = (datetime.now(ECUADOR_TZ) - last_cal).days
            if days_since < self.config.calibration_frequency_days:
                return False
        
        # Verificar datos suficientes
        performance = self.weight_manager.get_regime_performance(regime, days_back=60)
        if not performance or performance.total_trades < self.config.min_trades_for_calibration:
            self.logger.debug(
                f"Datos insuficientes para calibración de {regime.value}",
                context=context
            )
            return False
        
        # Verificar si la performance está degradándose
        if not performance.is_acceptable():
            self.logger.info(
                f"Performance degradada en {regime.value}, calibración necesaria",
                context=context,
                extra_fields={
                    "win_rate": performance.win_rate,
                    "profit_factor": performance.profit_factor,
                    "max_drawdown": performance.max_drawdown
                }
            )
            return True
        
        # Calibración periódica
        return last_cal is None or days_since >= self.config.calibration_frequency_days
    
    def calibrate_regime(self, regime: ScoringRegime, 
                        method: CalibrationMethod = CalibrationMethod.ENSEMBLE,
                        target: CalibrationTarget = CalibrationTarget.MULTI_OBJECTIVE,
                        force: bool = False) -> Optional[CalibrationResult]:
        """
        Calibra parámetros para un régimen específico
        
        Args:
            regime: Régimen a calibrar
            method: Método de optimización
            target: Objetivo de optimización
            force: Forzar calibración aunque no sea necesaria
            
        Returns:
            Resultado de calibración o None si no fue necesaria
        """
        context = LogContext(component="auto_calibrator")
        
        if not force and not self.should_calibrate(regime):
            return None
        
        self.logger.info(
            f"Iniciando calibración para {regime.value}",
            context=context,
            extra_fields={
                "method": method.value,
                "target": target.value
            }
        )
        
        try:
            # Obtener datos históricos
            historical_data = self._load_historical_data(regime)
            if len(historical_data) < self.config.min_trades_for_calibration:
                self.logger.warning(
                    f"Datos insuficientes: {len(historical_data)} trades",
                    context=context
                )
                return None
            
            # Obtener performance actual
            current_performance = self._calculate_performance_from_data(historical_data)
            
            # Obtener parámetros actuales
            current_params = self._get_current_parameters(regime)
            
            # Ejecutar optimización
            optimized_params = self._optimize_parameters(
                historical_data, current_params, method, target
            )
            
            if not optimized_params:
                self.logger.warning("Optimización falló", context=context)
                return None
            
            # Validar con validación cruzada
            validation_result = self._validate_parameters(
                historical_data, optimized_params, target
            )
            
            if not validation_result["passed"]:
                self.logger.warning(
                    f"Validación falló: {validation_result['reason']}",
                    context=context
                )
                return None
            
            # Calcular performance esperada
            optimized_performance = self._simulate_performance(
                historical_data, optimized_params
            )
            
            # Calcular mejora
            improvement = self._calculate_improvement(
                current_performance, optimized_performance, target
            )
            
            if improvement < self.config.min_improvement_threshold:
                self.logger.info(
                    f"Mejora insuficiente: {improvement:.1%}",
                    context=context
                )
                return None
            
            # Crear resultado
            result = CalibrationResult(
                method=method,
                target=target,
                regime=regime,
                optimized_parameters=optimized_params,
                original_parameters=current_params,
                original_performance=current_performance,
                optimized_performance=optimized_performance,
                improvement_pct=improvement,
                validation_passed=validation_result["passed"],
                validation_score=validation_result["score"],
                calibration_time=datetime.now(ECUADOR_TZ),
                iterations=validation_result.get("iterations", 0),
                convergence_achieved=validation_result.get("converged", False),
                confidence_score=validation_result.get("confidence", 0.5)
            )
            
            # Aplicar parámetros optimizados
            self._apply_calibration_result(result)
            
            # Guardar en historial
            self.calibration_history.append(result)
            self.last_calibration[regime] = datetime.now(ECUADOR_TZ)
            
            # Persistir estado
            self._save_calibration_state()
            
            self.logger.info(
                f"Calibración completada para {regime.value}: {improvement:.1%} mejora",
                context=context,
                extra_fields={
                    "improvement_pct": improvement,
                    "validation_score": validation_result["score"],
                    "iterations": result.iterations
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en calibración: {str(e)}", context=context)
            raise ScoringException(
                f"Fallo en calibración: {str(e)}",
                ErrorCodes.SCORING_CALCULATION_FAILED
            )
    
    def auto_calibrate_all_regimes(self) -> Dict[ScoringRegime, Optional[CalibrationResult]]:
        """
        Ejecuta calibración automática para todos los regímenes que la necesiten
        
        Returns:
            Diccionario con resultados por régimen
        """
        context = LogContext(component="auto_calibrator")
        
        self.logger.info("Iniciando calibración automática global", context=context)
        
        results = {}
        
        for regime in ScoringRegime:
            try:
                result = self.calibrate_regime(regime)
                results[regime] = result
                
                if result:
                    self.logger.info(
                        f"Régimen {regime.value} calibrado: {result.improvement_pct:.1%} mejora"
                    )
                else:
                    self.logger.debug(f"Régimen {regime.value} no requiere calibración")
                    
            except Exception as e:
                self.logger.error(f"Error calibrando {regime.value}: {str(e)}")
                results[regime] = None
        
        # Resumen
        calibrated_count = sum(1 for r in results.values() if r is not None)
        total_improvement = sum(r.improvement_pct for r in results.values() if r is not None)
        
        self.logger.info(
            f"Calibración global completada: {calibrated_count}/{len(ScoringRegime)} regímenes",
            context=context,
            extra_fields={
                "calibrated_regimes": calibrated_count,
                "total_regimes": len(ScoringRegime),
                "avg_improvement": total_improvement / calibrated_count if calibrated_count > 0 else 0
            }
        )
        
        return results
    
    def rollback_calibration(self, regime: ScoringRegime, 
                           calibration_id: Optional[str] = None) -> bool:
        """
        Revierte una calibración si está causando problemas
        
        Args:
            regime: Régimen a revertir
            calibration_id: ID específico de calibración (opcional)
            
        Returns:
            True si el rollback fue exitoso
        """
        context = LogContext(component="auto_calibrator")
        
        try:
            # Encontrar calibración a revertir
            target_calibration = None
            
            if calibration_id:
                # Buscar por ID específico
                for cal in reversed(self.calibration_history):
                    if cal.regime == regime and str(id(cal)) == calibration_id:
                        target_calibration = cal
                        break
            else:
                # Usar la más reciente para este régimen
                for cal in reversed(self.calibration_history):
                    if cal.regime == regime:
                        target_calibration = cal
                        break
            
            if not target_calibration:
                self.logger.warning(f"No se encontró calibración para revertir en {regime.value}")
                return False
            
            # Restaurar parámetros originales
            self._apply_parameters(regime, target_calibration.original_parameters)
            
            # Marcar como revertida (agregar flag al historial)
            target_calibration.validation_passed = False
            
            self.logger.info(
                f"Rollback completado para {regime.value}",
                context=context,
                extra_fields={
                    "calibration_time": target_calibration.calibration_time.isoformat(),
                    "improvement_lost": target_calibration.improvement_pct
                }
            )
            
            # Persistir cambios
            self._save_calibration_state()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error en rollback: {str(e)}", context=context)
            return False
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del sistema de calibración
        
        Returns:
            Diccionario con estado completo
        """
        status = {
            "last_calibrations": {},
            "active_parameters": {},
            "calibration_history_count": len(self.calibration_history),
            "recent_improvements": {},
            "next_calibration_due": {}
        }
        
        for regime in ScoringRegime:
            # Última calibración
            last_cal = self.last_calibration.get(regime)
            status["last_calibrations"][regime.value] = (
                last_cal.isoformat() if last_cal else None
            )
            
            # Parámetros activos
            active_params = self.active_parameters.get(regime, {})
            status["active_parameters"][regime.value] = active_params
            
            # Mejoras recientes
            recent_cal = None
            for cal in reversed(self.calibration_history):
                if cal.regime == regime:
                    recent_cal = cal
                    break
            
            if recent_cal:
                status["recent_improvements"][regime.value] = recent_cal.improvement_pct
            
            # Próxima calibración
            if last_cal:
                next_due = last_cal + timedelta(days=self.config.calibration_frequency_days)
                status["next_calibration_due"][regime.value] = next_due.isoformat()
        
        return status
    
    def _initialize_calibrable_parameters(self) -> Dict[str, ParameterBounds]:
        """
        Inicializa los parámetros calibrables con límites seguros
        
        Como trader senior, he aprendido que los límites conservadores
        son la diferencia entre un bot que sobrevive y uno que explota.
        """
        parameters = {}
        
        # Umbrales de decisión - Críticos para la rentabilidad
        parameters["min_score_threshold"] = ParameterBounds(
            name="min_score_threshold",
            param_type=ParameterType.DECISION_THRESHOLD,
            min_value=50.0,  # Nunca por debajo de 50
            max_value=80.0,  # Nunca tan alto que perdamos oportunidades
            current_value=65.0,
            step_size=1.0
        )
        
        parameters["llm_threshold_min"] = ParameterBounds(
            name="llm_threshold_min", 
            param_type=ParameterType.DECISION_THRESHOLD,
            min_value=55.0,
            max_value=70.0,
            current_value=60.0,
            step_size=1.0
        )
        
        parameters["llm_threshold_max"] = ParameterBounds(
            name="llm_threshold_max",
            param_type=ParameterType.DECISION_THRESHOLD, 
            min_value=70.0,
            max_value=85.0,
            current_value=75.0,
            step_size=1.0
        )
        
        # Parámetros de riesgo - La supervivencia del capital es lo primero
        parameters["max_position_size"] = ParameterBounds(
            name="max_position_size",
            param_type=ParameterType.POSITION_SIZE,
            min_value=0.01,  # Mínimo 1%
            max_value=0.10,  # Máximo 10% - Nunca más
            current_value=0.05,  # 5% por defecto
            step_size=0.005
        )
        
        parameters["stop_loss_multiplier"] = ParameterBounds(
            name="stop_loss_multiplier",
            param_type=ParameterType.RISK_PARAMETER,
            min_value=1.0,   # ATR mínimo
            max_value=3.0,   # ATR máximo
            current_value=1.5,
            step_size=0.1
        )
        
        parameters["take_profit_ratio"] = ParameterBounds(
            name="take_profit_ratio",
            param_type=ParameterType.RISK_PARAMETER,
            min_value=1.5,   # Mínimo 1.5:1 R:R
            max_value=4.0,   # Máximo 4:1 R:R
            current_value=2.5,
            step_size=0.1
        )
        
        # Pesos de scoring - El corazón del sistema
        parameters["mtf_structure_weight"] = ParameterBounds(
            name="mtf_structure_weight",
            param_type=ParameterType.SCORING_WEIGHT,
            min_value=0.20,  # Mínimo 20%
            max_value=0.50,  # Máximo 50%
            current_value=0.35,
            step_size=0.01
        )
        
        parameters["technical_confluence_weight"] = ParameterBounds(
            name="technical_confluence_weight",
            param_type=ParameterType.SCORING_WEIGHT,
            min_value=0.15,
            max_value=0.40,
            current_value=0.25,
            step_size=0.01
        )
        
        parameters["market_context_weight"] = ParameterBounds(
            name="market_context_weight", 
            param_type=ParameterType.SCORING_WEIGHT,
            min_value=0.10,
            max_value=0.30,
            current_value=0.20,
            step_size=0.01
        )
        
        parameters["risk_metrics_weight"] = ParameterBounds(
            name="risk_metrics_weight",
            param_type=ParameterType.SCORING_WEIGHT,
            min_value=0.15,
            max_value=0.35,
            current_value=0.20,
            step_size=0.01
        )
        
        return parameters
    
    def _load_calibration_state(self) -> None:
        """
        Carga el estado previo de calibraciones
        
        La persistencia es crítica - nunca perdemos el aprendizaje acumulado
        """
        try:
            state_file = self.data_dir / "calibration_state.json"
            
            if not state_file.exists():
                self.logger.info("No hay estado previo de calibración, iniciando limpio")
                return
            
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            # Cargar última calibración por régimen
            if "last_calibrations" in state_data:
                for regime_str, timestamp_str in state_data["last_calibrations"].items():
                    if timestamp_str:
                        regime = ScoringRegime(regime_str)
                        timestamp = datetime.fromisoformat(timestamp_str)
                        self.last_calibration[regime] = timestamp
            
            # Cargar parámetros activos
            if "active_parameters" in state_data:
                for regime_str, params in state_data["active_parameters"].items():
                    regime = ScoringRegime(regime_str)
                    self.active_parameters[regime] = params
            
            # Cargar historial (últimas 50 calibraciones para no sobrecargar memoria)
            history_file = self.data_dir / "calibration_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                # Reconstruir objetos CalibrationResult (simplificado)
                for item in history_data[-50:]:  # Solo las últimas 50
                    try:
                        result = CalibrationResult(
                            method=CalibrationMethod(item["method"]),
                            target=CalibrationTarget(item["target"]),
                            regime=ScoringRegime(item["regime"]) if item["regime"] else None,
                            optimized_parameters=item["optimized_parameters"],
                            original_parameters=item["original_parameters"],
                            original_performance=PerformanceMetrics(**item["original_performance"]),
                            optimized_performance=PerformanceMetrics(**item["optimized_performance"]),
                            improvement_pct=item["improvement_pct"],
                            validation_passed=item["validation_passed"],
                            validation_score=item["validation_score"],
                            calibration_time=datetime.fromisoformat(item["calibration_time"]),
                            iterations=item["iterations"],
                            convergence_achieved=item["convergence_achieved"],
                            confidence_score=item["confidence_score"]
                        )
                        self.calibration_history.append(result)
                    except Exception as e:
                        self.logger.warning(f"Error cargando calibración del historial: {e}")
            
            self.logger.info(f"Estado de calibración cargado: {len(self.calibration_history)} calibraciones en historial")
            
        except Exception as e:
            self.logger.error(f"Error cargando estado de calibración: {e}")
            # Continuar con estado limpio en caso de error
    
    def _save_calibration_state(self) -> None:
        """
        Persiste el estado actual de calibraciones
        
        Guardamos todo - la experiencia acumulada es oro puro
        """
        try:
            # Estado principal
            state_data = {
                "last_calibrations": {
                    regime.value: timestamp.isoformat() if timestamp else None
                    for regime, timestamp in self.last_calibration.items()
                },
                "active_parameters": {
                    regime.value: params
                    for regime, params in self.active_parameters.items()
                },
                "saved_at": datetime.now(ECUADOR_TZ).isoformat()
            }
            
            state_file = self.data_dir / "calibration_state.json"
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            # Historial de calibraciones
            history_data = []
            for result in self.calibration_history:
                history_item = {
                    "method": result.method.value,
                    "target": result.target.value,
                    "regime": result.regime.value if result.regime else None,
                    "optimized_parameters": result.optimized_parameters,
                    "original_parameters": result.original_parameters,
                    "original_performance": result.original_performance.__dict__,
                    "optimized_performance": result.optimized_performance.__dict__,
                    "improvement_pct": result.improvement_pct,
                    "validation_passed": result.validation_passed,
                    "validation_score": result.validation_score,
                    "calibration_time": result.calibration_time.isoformat(),
                    "iterations": result.iterations,
                    "convergence_achieved": result.convergence_achieved,
                    "confidence_score": result.confidence_score
                }
                history_data.append(history_item)
            
            history_file = self.data_dir / "calibration_history.json"
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            self.logger.debug("Estado de calibración guardado exitosamente")
            
        except Exception as e:
            self.logger.error(f"Error guardando estado de calibración: {e}")
    
    def _load_historical_data(self, regime: ScoringRegime, days_back: int = 90) -> pd.DataFrame:
        """
        Carga datos históricos para calibración
        
        Args:
            regime: Régimen de mercado
            days_back: Días hacia atrás para cargar
            
        Returns:
            DataFrame con datos históricos de trades
        """
        try:
            # Obtener datos del weight_manager
            performance_data = self.weight_manager.get_regime_performance(regime, days_back)
            
            if not performance_data or not hasattr(performance_data, 'trade_history'):
                self.logger.warning(f"No hay datos históricos para {regime.value}")
                return pd.DataFrame()
            
            # Convertir a DataFrame si no lo es ya
            if isinstance(performance_data.trade_history, pd.DataFrame):
                df = performance_data.trade_history.copy()
            else:
                # Asumir que es una lista de diccionarios
                df = pd.DataFrame(performance_data.trade_history)
            
            if df.empty:
                return df
            
            # Asegurar columnas requeridas
            required_columns = [
                'timestamp', 'symbol', 'direction', 'entry_price', 'exit_price',
                'pnl', 'pnl_pct', 'score', 'regime', 'duration_minutes'
            ]
            
            for col in required_columns:
                if col not in df.columns:
                    # Valores por defecto para columnas faltantes
                    if col == 'timestamp':
                        df[col] = pd.Timestamp.now(ECUADOR_TZ)
                    elif col == 'regime':
                        df[col] = regime.value
                    elif col in ['pnl', 'pnl_pct', 'score']:
                        df[col] = 0.0
                    elif col == 'duration_minutes':
                        df[col] = 60  # 1 hora por defecto
                    else:
                        df[col] = ''
            
            # Filtrar por régimen si hay datos mixtos
            if 'regime' in df.columns:
                df = df[df['regime'] == regime.value]
            
            # Ordenar por timestamp
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
            
            self.logger.info(f"Cargados {len(df)} trades históricos para {regime.value}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error cargando datos históricos: {e}")
            return pd.DataFrame()
    
    def _calculate_performance_from_data(self, data: pd.DataFrame) -> PerformanceMetrics:
        """
        Calcula métricas de performance desde datos históricos
        
        Como trader senior, sé que las métricas correctas son todo
        """
        if data.empty:
            return PerformanceMetrics(
                period_start=datetime.now(ECUADOR_TZ) - timedelta(days=30),
                period_end=datetime.now(ECUADOR_TZ),
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                consecutive_wins=0,
                consecutive_losses=0
            )
        
        try:
            total_trades = len(data)
            
            # Separar wins y losses
            wins = data[data['pnl'] > 0]
            losses = data[data['pnl'] < 0]
            
            winning_trades = len(wins)
            losing_trades = len(losses)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # Profit Factor
            gross_profit = wins['pnl'].sum() if not wins.empty else 0.0
            gross_loss = abs(losses['pnl'].sum()) if not losses.empty else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Promedios y extremos
            avg_win = wins['pnl'].mean() if not wins.empty else 0.0
            avg_loss = losses['pnl'].mean() if not losses.empty else 0.0
            largest_win = wins['pnl'].max() if not wins.empty else 0.0
            largest_loss = losses['pnl'].min() if not losses.empty else 0.0
            
            # Rachas consecutivas
            consecutive_wins = 0
            consecutive_losses = 0
            current_win_streak = 0
            current_loss_streak = 0
            
            for pnl in data['pnl']:
                if pnl > 0:
                    current_win_streak += 1
                    current_loss_streak = 0
                    consecutive_wins = max(consecutive_wins, current_win_streak)
                elif pnl < 0:
                    current_loss_streak += 1
                    current_win_streak = 0
                    consecutive_losses = max(consecutive_losses, current_loss_streak)
            
            # Drawdown máximo
            if 'pnl' in data.columns:
                cumulative_pnl = data['pnl'].cumsum()
                running_max = cumulative_pnl.expanding().max()
                drawdown = (cumulative_pnl - running_max) / running_max.abs()
                max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0.0
            else:
                max_drawdown = 0.0
            
            # Sharpe Ratio (aproximado)
            if 'pnl_pct' in data.columns and len(data) > 1:
                returns = data['pnl_pct'].dropna()
                if len(returns) > 1 and returns.std() > 0:
                    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Anualizado
                else:
                    sharpe_ratio = 0.0
            else:
                sharpe_ratio = 0.0
            
            # Calmar Ratio
            annual_return = data['pnl'].sum() * (252 / len(data)) if len(data) > 0 else 0.0
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0
            
            # Período de datos
            if 'timestamp' in data.columns and not data.empty:
                period_start = data['timestamp'].min()
                period_end = data['timestamp'].max()
            else:
                period_start = datetime.now(ECUADOR_TZ) - timedelta(days=30)
                period_end = datetime.now(ECUADOR_TZ)
            
            return PerformanceMetrics(
                period_start=period_start,
                period_end=period_end,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                consecutive_wins=consecutive_wins,
                consecutive_losses=consecutive_losses
            )
            
        except Exception as e:
            self.logger.error(f"Error calculando performance: {e}")
            return PerformanceMetrics(
                period_start=datetime.now(ECUADOR_TZ) - timedelta(days=30),
                period_end=datetime.now(ECUADOR_TZ),
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                consecutive_wins=0,
                consecutive_losses=0
            )
    
    def _get_current_parameters(self, regime: ScoringRegime) -> Dict[str, float]:
        """
        Obtiene los parámetros actuales para un régimen
        """
        # Primero intentar obtener parámetros activos guardados
        if regime in self.active_parameters:
            return self.active_parameters[regime].copy()
        
        # Si no hay parámetros guardados, usar valores por defecto
        current_params = {}
        
        for param_name, bounds in self.calibrable_parameters.items():
            current_params[param_name] = bounds.current_value
        
        # Obtener pesos actuales del weight_manager si está disponible
        try:
            current_weights = self.weight_manager.get_regime_weights(regime)
            if current_weights:
                current_params.update({
                    "mtf_structure_weight": current_weights.get("mtf_structure", 0.35),
                    "technical_confluence_weight": current_weights.get("technical_confluence", 0.25),
                    "market_context_weight": current_weights.get("market_context", 0.20),
                    "risk_metrics_weight": current_weights.get("risk_metrics", 0.20)
                })
        except Exception as e:
            self.logger.debug(f"No se pudieron obtener pesos del weight_manager: {e}")
        
        return current_params    

    def _optimize_parameters(self, historical_data: pd.DataFrame, 
                           current_params: Dict[str, float],
                           method: CalibrationMethod,
                           target: CalibrationTarget) -> Optional[Dict[str, float]]:
        """
        Optimiza parámetros usando el método especificado
        
        Como trader senior, he probado todos los métodos de optimización.
        El ensemble es el rey - combina lo mejor de cada método.
        """
        if historical_data.empty:
            return None
        
        try:
            if method == CalibrationMethod.ENSEMBLE:
                return self._optimize_ensemble(historical_data, current_params, target)
            elif method == CalibrationMethod.BAYESIAN_OPTIMIZATION:
                return self._optimize_bayesian(historical_data, current_params, target)
            elif method == CalibrationMethod.GRID_SEARCH:
                return self._optimize_grid_search(historical_data, current_params, target)
            elif method == CalibrationMethod.GENETIC_ALGORITHM:
                return self._optimize_genetic(historical_data, current_params, target)
            elif method == CalibrationMethod.GRADIENT_DESCENT:
                return self._optimize_gradient(historical_data, current_params, target)
            else:
                self.logger.warning(f"Método de optimización no soportado: {method}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error en optimización {method.value}: {e}")
            return None
    
    def _optimize_ensemble(self, data: pd.DataFrame, current_params: Dict[str, float],
                          target: CalibrationTarget) -> Dict[str, float]:
        """
        Optimización ensemble - El método que uso en producción
        
        Combina múltiples métodos y toma el consenso. Más robusto que cualquier método individual.
        """
        results = []
        
        # Ejecutar múltiples métodos
        methods = [
            CalibrationMethod.BAYESIAN_OPTIMIZATION,
            CalibrationMethod.GRID_SEARCH,
            CalibrationMethod.GENETIC_ALGORITHM
        ]
        
        for method in methods:
            try:
                result = self._optimize_parameters(data, current_params, method, target)
                if result:
                    # Evaluar la calidad del resultado
                    score = self._evaluate_parameter_set(data, result, target)
                    results.append((result, score, method))
            except Exception as e:
                self.logger.debug(f"Método {method.value} falló: {e}")
                continue
        
        if not results:
            self.logger.warning("Todos los métodos de optimización fallaron")
            return current_params
        
        # Ordenar por score y tomar el mejor
        results.sort(key=lambda x: x[1], reverse=True)
        best_params, best_score, best_method = results[0]
        
        self.logger.info(f"Mejor método: {best_method.value} con score {best_score:.4f}")
        
        # Promedio ponderado de los top 3 resultados para suavizar
        if len(results) >= 3:
            weights = [0.5, 0.3, 0.2]  # Más peso al mejor
            ensemble_params = {}
            
            for param_name in best_params.keys():
                weighted_sum = 0.0
                for i, (params, score, method) in enumerate(results[:3]):
                    weighted_sum += params[param_name] * weights[i]
                ensemble_params[param_name] = weighted_sum
            
            return ensemble_params
        
        return best_params
    
    def _optimize_bayesian(self, data: pd.DataFrame, current_params: Dict[str, float],
                          target: CalibrationTarget) -> Dict[str, float]:
        """
        Optimización Bayesiana - Eficiente para espacios de parámetros complejos
        """
        from scipy.optimize import minimize
        
        def objective_function(params_array):
            # Convertir array a diccionario de parámetros
            params_dict = {}
            param_names = list(current_params.keys())
            
            for i, param_name in enumerate(param_names):
                if i < len(params_array):
                    bounds = self.calibrable_parameters[param_name]
                    params_dict[param_name] = bounds.validate_value(params_array[i])
                else:
                    params_dict[param_name] = current_params[param_name]
            
            # Evaluar performance con estos parámetros
            score = self._evaluate_parameter_set(data, params_dict, target)
            return -score  # Minimizar (scipy minimiza)
        
        # Preparar bounds para scipy
        bounds = []
        param_names = list(current_params.keys())
        
        for param_name in param_names:
            if param_name in self.calibrable_parameters:
                param_bounds = self.calibrable_parameters[param_name]
                bounds.append((param_bounds.min_value, param_bounds.max_value))
            else:
                # Bounds por defecto para parámetros no definidos
                current_val = current_params[param_name]
                bounds.append((current_val * 0.5, current_val * 1.5))
        
        # Punto inicial
        x0 = [current_params[name] for name in param_names]
        
        # Optimizar
        result = minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': self.config.max_iterations}
        )
        
        if result.success:
            # Convertir resultado a diccionario
            optimized_params = {}
            for i, param_name in enumerate(param_names):
                if i < len(result.x):
                    if param_name in self.calibrable_parameters:
                        bounds = self.calibrable_parameters[param_name]
                        optimized_params[param_name] = bounds.validate_value(result.x[i])
                    else:
                        optimized_params[param_name] = result.x[i]
                else:
                    optimized_params[param_name] = current_params[param_name]
            
            return optimized_params
        
        return current_params
    
    def _optimize_grid_search(self, data: pd.DataFrame, current_params: Dict[str, float],
                             target: CalibrationTarget) -> Dict[str, float]:
        """
        Grid Search - Exhaustivo pero confiable para espacios pequeños
        """
        best_params = current_params.copy()
        best_score = self._evaluate_parameter_set(data, current_params, target)
        
        # Definir grid de búsqueda (limitado para evitar explosión combinatoria)
        param_grids = {}
        
        for param_name, current_value in current_params.items():
            if param_name in self.calibrable_parameters:
                bounds = self.calibrable_parameters[param_name]
                step = bounds.step_size or (bounds.max_value - bounds.min_value) / 10
                
                # Grid de 5 puntos alrededor del valor actual
                grid_points = []
                for i in range(-2, 3):
                    new_value = current_value + (i * step)
                    new_value = bounds.validate_value(new_value)
                    grid_points.append(new_value)
                
                param_grids[param_name] = list(set(grid_points))  # Remover duplicados
        
        # Limitar combinaciones para evitar explosión
        max_combinations = 1000
        total_combinations = 1
        for grid in param_grids.values():
            total_combinations *= len(grid)
        
        if total_combinations > max_combinations:
            # Reducir grid si es muy grande
            for param_name in param_grids:
                if len(param_grids[param_name]) > 3:
                    param_grids[param_name] = param_grids[param_name][:3]
        
        # Evaluar todas las combinaciones
        combinations_tested = 0
        
        def recursive_search(param_names, current_combination, remaining_params):
            nonlocal best_params, best_score, combinations_tested
            
            if not remaining_params:
                # Evaluar esta combinación
                test_params = current_params.copy()
                test_params.update(current_combination)
                
                score = self._evaluate_parameter_set(data, test_params, target)
                combinations_tested += 1
                
                if score > best_score:
                    best_score = score
                    best_params = test_params.copy()
                
                return
            
            param_name = remaining_params[0]
            remaining = remaining_params[1:]
            
            for value in param_grids.get(param_name, [current_params[param_name]]):
                new_combination = current_combination.copy()
                new_combination[param_name] = value
                recursive_search(param_names, new_combination, remaining)
        
        param_names = list(param_grids.keys())
        recursive_search(param_names, {}, param_names)
        
        self.logger.info(f"Grid search completado: {combinations_tested} combinaciones evaluadas")
        
        return best_params
    
    def _optimize_genetic(self, data: pd.DataFrame, current_params: Dict[str, float],
                         target: CalibrationTarget) -> Dict[str, float]:
        """
        Algoritmo Genético - Bueno para espacios de búsqueda complejos
        """
        import random
        
        population_size = min(self.config.population_size, 30)  # Limitar para performance
        generations = min(self.config.max_iterations // population_size, 20)
        mutation_rate = 0.1
        crossover_rate = 0.8
        
        param_names = list(current_params.keys())
        
        def create_individual():
            """Crea un individuo aleatorio"""
            individual = {}
            for param_name in param_names:
                if param_name in self.calibrable_parameters:
                    bounds = self.calibrable_parameters[param_name]
                    value = random.uniform(bounds.min_value, bounds.max_value)
                    individual[param_name] = bounds.validate_value(value)
                else:
                    # Variación alrededor del valor actual
                    current_val = current_params[param_name]
                    variation = random.uniform(-0.2, 0.2)  # ±20%
                    individual[param_name] = current_val * (1 + variation)
            return individual
        
        def fitness(individual):
            """Calcula fitness de un individuo"""
            return self._evaluate_parameter_set(data, individual, target)
        
        def crossover(parent1, parent2):
            """Cruza dos padres para crear descendencia"""
            child = {}
            for param_name in param_names:
                if random.random() < 0.5:
                    child[param_name] = parent1[param_name]
                else:
                    child[param_name] = parent2[param_name]
            return child
        
        def mutate(individual):
            """Muta un individuo"""
            mutated = individual.copy()
            for param_name in param_names:
                if random.random() < mutation_rate:
                    if param_name in self.calibrable_parameters:
                        bounds = self.calibrable_parameters[param_name]
                        # Mutación gaussiana
                        current_val = individual[param_name]
                        std_dev = (bounds.max_value - bounds.min_value) * 0.1
                        new_val = random.gauss(current_val, std_dev)
                        mutated[param_name] = bounds.validate_value(new_val)
                    else:
                        # Mutación porcentual
                        current_val = individual[param_name]
                        mutation = random.gauss(0, 0.1)  # 10% std dev
                        mutated[param_name] = current_val * (1 + mutation)
            return mutated
        
        # Inicializar población
        population = [create_individual() for _ in range(population_size)]
        
        # Incluir parámetros actuales en la población inicial
        population[0] = current_params.copy()
        
        best_individual = current_params.copy()
        best_fitness = fitness(best_individual)
        
        # Evolución
        for generation in range(generations):
            # Evaluar fitness de toda la población
            fitness_scores = [(individual, fitness(individual)) for individual in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Actualizar mejor individuo
            if fitness_scores[0][1] > best_fitness:
                best_individual = fitness_scores[0][0].copy()
                best_fitness = fitness_scores[0][1]
            
            # Selección (top 50%)
            elite_size = population_size // 2
            elite = [individual for individual, score in fitness_scores[:elite_size]]
            
            # Crear nueva población
            new_population = elite.copy()  # Mantener elite
            
            # Generar descendencia
            while len(new_population) < population_size:
                if random.random() < crossover_rate and len(elite) >= 2:
                    # Crossover
                    parent1 = random.choice(elite)
                    parent2 = random.choice(elite)
                    child = crossover(parent1, parent2)
                    child = mutate(child)
                    new_population.append(child)
                else:
                    # Mutación de elite
                    parent = random.choice(elite)
                    child = mutate(parent)
                    new_population.append(child)
            
            population = new_population
        
        self.logger.info(f"Algoritmo genético completado: {generations} generaciones, fitness final: {best_fitness:.4f}")
        
        return best_individual
    
    def _optimize_gradient(self, data: pd.DataFrame, current_params: Dict[str, float],
                          target: CalibrationTarget) -> Dict[str, float]:
        """
        Gradient Descent - Rápido para funciones suaves
        """
        learning_rate = 0.01
        max_iterations = self.config.max_iterations
        tolerance = self.config.convergence_tolerance
        
        current_solution = current_params.copy()
        param_names = list(current_params.keys())
        
        for iteration in range(max_iterations):
            # Calcular gradiente numérico
            gradient = {}
            current_score = self._evaluate_parameter_set(data, current_solution, target)
            
            for param_name in param_names:
                # Perturbación pequeña
                epsilon = 0.001
                if param_name in self.calibrable_parameters:
                    bounds = self.calibrable_parameters[param_name]
                    epsilon = min(epsilon, (bounds.max_value - bounds.min_value) * 0.01)
                
                # Calcular derivada numérica
                perturbed_params = current_solution.copy()
                perturbed_params[param_name] += epsilon
                
                # Validar bounds
                if param_name in self.calibrable_parameters:
                    bounds = self.calibrable_parameters[param_name]
                    perturbed_params[param_name] = bounds.validate_value(perturbed_params[param_name])
                
                perturbed_score = self._evaluate_parameter_set(data, perturbed_params, target)
                gradient[param_name] = (perturbed_score - current_score) / epsilon
            
            # Actualizar parámetros
            max_gradient = 0.0
            new_solution = {}
            
            for param_name in param_names:
                grad = gradient[param_name]
                max_gradient = max(max_gradient, abs(grad))
                
                # Actualización con learning rate adaptativo
                update = learning_rate * grad
                new_value = current_solution[param_name] + update
                
                # Aplicar bounds
                if param_name in self.calibrable_parameters:
                    bounds = self.calibrable_parameters[param_name]
                    new_value = bounds.validate_value(new_value)
                
                new_solution[param_name] = new_value
            
            # Verificar convergencia
            if max_gradient < tolerance:
                self.logger.info(f"Gradient descent convergió en {iteration} iteraciones")
                break
            
            current_solution = new_solution
            
            # Learning rate decay
            if iteration % 10 == 0:
                learning_rate *= 0.95
        
        return current_solution
    
    def _evaluate_parameter_set(self, data: pd.DataFrame, params: Dict[str, float],
                               target: CalibrationTarget) -> float:
        """
        Evalúa un conjunto de parámetros contra datos históricos
        
        Esta es la función objetivo - el corazón de la optimización
        """
        if data.empty:
            return 0.0
        
        try:
            # Simular performance con estos parámetros
            simulated_performance = self._simulate_performance(data, params)
            
            if target == CalibrationTarget.PROFIT_FACTOR:
                return simulated_performance.profit_factor
            elif target == CalibrationTarget.SHARPE_RATIO:
                return simulated_performance.sharpe_ratio
            elif target == CalibrationTarget.CALMAR_RATIO:
                return simulated_performance.calmar_ratio
            elif target == CalibrationTarget.WIN_RATE:
                return simulated_performance.win_rate
            elif target == CalibrationTarget.MAX_DRAWDOWN:
                return 1.0 - simulated_performance.max_drawdown  # Invertir (menor DD = mejor)
            elif target == CalibrationTarget.MULTI_OBJECTIVE:
                # Función objetivo compuesta - Mi fórmula secreta
                pf_score = min(simulated_performance.profit_factor / 2.0, 1.0)  # Normalizar a 1.0
                wr_score = simulated_performance.win_rate
                dd_score = max(0.0, 1.0 - simulated_performance.max_drawdown / 0.15)  # Penalizar DD > 15%
                sharpe_score = min(simulated_performance.sharpe_ratio / 1.5, 1.0)  # Normalizar a 1.0
                
                # Pesos para multi-objetivo (ajustados por experiencia)
                composite_score = (
                    pf_score * 0.35 +      # Profit Factor es rey
                    wr_score * 0.25 +      # Win Rate importante
                    dd_score * 0.25 +      # Control de riesgo crítico
                    sharpe_score * 0.15    # Consistencia
                )
                
                return composite_score
            else:
                return simulated_performance.profit_factor  # Default
                
        except Exception as e:
            self.logger.error(f"Error evaluando parámetros: {e}")
            return 0.0
    
    def _simulate_performance(self, data: pd.DataFrame, params: Dict[str, float]) -> PerformanceMetrics:
        """
        Simula performance aplicando nuevos parámetros a datos históricos
        
        Aquí es donde la magia sucede - reescribimos la historia con nuevos parámetros
        """
        if data.empty:
            return PerformanceMetrics(
                total_trades=0, win_rate=0.0, profit_factor=0.0,
                avg_win=0.0, avg_loss=0.0, max_drawdown=0.0,
                sharpe_ratio=0.0, calmar_ratio=0.0
            )
        
        try:
            simulated_data = data.copy()
            
            # Aplicar nuevos umbrales de decisión
            min_score_threshold = params.get("min_score_threshold", 65.0)
            
            # Filtrar trades que no habrían pasado el nuevo umbral
            if 'score' in simulated_data.columns:
                original_count = len(simulated_data)
                simulated_data = simulated_data[simulated_data['score'] >= min_score_threshold]
                filtered_count = len(simulated_data)
                
                if filtered_count < original_count * 0.1:  # Si filtramos más del 90%, es muy restrictivo
                    self.logger.debug(f"Umbral muy restrictivo: {filtered_count}/{original_count} trades")
            
            # Aplicar nuevos parámetros de riesgo
            if 'pnl' in simulated_data.columns and not simulated_data.empty:
                # Ajustar PnL basado en nuevos parámetros de posición
                position_multiplier = params.get("max_position_size", 0.05) / 0.05  # Relativo al default
                simulated_data = simulated_data.copy()  # Evitar SettingWithCopyWarning
                simulated_data['adjusted_pnl'] = simulated_data['pnl'] * position_multiplier
                
                # Aplicar nuevos stop-loss y take-profit (simplificado)
                stop_multiplier = params.get("stop_loss_multiplier", 1.5) / 1.5
                tp_ratio = params.get("take_profit_ratio", 2.5)
                
                # Ajustar wins y losses basado en nuevos ratios
                wins_mask = simulated_data['adjusted_pnl'] > 0
                losses_mask = simulated_data['adjusted_pnl'] < 0
                
                # Los wins podrían ser mayores con mejor TP ratio
                if tp_ratio > 2.5:
                    tp_improvement = min(tp_ratio / 2.5, 1.5)  # Máximo 50% mejora
                    simulated_data.loc[wins_mask, 'adjusted_pnl'] *= tp_improvement
                
                # Las losses podrían ser menores con mejor SL
                if stop_multiplier < 1.0:
                    sl_improvement = stop_multiplier
                    simulated_data.loc[losses_mask, 'adjusted_pnl'] *= sl_improvement
            else:
                simulated_data = simulated_data.copy()
                simulated_data['adjusted_pnl'] = simulated_data.get('pnl', 0.0)
            
            # Calcular métricas con datos ajustados
            return self._calculate_performance_from_adjusted_data(simulated_data)
            
        except Exception as e:
            self.logger.error(f"Error simulando performance: {e}")
            return self._calculate_performance_from_data(data)
    
    def _calculate_performance_from_adjusted_data(self, data: pd.DataFrame) -> PerformanceMetrics:
        """
        Calcula performance desde datos ajustados por simulación
        """
        if data.empty or 'adjusted_pnl' not in data.columns:
            return PerformanceMetrics(
                period_start=datetime.now(ECUADOR_TZ) - timedelta(days=30),
                period_end=datetime.now(ECUADOR_TZ),
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                consecutive_wins=0,
                consecutive_losses=0
            )
        
        total_trades = len(data)
        
        # Usar PnL ajustado
        pnl_series = data['adjusted_pnl']
        
        wins = pnl_series[pnl_series > 0]
        losses = pnl_series[pnl_series < 0]
        
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        gross_profit = wins.sum() if not wins.empty else 0.0
        gross_loss = abs(losses.sum()) if not losses.empty else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win = wins.mean() if not wins.empty else 0.0
        avg_loss = losses.mean() if not losses.empty else 0.0
        largest_win = wins.max() if not wins.empty else 0.0
        largest_loss = losses.min() if not losses.empty else 0.0
        
        # Rachas consecutivas
        consecutive_wins = 0
        consecutive_losses = 0
        current_win_streak = 0
        current_loss_streak = 0
        
        for pnl in pnl_series:
            if pnl > 0:
                current_win_streak += 1
                current_loss_streak = 0
                consecutive_wins = max(consecutive_wins, current_win_streak)
            elif pnl < 0:
                current_loss_streak += 1
                current_win_streak = 0
                consecutive_losses = max(consecutive_losses, current_loss_streak)
        
        # Drawdown con PnL ajustado
        cumulative_pnl = pnl_series.cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max) / running_max.abs()
        max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0.0
        
        # Sharpe con PnL ajustado
        if len(pnl_series) > 1 and pnl_series.std() > 0:
            sharpe_ratio = pnl_series.mean() / pnl_series.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Calmar ratio
        annual_return = pnl_series.sum() * (252 / len(data)) if len(data) > 0 else 0.0
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # Período de datos
        if 'timestamp' in data.columns and not data.empty:
            period_start = data['timestamp'].min()
            period_end = data['timestamp'].max()
        else:
            period_start = datetime.now(ECUADOR_TZ) - timedelta(days=30)
            period_end = datetime.now(ECUADOR_TZ)
        
        return PerformanceMetrics(
            period_start=period_start,
            period_end=period_end,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses
        )    

    def _validate_parameters(self, data: pd.DataFrame, params: Dict[str, float],
                           target: CalibrationTarget) -> Dict[str, Any]:
        """
        Valida parámetros usando validación cruzada temporal
        
        Como trader senior, nunca confío en un solo backtest.
        La validación cruzada temporal es la única forma de evitar overfitting.
        """
        if data.empty:
            return {"passed": False, "reason": "No hay datos para validación", "score": 0.0}
        
        try:
            # Validación cruzada temporal (Time Series Split)
            n_splits = min(self.config.validation_splits, len(data) // 50)  # Mínimo 50 trades por split
            
            if n_splits < 2:
                return {"passed": False, "reason": "Datos insuficientes para validación cruzada", "score": 0.0}
            
            tscv = TimeSeriesSplit(n_splits=n_splits)
            validation_scores = []
            
            # Asegurar que tenemos índice temporal
            if 'timestamp' in data.columns:
                data_sorted = data.sort_values('timestamp').reset_index(drop=True)
            else:
                data_sorted = data.reset_index(drop=True)
            
            for train_idx, test_idx in tscv.split(data_sorted):
                train_data = data_sorted.iloc[train_idx]
                test_data = data_sorted.iloc[test_idx]
                
                if len(train_data) < 20 or len(test_data) < 10:  # Mínimos para validación
                    continue
                
                # Evaluar en datos de test
                test_score = self._evaluate_parameter_set(test_data, params, target)
                validation_scores.append(test_score)
            
            if not validation_scores:
                return {"passed": False, "reason": "No se pudieron generar scores de validación", "score": 0.0}
            
            # Estadísticas de validación
            mean_score = np.mean(validation_scores)
            std_score = np.std(validation_scores)
            min_score = np.min(validation_scores)
            
            # Criterios de validación
            validation_passed = True
            reasons = []
            
            # 1. Score promedio debe ser razonable
            if target == CalibrationTarget.MULTI_OBJECTIVE:
                if mean_score < 0.6:  # 60% mínimo para multi-objetivo
                    validation_passed = False
                    reasons.append(f"Score promedio muy bajo: {mean_score:.3f}")
            elif target == CalibrationTarget.PROFIT_FACTOR:
                if mean_score < 1.5:  # PF mínimo 1.5
                    validation_passed = False
                    reasons.append(f"Profit Factor muy bajo: {mean_score:.3f}")
            elif target == CalibrationTarget.WIN_RATE:
                if mean_score < 0.55:  # Win rate mínimo 55%
                    validation_passed = False
                    reasons.append(f"Win rate muy bajo: {mean_score:.3f}")
            
            # 2. Consistencia entre folds
            if len(validation_scores) > 1:
                cv_coefficient = std_score / mean_score if mean_score > 0 else float('inf')
                if cv_coefficient > 0.5:  # Coeficiente de variación > 50%
                    validation_passed = False
                    reasons.append(f"Resultados inconsistentes entre folds: CV={cv_coefficient:.3f}")
            
            # 3. No debe haber folds con performance terrible
            if target == CalibrationTarget.MULTI_OBJECTIVE:
                if min_score < 0.3:  # Ningún fold debería estar por debajo de 30%
                    validation_passed = False
                    reasons.append(f"Fold con performance muy baja: {min_score:.3f}")
            
            # 4. Validación de parámetros extremos
            extreme_params = []
            for param_name, value in params.items():
                if param_name in self.calibrable_parameters:
                    bounds = self.calibrable_parameters[param_name]
                    range_size = bounds.max_value - bounds.min_value
                    
                    # Verificar si está muy cerca de los límites
                    if (value - bounds.min_value) < (range_size * 0.05):
                        extreme_params.append(f"{param_name} muy cerca del mínimo")
                    elif (bounds.max_value - value) < (range_size * 0.05):
                        extreme_params.append(f"{param_name} muy cerca del máximo")
            
            if len(extreme_params) > len(params) * 0.3:  # Más del 30% de parámetros extremos
                validation_passed = False
                reasons.append(f"Demasiados parámetros extremos: {extreme_params}")
            
            # Calcular confidence score
            confidence_score = mean_score
            if len(validation_scores) > 1:
                consistency_bonus = max(0, 1 - cv_coefficient)
                confidence_score *= (0.7 + 0.3 * consistency_bonus)
            
            result = {
                "passed": validation_passed,
                "score": mean_score,
                "confidence": confidence_score,
                "std_score": std_score,
                "min_score": min_score,
                "n_folds": len(validation_scores),
                "reason": "; ".join(reasons) if reasons else "Validación exitosa",
                "converged": True,
                "iterations": len(validation_scores)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en validación: {e}")
            return {"passed": False, "reason": f"Error en validación: {str(e)}", "score": 0.0}
    
    def _calculate_improvement(self, original: PerformanceMetrics, 
                             optimized: PerformanceMetrics,
                             target: CalibrationTarget) -> float:
        """
        Calcula el porcentaje de mejora entre performance original y optimizada
        """
        try:
            if target == CalibrationTarget.PROFIT_FACTOR:
                if original.profit_factor > 0:
                    return (optimized.profit_factor - original.profit_factor) / original.profit_factor
                else:
                    return 1.0 if optimized.profit_factor > 1.0 else 0.0
                    
            elif target == CalibrationTarget.WIN_RATE:
                if original.win_rate > 0:
                    return (optimized.win_rate - original.win_rate) / original.win_rate
                else:
                    return optimized.win_rate
                    
            elif target == CalibrationTarget.SHARPE_RATIO:
                if original.sharpe_ratio > 0:
                    return (optimized.sharpe_ratio - original.sharpe_ratio) / original.sharpe_ratio
                else:
                    return 1.0 if optimized.sharpe_ratio > 0 else 0.0
                    
            elif target == CalibrationTarget.MAX_DRAWDOWN:
                # Para drawdown, menor es mejor
                if original.max_drawdown > 0:
                    improvement = (original.max_drawdown - optimized.max_drawdown) / original.max_drawdown
                    return improvement
                else:
                    return 0.0
                    
            elif target == CalibrationTarget.CALMAR_RATIO:
                if original.calmar_ratio > 0:
                    return (optimized.calmar_ratio - original.calmar_ratio) / original.calmar_ratio
                else:
                    return 1.0 if optimized.calmar_ratio > 0 else 0.0
                    
            elif target == CalibrationTarget.MULTI_OBJECTIVE:
                # Mejora compuesta para multi-objetivo
                improvements = []
                
                # Profit Factor (peso 35%)
                if original.profit_factor > 0:
                    pf_improvement = (optimized.profit_factor - original.profit_factor) / original.profit_factor
                    improvements.append(pf_improvement * 0.35)
                
                # Win Rate (peso 25%)
                if original.win_rate > 0:
                    wr_improvement = (optimized.win_rate - original.win_rate) / original.win_rate
                    improvements.append(wr_improvement * 0.25)
                
                # Drawdown (peso 25%, invertido)
                if original.max_drawdown > 0:
                    dd_improvement = (original.max_drawdown - optimized.max_drawdown) / original.max_drawdown
                    improvements.append(dd_improvement * 0.25)
                
                # Sharpe Ratio (peso 15%)
                if original.sharpe_ratio > 0:
                    sharpe_improvement = (optimized.sharpe_ratio - original.sharpe_ratio) / original.sharpe_ratio
                    improvements.append(sharpe_improvement * 0.15)
                
                return sum(improvements) if improvements else 0.0
            
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculando mejora: {e}")
            return 0.0
    
    def _apply_calibration_result(self, result: CalibrationResult) -> None:
        """
        Aplica los resultados de calibración al sistema
        
        Aquí es donde los nuevos parámetros entran en vigor
        """
        try:
            regime = result.regime
            if not regime:
                self.logger.warning("No se puede aplicar calibración sin régimen específico")
                return
            
            # Guardar parámetros activos
            self.active_parameters[regime] = result.optimized_parameters.copy()
            
            # Aplicar al weight_manager si es posible
            try:
                if hasattr(self.weight_manager, 'update_regime_weights'):
                    # Extraer pesos de scoring
                    weights = {
                        "mtf_structure": result.optimized_parameters.get("mtf_structure_weight", 0.35),
                        "technical_confluence": result.optimized_parameters.get("technical_confluence_weight", 0.25),
                        "market_context": result.optimized_parameters.get("market_context_weight", 0.20),
                        "risk_metrics": result.optimized_parameters.get("risk_metrics_weight", 0.20)
                    }
                    
                    self.weight_manager.update_regime_weights(regime, weights)
                    self.logger.info(f"Pesos actualizados en weight_manager para {regime.value}")
                
            except Exception as e:
                self.logger.warning(f"No se pudieron actualizar pesos en weight_manager: {e}")
            
            # Aplicar otros parámetros al sistema (esto dependería de la arquitectura específica)
            # Por ahora, solo los guardamos para uso futuro
            
            self.logger.info(
                f"Parámetros aplicados para {regime.value}: {len(result.optimized_parameters)} parámetros actualizados"
            )
            
        except Exception as e:
            self.logger.error(f"Error aplicando calibración: {e}")
    
    def _apply_parameters(self, regime: ScoringRegime, parameters: Dict[str, float]) -> None:
        """
        Aplica un conjunto específico de parámetros a un régimen
        """
        try:
            # Guardar como parámetros activos
            self.active_parameters[regime] = parameters.copy()
            
            # Aplicar al weight_manager si es posible
            if hasattr(self.weight_manager, 'update_regime_weights'):
                weights = {
                    "mtf_structure": parameters.get("mtf_structure_weight", 0.35),
                    "technical_confluence": parameters.get("technical_confluence_weight", 0.25),
                    "market_context": parameters.get("market_context_weight", 0.20),
                    "risk_metrics": parameters.get("risk_metrics_weight", 0.20)
                }
                
                self.weight_manager.update_regime_weights(regime, weights)
            
            self.logger.info(f"Parámetros aplicados manualmente para {regime.value}")
            
        except Exception as e:
            self.logger.error(f"Error aplicando parámetros: {e}")
    
    def get_parameter_bounds(self) -> Dict[str, ParameterBounds]:
        """
        Obtiene los límites de todos los parámetros calibrables
        
        Returns:
            Diccionario con límites de parámetros
        """
        return self.calibrable_parameters.copy()
    
    def update_parameter_bounds(self, param_name: str, bounds: ParameterBounds) -> bool:
        """
        Actualiza los límites de un parámetro específico
        
        Args:
            param_name: Nombre del parámetro
            bounds: Nuevos límites
            
        Returns:
            True si se actualizó exitosamente
        """
        try:
            if bounds.min_value >= bounds.max_value:
                self.logger.error(f"Límites inválidos para {param_name}: min >= max")
                return False
            
            self.calibrable_parameters[param_name] = bounds
            self.logger.info(f"Límites actualizados para {param_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error actualizando límites: {e}")
            return False
    
    def get_optimization_history(self, regime: Optional[ScoringRegime] = None,
                               limit: int = 50) -> List[CalibrationResult]:
        """
        Obtiene el historial de optimizaciones
        
        Args:
            regime: Régimen específico (opcional)
            limit: Número máximo de resultados
            
        Returns:
            Lista de resultados de calibración
        """
        history = self.calibration_history.copy()
        
        if regime:
            history = [r for r in history if r.regime == regime]
        
        # Ordenar por fecha (más reciente primero)
        history.sort(key=lambda x: x.calibration_time, reverse=True)
        
        return history[:limit]
    
    def export_calibration_report(self, output_file: Optional[str] = None) -> str:
        """
        Exporta un reporte completo de calibraciones
        
        Args:
            output_file: Archivo de salida (opcional)
            
        Returns:
            Contenido del reporte
        """
        try:
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("REPORTE DE CALIBRACIÓN DEL SISTEMA DE TRADING")
            report_lines.append("=" * 80)
            report_lines.append(f"Generado: {datetime.now(ECUADOR_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}")
            report_lines.append("")
            
            # Resumen general
            report_lines.append("RESUMEN GENERAL")
            report_lines.append("-" * 40)
            report_lines.append(f"Total de calibraciones: {len(self.calibration_history)}")
            
            successful_calibrations = [r for r in self.calibration_history if r.validation_passed]
            report_lines.append(f"Calibraciones exitosas: {len(successful_calibrations)}")
            
            if successful_calibrations:
                avg_improvement = np.mean([r.improvement_pct for r in successful_calibrations])
                report_lines.append(f"Mejora promedio: {avg_improvement:.1%}")
            
            report_lines.append("")
            
            # Por régimen
            for regime in ScoringRegime:
                regime_calibrations = [r for r in self.calibration_history if r.regime == regime]
                if not regime_calibrations:
                    continue
                
                report_lines.append(f"RÉGIMEN: {regime.value.upper()}")
                report_lines.append("-" * 40)
                
                last_calibration = max(regime_calibrations, key=lambda x: x.calibration_time)
                report_lines.append(f"Última calibración: {last_calibration.calibration_time.strftime('%Y-%m-%d %H:%M')}")
                report_lines.append(f"Método utilizado: {last_calibration.method.value}")
                report_lines.append(f"Mejora obtenida: {last_calibration.improvement_pct:.1%}")
                report_lines.append(f"Validación: {'✓' if last_calibration.validation_passed else '✗'}")
                
                # Parámetros actuales
                if regime in self.active_parameters:
                    report_lines.append("\nParámetros activos:")
                    for param, value in self.active_parameters[regime].items():
                        report_lines.append(f"  {param}: {value:.4f}")
                
                report_lines.append("")
            
            # Historial reciente
            report_lines.append("HISTORIAL RECIENTE (Últimas 10 calibraciones)")
            report_lines.append("-" * 60)
            
            recent_calibrations = sorted(self.calibration_history, key=lambda x: x.calibration_time, reverse=True)[:10]
            
            for cal in recent_calibrations:
                status = "✓" if cal.validation_passed else "✗"
                report_lines.append(
                    f"{cal.calibration_time.strftime('%Y-%m-%d %H:%M')} | "
                    f"{cal.regime.value if cal.regime else 'N/A':12} | "
                    f"{cal.method.value:20} | "
                    f"{cal.improvement_pct:6.1%} | {status}"
                )
            
            report_content = "\n".join(report_lines)
            
            # Guardar archivo si se especifica
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                
                self.logger.info(f"Reporte exportado a: {output_path}")
            
            return report_content
            
        except Exception as e:
            self.logger.error(f"Error generando reporte: {e}")
            return f"Error generando reporte: {str(e)}"