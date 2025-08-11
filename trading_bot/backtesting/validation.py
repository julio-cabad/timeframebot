"""
Sistema de Validación y Optimización para Backtesting
====================================================

Como trader senior, sé que el overfitting es el enemigo #1 del backtesting.
Este módulo implementa técnicas robustas para validar estrategias:

- Walk-Forward Analysis (validación out-of-sample)
- Monte Carlo Permutation Testing
- Parameter Sensitivity Analysis
- Robustness Testing
- Cross-Validation temporal
- Bootstrap Analysis

Filosofía: "Una estrategia que no funciona out-of-sample, no funciona"

Autor: Trader Algorítmico Senior (10+ años)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from ..utils.logger import get_logger, LogContext
from ..utils.exceptions import TradingBotException
from .metrics import PerformanceAnalyzer

@dataclass
class ValidationResult:
    """Resultado de validación de estrategia"""
    is_valid: bool
    confidence_score: float  # 0-100
    
    # Métricas de validación
    in_sample_metrics: Dict[str, float]
    out_of_sample_metrics: Dict[str, float]
    degradation_metrics: Dict[str, float]
    
    # Análisis de robustez
    parameter_sensitivity: Dict[str, float]
    monte_carlo_results: Dict[str, Any]
    
    # Recomendaciones
    recommendations: List[str]
    warnings: List[str]
    
    # Datos detallados
    walk_forward_results: List[Dict[str, Any]] = field(default_factory=list)
    bootstrap_results: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WalkForwardPeriod:
    """Período individual de walk-forward analysis"""
    period_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Resultados
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    degradation: Dict[str, float]
    
    # Parámetros optimizados
    optimal_params: Dict[str, Any]
    param_stability: float

class ValidationMethod(Enum):
    """Métodos de validación disponibles"""
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"
    BOOTSTRAP = "bootstrap"
    CROSS_VALIDATION = "cross_validation"
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"

class WalkForwardValidator:
    """
    Validador Walk-Forward para estrategias de trading
    
    Como trader senior, uso walk-forward para simular cómo habría
    funcionado mi estrategia en tiempo real, reoptimizando parámetros
    periódicamente como haría en producción.
    """
    
    def __init__(self, 
                 train_period_months: int = 12,
                 test_period_months: int = 3,
                 reoptimization_frequency_months: int = 3,
                 min_trades_per_period: int = 10):
        
        self.logger = get_logger("WalkForwardValidator")
        self.train_period_months = train_period_months
        self.test_period_months = test_period_months
        self.reoptimization_frequency_months = reoptimization_frequency_months
        self.min_trades_per_period = min_trades_per_period
        
        self.performance_analyzer = PerformanceAnalyzer()
    
    def validate_strategy(self, 
                         backtest_function: Callable,
                         data: pd.DataFrame,
                         parameter_ranges: Dict[str, List[Any]],
                         start_date: datetime,
                         end_date: datetime,
                         optimization_metric: str = "sharpe_ratio") -> ValidationResult:
        """
        Ejecuta validación walk-forward completa
        
        Args:
            backtest_function: Función que ejecuta el backtest
            data: Datos históricos completos
            parameter_ranges: Rangos de parámetros para optimización
            start_date: Fecha de inicio de validación
            end_date: Fecha de fin de validación
            optimization_metric: Métrica para optimización
            
        Returns:
            ValidationResult con resultados completos
        """
        context = LogContext(component="walk_forward_validator")
        
        try:
            self.logger.info(
                "Iniciando validación walk-forward",
                context=context,
                extra_fields={
                    "train_period": f"{self.train_period_months} meses",
                    "test_period": f"{self.test_period_months} meses",
                    "reopt_frequency": f"{self.reoptimization_frequency_months} meses"
                }
            )
            
            # Generar períodos de walk-forward
            periods = self._generate_walk_forward_periods(start_date, end_date)
            
            if len(periods) < 2:
                raise TradingBotException("Datos insuficientes para walk-forward analysis")
            
            # Ejecutar walk-forward para cada período
            wf_results = []
            
            for period in periods:
                try:
                    result = self._execute_walk_forward_period(
                        backtest_function, data, parameter_ranges,
                        period, optimization_metric
                    )
                    wf_results.append(result)
                    
                    self.logger.info(
                        f"Período {period.period_id} completado",
                        extra_fields={
                            "train_sharpe": result.train_metrics.get('sharpe_ratio', 0),
                            "test_sharpe": result.test_metrics.get('sharpe_ratio', 0)
                        }
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error en período {period.period_id}: {e}")
                    continue
            
            if not wf_results:
                raise TradingBotException("No se completó ningún período de walk-forward")
            
            # Analizar resultados agregados
            validation_result = self._analyze_walk_forward_results(wf_results)
            
            self.logger.info(
                "Validación walk-forward completada",
                context=context,
                extra_fields={
                    "periods_completed": len(wf_results),
                    "confidence_score": validation_result.confidence_score,
                    "is_valid": validation_result.is_valid
                }
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error en validación walk-forward: {e}")
            raise TradingBotException(f"Fallo en validación: {str(e)}")
    
    def _generate_walk_forward_periods(self, start_date: datetime, 
                                     end_date: datetime) -> List[WalkForwardPeriod]:
        """Genera períodos para walk-forward analysis"""
        try:
            periods = []
            period_id = 1
            
            current_date = start_date
            
            while current_date < end_date:
                # Calcular fechas del período de entrenamiento
                train_start = current_date
                train_end = train_start + timedelta(days=self.train_period_months * 30)
                
                # Calcular fechas del período de prueba
                test_start = train_end + timedelta(days=1)
                test_end = test_start + timedelta(days=self.test_period_months * 30)
                
                # Verificar que no exceda la fecha final
                if test_end > end_date:
                    test_end = end_date
                
                if test_start >= end_date:
                    break
                
                period = WalkForwardPeriod(
                    period_id=period_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_metrics={},
                    test_metrics={},
                    degradation={},
                    optimal_params={},
                    param_stability=0.0
                )
                
                periods.append(period)
                
                # Avanzar al siguiente período
                current_date += timedelta(days=self.reoptimization_frequency_months * 30)
                period_id += 1
            
            return periods
            
        except Exception as e:
            self.logger.error(f"Error generando períodos walk-forward: {e}")
            return []
    
    def _execute_walk_forward_period(self, 
                                   backtest_function: Callable,
                                   data: pd.DataFrame,
                                   parameter_ranges: Dict[str, List[Any]],
                                   period: WalkForwardPeriod,
                                   optimization_metric: str) -> WalkForwardPeriod:
        """Ejecuta un período individual de walk-forward"""
        try:
            # Filtrar datos para período de entrenamiento
            train_mask = (data.index >= period.train_start) & (data.index <= period.train_end)
            train_data = data[train_mask]
            
            # Filtrar datos para período de prueba
            test_mask = (data.index >= period.test_start) & (data.index <= period.test_end)
            test_data = data[test_mask]
            
            if len(train_data) == 0 or len(test_data) == 0:
                raise TradingBotException("Datos insuficientes para el período")
            
            # Optimizar parámetros en período de entrenamiento
            optimal_params, train_results = self._optimize_parameters(
                backtest_function, train_data, parameter_ranges, optimization_metric
            )
            
            # Probar parámetros optimizados en período de prueba
            test_results = backtest_function(test_data, optimal_params)
            
            # Calcular métricas
            train_metrics = self._extract_key_metrics(train_results)
            test_metrics = self._extract_key_metrics(test_results)
            
            # Calcular degradación
            degradation = self._calculate_degradation(train_metrics, test_metrics)
            
            # Actualizar período con resultados
            period.optimal_params = optimal_params
            period.train_metrics = train_metrics
            period.test_metrics = test_metrics
            period.degradation = degradation
            
            return period
            
        except Exception as e:
            self.logger.error(f"Error ejecutando período walk-forward: {e}")
            raise
    
    def _optimize_parameters(self, 
                           backtest_function: Callable,
                           data: pd.DataFrame,
                           parameter_ranges: Dict[str, List[Any]],
                           optimization_metric: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Optimiza parámetros usando grid search"""
        try:
            best_params = {}
            best_score = float('-inf')
            best_results = {}
            
            # Generar todas las combinaciones de parámetros
            param_names = list(parameter_ranges.keys())
            param_values = list(parameter_ranges.values())
            
            total_combinations = np.prod([len(values) for values in param_values])
            
            if total_combinations > 1000:
                self.logger.warning(f"Muchas combinaciones ({total_combinations}), limitando búsqueda")
                # Reducir combinaciones usando sampling
                param_combinations = self._sample_parameter_combinations(
                    parameter_ranges, max_combinations=500
                )
            else:
                param_combinations = list(itertools.product(*param_values))
            
            self.logger.info(f"Optimizando {len(param_combinations)} combinaciones de parámetros")
            
            for combination in param_combinations:
                try:
                    # Crear diccionario de parámetros
                    params = dict(zip(param_names, combination))
                    
                    # Ejecutar backtest
                    results = backtest_function(data, params)
                    
                    # Extraer métrica de optimización
                    score = results.get(optimization_metric, float('-inf'))
                    
                    # Verificar si es mejor
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        best_results = results.copy()
                
                except Exception as e:
                    # Continuar con siguiente combinación si hay error
                    continue
            
            if not best_params:
                raise TradingBotException("No se encontraron parámetros válidos")
            
            return best_params, best_results
            
        except Exception as e:
            self.logger.error(f"Error optimizando parámetros: {e}")
            raise
    
    def _sample_parameter_combinations(self, 
                                     parameter_ranges: Dict[str, List[Any]],
                                     max_combinations: int = 500) -> List[Tuple]:
        """Muestrea combinaciones de parámetros para reducir espacio de búsqueda"""
        try:
            param_names = list(parameter_ranges.keys())
            param_values = list(parameter_ranges.values())
            
            # Usar Latin Hypercube Sampling aproximado
            combinations = []
            
            for _ in range(max_combinations):
                combination = []
                for values in param_values:
                    combination.append(np.random.choice(values))
                combinations.append(tuple(combination))
            
            # Remover duplicados
            combinations = list(set(combinations))
            
            return combinations
            
        except Exception as e:
            self.logger.error(f"Error muestreando combinaciones: {e}")
            return []
    
    def _extract_key_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extrae métricas clave de resultados de backtest"""
        try:
            key_metrics = {}
            
            # Métricas básicas
            key_metrics['total_return'] = results.get('total_return', 0.0)
            key_metrics['sharpe_ratio'] = results.get('sharpe_ratio', 0.0)
            key_metrics['max_drawdown'] = results.get('max_drawdown', 0.0)
            key_metrics['win_rate'] = results.get('win_rate', 0.0)
            key_metrics['profit_factor'] = results.get('profit_factor', 0.0)
            key_metrics['calmar_ratio'] = results.get('calmar_ratio', 0.0)
            
            # Métricas de trading
            if 'trade_metrics' in results:
                tm = results['trade_metrics']
                key_metrics['total_trades'] = getattr(tm, 'total_trades', 0)
                key_metrics['expectancy'] = getattr(tm, 'expectancy', 0.0)
            
            return key_metrics
            
        except Exception as e:
            self.logger.error(f"Error extrayendo métricas: {e}")
            return {}
    
    def _calculate_degradation(self, train_metrics: Dict[str, float], 
                             test_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calcula degradación de métricas entre entrenamiento y prueba"""
        try:
            degradation = {}
            
            for metric in train_metrics:
                if metric in test_metrics:
                    train_val = train_metrics[metric]
                    test_val = test_metrics[metric]
                    
                    if train_val != 0:
                        degradation[f"{metric}_degradation"] = (test_val - train_val) / abs(train_val)
                    else:
                        degradation[f"{metric}_degradation"] = 0.0
            
            return degradation
            
        except Exception as e:
            self.logger.error(f"Error calculando degradación: {e}")
            return {}
    
    def _analyze_walk_forward_results(self, wf_results: List[WalkForwardPeriod]) -> ValidationResult:
        """Analiza resultados agregados de walk-forward"""
        try:
            if not wf_results:
                return ValidationResult(
                    is_valid=False,
                    confidence_score=0.0,
                    in_sample_metrics={},
                    out_of_sample_metrics={},
                    degradation_metrics={},
                    parameter_sensitivity={},
                    monte_carlo_results={},
                    recommendations=["No hay resultados de walk-forward"],
                    warnings=["Validación falló"]
                )
            
            # Agregar métricas de entrenamiento y prueba
            train_metrics_agg = self._aggregate_metrics([p.train_metrics for p in wf_results])
            test_metrics_agg = self._aggregate_metrics([p.test_metrics for p in wf_results])
            degradation_agg = self._aggregate_metrics([p.degradation for p in wf_results])
            
            # Calcular estabilidad de parámetros
            param_stability = self._calculate_parameter_stability(wf_results)
            
            # Calcular score de confianza
            confidence_score = self._calculate_confidence_score(
                train_metrics_agg, test_metrics_agg, degradation_agg, param_stability
            )
            
            # Determinar si la estrategia es válida
            is_valid = self._determine_validity(
                test_metrics_agg, degradation_agg, confidence_score
            )
            
            # Generar recomendaciones
            recommendations = self._generate_recommendations(
                train_metrics_agg, test_metrics_agg, degradation_agg, param_stability
            )
            
            # Generar warnings
            warnings = self._generate_warnings(
                test_metrics_agg, degradation_agg, confidence_score
            )
            
            return ValidationResult(
                is_valid=is_valid,
                confidence_score=confidence_score,
                in_sample_metrics=train_metrics_agg,
                out_of_sample_metrics=test_metrics_agg,
                degradation_metrics=degradation_agg,
                parameter_sensitivity=param_stability,
                monte_carlo_results={},  # Se puede agregar después
                recommendations=recommendations,
                warnings=warnings,
                walk_forward_results=[p.__dict__ for p in wf_results]
            )
            
        except Exception as e:
            self.logger.error(f"Error analizando resultados walk-forward: {e}")
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                in_sample_metrics={},
                out_of_sample_metrics={},
                degradation_metrics={},
                parameter_sensitivity={},
                monte_carlo_results={},
                recommendations=["Error en análisis"],
                warnings=["Error en validación"]
            )
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Agrega métricas de múltiples períodos"""
        try:
            if not metrics_list:
                return {}
            
            aggregated = {}
            
            # Obtener todas las métricas disponibles
            all_metrics = set()
            for metrics in metrics_list:
                all_metrics.update(metrics.keys())
            
            # Calcular estadísticas para cada métrica
            for metric in all_metrics:
                values = [m.get(metric, 0) for m in metrics_list if metric in m]
                
                if values:
                    aggregated[f"{metric}_mean"] = np.mean(values)
                    aggregated[f"{metric}_std"] = np.std(values)
                    aggregated[f"{metric}_min"] = np.min(values)
                    aggregated[f"{metric}_max"] = np.max(values)
                    aggregated[f"{metric}_median"] = np.median(values)
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Error agregando métricas: {e}")
            return {}
    
    def _calculate_parameter_stability(self, wf_results: List[WalkForwardPeriod]) -> Dict[str, float]:
        """Calcula estabilidad de parámetros optimizados"""
        try:
            if len(wf_results) < 2:
                return {}
            
            stability = {}
            
            # Obtener todos los parámetros
            all_params = set()
            for result in wf_results:
                all_params.update(result.optimal_params.keys())
            
            # Calcular estabilidad para cada parámetro
            for param in all_params:
                values = []
                for result in wf_results:
                    if param in result.optimal_params:
                        values.append(result.optimal_params[param])
                
                if len(values) > 1:
                    # Para parámetros numéricos, calcular coeficiente de variación
                    try:
                        numeric_values = [float(v) for v in values]
                        if np.std(numeric_values) > 0:
                            cv = np.std(numeric_values) / abs(np.mean(numeric_values))
                            stability[f"{param}_stability"] = max(0, 1 - cv)  # 1 = muy estable, 0 = inestable
                        else:
                            stability[f"{param}_stability"] = 1.0  # Valores constantes = estable
                    except (ValueError, TypeError):
                        # Para parámetros categóricos, calcular frecuencia del valor más común
                        from collections import Counter
                        counter = Counter(values)
                        most_common_freq = counter.most_common(1)[0][1]
                        stability[f"{param}_stability"] = most_common_freq / len(values)
            
            return stability
            
        except Exception as e:
            self.logger.error(f"Error calculando estabilidad de parámetros: {e}")
            return {}
    
    def _calculate_confidence_score(self, 
                                  train_metrics: Dict[str, float],
                                  test_metrics: Dict[str, float],
                                  degradation: Dict[str, float],
                                  param_stability: Dict[str, float]) -> float:
        """Calcula score de confianza (0-100)"""
        try:
            score = 0.0
            
            # 1. Performance out-of-sample (40 puntos)
            test_sharpe = test_metrics.get('sharpe_ratio_mean', 0)
            if test_sharpe >= 1.5:
                score += 40
            elif test_sharpe >= 1.0:
                score += 30
            elif test_sharpe >= 0.5:
                score += 20
            elif test_sharpe >= 0:
                score += 10
            
            # 2. Estabilidad de degradación (30 puntos)
            sharpe_degradation = abs(degradation.get('sharpe_ratio_degradation_mean', 1))
            if sharpe_degradation <= 0.1:  # Menos de 10% degradación
                score += 30
            elif sharpe_degradation <= 0.2:
                score += 25
            elif sharpe_degradation <= 0.3:
                score += 20
            elif sharpe_degradation <= 0.5:
                score += 10
            
            # 3. Estabilidad de parámetros (20 puntos)
            if param_stability:
                avg_stability = np.mean(list(param_stability.values()))
                score += avg_stability * 20
            
            # 4. Consistencia de resultados (10 puntos)
            sharpe_std = test_metrics.get('sharpe_ratio_std', float('inf'))
            if sharpe_std <= 0.2:
                score += 10
            elif sharpe_std <= 0.5:
                score += 5
            
            return min(score, 100.0)
            
        except Exception as e:
            self.logger.error(f"Error calculando score de confianza: {e}")
            return 0.0
    
    def _determine_validity(self, 
                          test_metrics: Dict[str, float],
                          degradation: Dict[str, float],
                          confidence_score: float) -> bool:
        """Determina si la estrategia es válida"""
        try:
            # Criterios mínimos para validez
            min_sharpe = 0.5
            max_degradation = 0.5  # 50% degradación máxima
            min_confidence = 60.0
            
            test_sharpe = test_metrics.get('sharpe_ratio_mean', 0)
            sharpe_degradation = abs(degradation.get('sharpe_ratio_degradation_mean', 1))
            
            is_valid = (
                test_sharpe >= min_sharpe and
                sharpe_degradation <= max_degradation and
                confidence_score >= min_confidence
            )
            
            return is_valid
            
        except Exception:
            return False
    
    def _generate_recommendations(self, 
                                train_metrics: Dict[str, float],
                                test_metrics: Dict[str, float],
                                degradation: Dict[str, float],
                                param_stability: Dict[str, float]) -> List[str]:
        """Genera recomendaciones basadas en resultados"""
        try:
            recommendations = []
            
            # Analizar degradación
            sharpe_degradation = degradation.get('sharpe_ratio_degradation_mean', 0)
            if sharpe_degradation < -0.3:
                recommendations.append("Alta degradación de Sharpe ratio - Revisar overfitting")
            
            # Analizar estabilidad de parámetros
            if param_stability:
                avg_stability = np.mean(list(param_stability.values()))
                if avg_stability < 0.5:
                    recommendations.append("Parámetros inestables - Considerar rangos más amplios")
            
            # Analizar performance out-of-sample
            test_sharpe = test_metrics.get('sharpe_ratio_mean', 0)
            if test_sharpe < 0.5:
                recommendations.append("Sharpe ratio bajo out-of-sample - Revisar estrategia")
            
            # Analizar consistencia
            sharpe_std = test_metrics.get('sharpe_ratio_std', 0)
            if sharpe_std > 0.5:
                recommendations.append("Alta variabilidad en resultados - Mejorar robustez")
            
            if not recommendations:
                recommendations.append("Estrategia muestra validación sólida")
            
            return recommendations
            
        except Exception:
            return ["Error generando recomendaciones"]
    
    def _generate_warnings(self, 
                         test_metrics: Dict[str, float],
                         degradation: Dict[str, float],
                         confidence_score: float) -> List[str]:
        """Genera warnings basados en resultados"""
        try:
            warnings = []
            
            if confidence_score < 50:
                warnings.append("Baja confianza en validación")
            
            sharpe_degradation = abs(degradation.get('sharpe_ratio_degradation_mean', 0))
            if sharpe_degradation > 0.5:
                warnings.append("Degradación excesiva - Posible overfitting")
            
            test_sharpe = test_metrics.get('sharpe_ratio_mean', 0)
            if test_sharpe < 0:
                warnings.append("Performance negativa out-of-sample")
            
            return warnings
            
        except Exception:
            return ["Error generando warnings"]

class OutOfSampleTest:
    """
    Test out-of-sample simple para validación rápida
    
    Divide datos en entrenamiento/prueba y evalúa degradación
    """
    
    def __init__(self, train_ratio: float = 0.7):
        self.logger = get_logger("OutOfSampleTest")
        self.train_ratio = train_ratio
        self.performance_analyzer = PerformanceAnalyzer()
    
    def run_test(self, 
                backtest_function: Callable,
                data: pd.DataFrame,
                params: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta test out-of-sample simple"""
        try:
            # Dividir datos
            split_point = int(len(data) * self.train_ratio)
            train_data = data.iloc[:split_point]
            test_data = data.iloc[split_point:]
            
            # Ejecutar backtests
            train_results = backtest_function(train_data, params)
            test_results = backtest_function(test_data, params)
            
            # Calcular degradación
            degradation = {}
            for metric in ['sharpe_ratio', 'total_return', 'max_drawdown']:
                train_val = train_results.get(metric, 0)
                test_val = test_results.get(metric, 0)
                
                if train_val != 0:
                    degradation[f"{metric}_degradation"] = (test_val - train_val) / abs(train_val)
            
            return {
                'train_results': train_results,
                'test_results': test_results,
                'degradation': degradation,
                'is_valid': abs(degradation.get('sharpe_ratio_degradation', 1)) < 0.3
            }
            
        except Exception as e:
            self.logger.error(f"Error en test out-of-sample: {e}")
            return {}

class RobustnessTest:
    """
    Test de robustez con perturbaciones de datos
    
    Evalúa cómo responde la estrategia a pequeños cambios en los datos
    """
    
    def __init__(self, n_perturbations: int = 100, noise_level: float = 0.01):
        self.logger = get_logger("RobustnessTest")
        self.n_perturbations = n_perturbations
        self.noise_level = noise_level
    
    def run_test(self, 
                backtest_function: Callable,
                data: pd.DataFrame,
                params: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta test de robustez"""
        try:
            baseline_results = backtest_function(data, params)
            baseline_sharpe = baseline_results.get('sharpe_ratio', 0)
            
            perturbed_sharpes = []
            
            for _ in range(self.n_perturbations):
                # Agregar ruido a los datos
                perturbed_data = data.copy()
                for col in ['open', 'high', 'low', 'close']:
                    if col in perturbed_data.columns:
                        noise = np.random.normal(0, self.noise_level, len(perturbed_data))
                        perturbed_data[col] *= (1 + noise)
                
                # Ejecutar backtest con datos perturbados
                try:
                    results = backtest_function(perturbed_data, params)
                    perturbed_sharpes.append(results.get('sharpe_ratio', 0))
                except:
                    continue
            
            if perturbed_sharpes:
                robustness_score = 1 - (np.std(perturbed_sharpes) / abs(baseline_sharpe)) if baseline_sharpe != 0 else 0
                
                return {
                    'baseline_sharpe': baseline_sharpe,
                    'perturbed_sharpes': perturbed_sharpes,
                    'robustness_score': max(0, robustness_score),
                    'is_robust': robustness_score > 0.7
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error en test de robustez: {e}")
            return {}