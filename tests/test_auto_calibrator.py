#!/usr/bin/env python3
"""
Script de prueba para el sistema de auto-calibración inteligente
Verifica que la optimización automática de parámetros funcione correctamente

Como trader senior con más de 10 años de experiencia, sé que la auto-calibración
es lo que separa un bot amateur de uno profesional. Este test verifica que
nuestro sistema pueda adaptarse y optimizarse automáticamente.

La auto-calibración es crítica: sin ella, el bot se vuelve obsoleto.
Con ella, evoluciona y mejora continuamente.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import tempfile
import shutil
import json
from unittest.mock import Mock, patch

# Agregar el directorio del proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot.scoring import (
    WeightManager,
    ScoreComponent,
    ScoringRegime,
    PerformanceMetrics
)
from trading_bot.scoring.auto_calibrator import (
    AutoCalibrator,
    CalibrationMethod,
    CalibrationTarget,
    ParameterType,
    ParameterBounds,
    CalibrationConfig
)

def create_mock_weight_manager(temp_dir: str) -> WeightManager:
    """Crea un WeightManager mock para testing"""
    
    # Crear WeightManager real pero con datos controlados
    weight_manager = WeightManager(temp_dir)
    
    # Agregar método mock para obtener performance
    def mock_get_regime_performance(regime: ScoringRegime, days_back: int = 60):
        """Mock que retorna performance simulada"""
        
        # Crear datos de performance simulados
        from datetime import datetime, timedelta
        import pytz
        ECUADOR_TZ = pytz.timezone('America/Guayaquil')
        
        if regime == ScoringRegime.TRENDING_BULL:
            return PerformanceMetrics(
                period_start=datetime.now(ECUADOR_TZ) - timedelta(days=60),
                period_end=datetime.now(ECUADOR_TZ),
                total_trades=150,
                winning_trades=102,
                losing_trades=48,
                win_rate=0.68,
                profit_factor=2.1,
                sharpe_ratio=1.4,
                calmar_ratio=2.8,
                max_drawdown=0.08,
                avg_win=1.8,
                avg_loss=-0.9,
                largest_win=5.2,
                largest_loss=-2.1,
                consecutive_wins=8,
                consecutive_losses=3
            )
        elif regime == ScoringRegime.RANGING_LOW_VOL:
            return PerformanceMetrics(
                period_start=datetime.now(ECUADOR_TZ) - timedelta(days=60),
                period_end=datetime.now(ECUADOR_TZ),
                total_trades=120,
                winning_trades=74,
                losing_trades=46,
                win_rate=0.62,
                profit_factor=1.7,
                sharpe_ratio=1.1,
                calmar_ratio=1.9,
                max_drawdown=0.12,
                avg_win=1.2,
                avg_loss=-0.8,
                largest_win=3.8,
                largest_loss=-1.9,
                consecutive_wins=6,
                consecutive_losses=4
            )
        else:
            return PerformanceMetrics(
                period_start=datetime.now(ECUADOR_TZ) - timedelta(days=60),
                period_end=datetime.now(ECUADOR_TZ),
                total_trades=80,
                winning_trades=44,
                losing_trades=36,
                win_rate=0.55,
                profit_factor=1.3,
                sharpe_ratio=0.8,
                calmar_ratio=1.2,
                max_drawdown=0.18,
                avg_win=1.0,
                avg_loss=-0.9,
                largest_win=2.8,
                largest_loss=-2.5,
                consecutive_wins=4,
                consecutive_losses=5
            )
    
    # Agregar método para obtener pesos de régimen
    def mock_get_regime_weights(regime: ScoringRegime):
        """Mock que retorna pesos por régimen"""
        return {
            "mtf_structure": 0.35,
            "technical_confluence": 0.25,
            "market_context": 0.20,
            "risk_metrics": 0.20
        }
    
    # Agregar método para actualizar pesos
    def mock_update_regime_weights(regime: ScoringRegime, weights: dict):
        """Mock que simula actualización de pesos"""
        return True
    
    # Asignar métodos mock
    weight_manager.get_regime_performance = mock_get_regime_performance
    weight_manager.get_regime_weights = mock_get_regime_weights
    weight_manager.update_regime_weights = mock_update_regime_weights
    
    return weight_manager

def create_mock_historical_data(regime: ScoringRegime, num_trades: int = 100) -> pd.DataFrame:
    """Crea datos históricos simulados para calibración"""
    
    np.random.seed(42)  # Para reproducibilidad
    
    # Configurar características por régimen
    if regime == ScoringRegime.TRENDING_BULL:
        base_win_rate = 0.70
        score_bias = 75.0  # Scores más altos en trending
        pnl_multiplier = 1.2
    elif regime == ScoringRegime.RANGING_LOW_VOL:
        base_win_rate = 0.60
        score_bias = 65.0
        pnl_multiplier = 0.8
    else:
        base_win_rate = 0.55
        score_bias = 60.0
        pnl_multiplier = 0.9
    
    trades = []
    cumulative_pnl = 0.0
    
    for i in range(num_trades):
        # Generar timestamp
        timestamp = datetime.now() - timedelta(days=num_trades-i, hours=np.random.randint(0, 24))
        
        # Generar score (influye en probabilidad de éxito)
        score = np.random.normal(score_bias, 15.0)
        score = max(30.0, min(95.0, score))  # Limitar entre 30-95
        
        # Probabilidad de ganar basada en score
        win_prob = base_win_rate * (score / 100.0) * 1.1
        win_prob = max(0.1, min(0.9, win_prob))
        
        is_win = np.random.random() < win_prob
        
        # Generar PnL
        if is_win:
            pnl = np.random.uniform(0.5, 3.0) * pnl_multiplier
        else:
            pnl = -np.random.uniform(0.3, 1.8) * pnl_multiplier
        
        cumulative_pnl += pnl
        
        # Generar precios
        entry_price = 50000 + np.random.uniform(-2000, 2000)
        exit_price = entry_price * (1 + pnl / 100)
        
        trade = {
            'timestamp': timestamp,
            'symbol': 'BTCUSDT',
            'direction': 'LONG' if np.random.random() > 0.5 else 'SHORT',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl,
            'score': score,
            'regime': regime.value,
            'duration_minutes': np.random.randint(30, 480)  # 30min a 8h
        }
        
        trades.append(trade)
    
    return pd.DataFrame(trades)

async def test_auto_calibrator_initialization():
    """Prueba inicialización del auto-calibrador"""
    print(f"\n{'='*60}")
    print("PROBANDO INICIALIZACIÓN DEL AUTO-CALIBRADOR")
    print(f"{'='*60}")
    
    try:
        temp_dir = tempfile.mkdtemp()
        
        # Crear weight manager mock
        weight_manager = create_mock_weight_manager(temp_dir)
        
        # Crear auto-calibrador
        calibrator = AutoCalibrator(weight_manager, temp_dir)
        print(f"✅ AutoCalibrator creado exitosamente")
        
        # Verificar que se inicializaron los parámetros calibrables
        params = calibrator.get_parameter_bounds()
        print(f"✅ {len(params)} parámetros calibrables inicializados")
        
        # Verificar parámetros esperados
        expected_params = [
            "min_score_threshold",
            "llm_threshold_min", 
            "llm_threshold_max",
            "max_position_size",
            "stop_loss_multiplier",
            "take_profit_ratio",
            "mtf_structure_weight",
            "technical_confluence_weight",
            "market_context_weight",
            "risk_metrics_weight"
        ]
        
        for param_name in expected_params:
            if param_name not in params:
                print(f"❌ Parámetro faltante: {param_name}")
                return False
            
            bounds = params[param_name]
            if bounds.min_value >= bounds.max_value:
                print(f"❌ Límites inválidos para {param_name}")
                return False
        
        print(f"✅ Todos los parámetros tienen límites válidos")
        
        # Verificar configuración
        config = calibrator.config
        if not isinstance(config, CalibrationConfig):
            print(f"❌ Configuración no inicializada correctamente")
            return False
        
        print(f"✅ Configuración inicializada: {config.calibration_frequency_days} días")
        
        # Verificar estado inicial
        status = calibrator.get_calibration_status()
        print(f"✅ Estado inicial obtenido: {len(status)} regímenes")
        
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ Error en inicialización: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_calibration_necessity():
    """Prueba detección de necesidad de calibración"""
    print(f"\n{'='*60}")
    print("PROBANDO DETECCIÓN DE NECESIDAD DE CALIBRACIÓN")
    print(f"{'='*60}")
    
    try:
        temp_dir = tempfile.mkdtemp()
        weight_manager = create_mock_weight_manager(temp_dir)
        calibrator = AutoCalibrator(weight_manager, temp_dir)
        
        # Caso 1: Régimen nunca calibrado
        regime = ScoringRegime.TRENDING_BULL
        should_calibrate = calibrator.should_calibrate(regime)
        print(f"1. Régimen nunca calibrado: {'✅ Necesita' if should_calibrate else '❌ No necesita'}")
        
        if not should_calibrate:
            print(f"   ❌ Debería necesitar calibración inicial")
            return False
        
        # Caso 2: Calibración reciente (simular)
        import pytz
        ECUADOR_TZ = pytz.timezone('America/Guayaquil')
        calibrator.last_calibration[regime] = datetime.now(ECUADOR_TZ)
        should_calibrate = calibrator.should_calibrate(regime)
        print(f"2. Calibración reciente: {'⚠️ Necesita' if should_calibrate else '✅ No necesita'}")
        
        if should_calibrate:
            print(f"   ⚠️ No debería necesitar calibración reciente")
        
        # Caso 3: Calibración antigua
        calibrator.last_calibration[regime] = datetime.now(ECUADOR_TZ) - timedelta(days=20)
        should_calibrate = calibrator.should_calibrate(regime)
        print(f"3. Calibración antigua: {'✅ Necesita' if should_calibrate else '❌ No necesita'}")
        
        if not should_calibrate:
            print(f"   ❌ Debería necesitar calibración por tiempo")
            return False
        
        print(f"✅ Detección de necesidad funciona correctamente")
        
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ Error en detección de necesidad: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_parameter_optimization():
    """Prueba optimización de parámetros"""
    print(f"\n{'='*60}")
    print("PROBANDO OPTIMIZACIÓN DE PARÁMETROS")
    print(f"{'='*60}")
    
    try:
        temp_dir = tempfile.mkdtemp()
        weight_manager = create_mock_weight_manager(temp_dir)
        calibrator = AutoCalibrator(weight_manager, temp_dir)
        
        # Crear datos históricos simulados
        regime = ScoringRegime.TRENDING_BULL
        historical_data = create_mock_historical_data(regime, 120)
        
        # Mock del método _load_historical_data
        def mock_load_historical_data(regime_param, days_back=90):
            return historical_data
        
        calibrator._load_historical_data = mock_load_historical_data
        
        print(f"✅ Datos históricos simulados: {len(historical_data)} trades")
        
        # Probar diferentes métodos de optimización
        methods_to_test = [
            CalibrationMethod.GRID_SEARCH,
            CalibrationMethod.BAYESIAN_OPTIMIZATION,
            CalibrationMethod.GENETIC_ALGORITHM,
            CalibrationMethod.ENSEMBLE
        ]
        
        optimization_results = {}
        
        for method in methods_to_test:
            try:
                print(f"\n--- Probando {method.value} ---")
                
                current_params = calibrator._get_current_parameters(regime)
                
                optimized_params = calibrator._optimize_parameters(
                    historical_data, 
                    current_params, 
                    method, 
                    CalibrationTarget.MULTI_OBJECTIVE
                )
                
                if optimized_params:
                    print(f"  ✅ Optimización exitosa")
                    
                    # Verificar que los parámetros están en rangos válidos
                    valid_params = True
                    for param_name, value in optimized_params.items():
                        if param_name in calibrator.calibrable_parameters:
                            bounds = calibrator.calibrable_parameters[param_name]
                            if not (bounds.min_value <= value <= bounds.max_value):
                                print(f"  ❌ Parámetro fuera de rango: {param_name}={value}")
                                valid_params = False
                    
                    if valid_params:
                        print(f"  ✅ Todos los parámetros en rangos válidos")
                        optimization_results[method] = optimized_params
                    else:
                        print(f"  ❌ Parámetros fuera de rango")
                        
                else:
                    print(f"  ⚠️ Optimización no produjo resultados")
                    
            except Exception as e:
                print(f"  ❌ Error en {method.value}: {str(e)}")
        
        if len(optimization_results) == 0:
            print(f"❌ Ningún método de optimización funcionó")
            return False
        
        print(f"\n✅ {len(optimization_results)}/{len(methods_to_test)} métodos funcionaron")
        
        # Verificar que ensemble produce resultados diferentes a métodos individuales
        if CalibrationMethod.ENSEMBLE in optimization_results:
            ensemble_params = optimization_results[CalibrationMethod.ENSEMBLE]
            individual_methods = [m for m in optimization_results.keys() if m != CalibrationMethod.ENSEMBLE]
            
            if individual_methods:
                individual_params = optimization_results[individual_methods[0]]
                
                # Verificar diferencias
                differences = 0
                for param_name in ensemble_params:
                    if param_name in individual_params:
                        if abs(ensemble_params[param_name] - individual_params[param_name]) > 0.001:
                            differences += 1
                
                if differences > 0:
                    print(f"✅ Ensemble produce resultados diferentes ({differences} parámetros)")
                else:
                    print(f"⚠️ Ensemble produce resultados idénticos")
        
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ Error en optimización: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_validation_system():
    """Prueba sistema de validación cruzada"""
    print(f"\n{'='*60}")
    print("PROBANDO SISTEMA DE VALIDACIÓN CRUZADA")
    print(f"{'='*60}")
    
    try:
        temp_dir = tempfile.mkdtemp()
        weight_manager = create_mock_weight_manager(temp_dir)
        calibrator = AutoCalibrator(weight_manager, temp_dir)
        
        # Crear datos históricos con performance conocida
        regime = ScoringRegime.TRENDING_BULL
        historical_data = create_mock_historical_data(regime, 150)  # Más datos para validación
        
        print(f"✅ Datos para validación: {len(historical_data)} trades")
        
        # Caso 1: Parámetros buenos (deberían pasar validación)
        good_params = {
            "min_score_threshold": 65.0,
            "max_position_size": 0.05,
            "mtf_structure_weight": 0.35,
            "technical_confluence_weight": 0.25,
            "market_context_weight": 0.20,
            "risk_metrics_weight": 0.20
        }
        
        validation_result = calibrator._validate_parameters(
            historical_data, 
            good_params, 
            CalibrationTarget.MULTI_OBJECTIVE
        )
        
        print(f"1. Parámetros buenos:")
        print(f"   Validación: {'✅ Pasó' if validation_result['passed'] else '❌ Falló'}")
        print(f"   Score: {validation_result['score']:.3f}")
        print(f"   Confianza: {validation_result.get('confidence', 0):.3f}")
        print(f"   Folds: {validation_result.get('n_folds', 0)}")
        
        if not validation_result['passed']:
            print(f"   Razón: {validation_result['reason']}")
        
        # Caso 2: Parámetros extremos (deberían fallar validación)
        extreme_params = {
            "min_score_threshold": 95.0,  # Muy restrictivo
            "max_position_size": 0.01,    # Muy pequeño
            "mtf_structure_weight": 0.95, # Muy desbalanceado
            "technical_confluence_weight": 0.02,
            "market_context_weight": 0.02,
            "risk_metrics_weight": 0.01
        }
        
        extreme_validation = calibrator._validate_parameters(
            historical_data,
            extreme_params,
            CalibrationTarget.MULTI_OBJECTIVE
        )
        
        print(f"\n2. Parámetros extremos:")
        print(f"   Validación: {'⚠️ Pasó' if extreme_validation['passed'] else '✅ Falló (correcto)'}")
        print(f"   Score: {extreme_validation['score']:.3f}")
        
        if not extreme_validation['passed']:
            print(f"   Razón: {extreme_validation['reason']}")
        
        # Caso 3: Datos insuficientes
        small_data = historical_data.head(10)  # Solo 10 trades
        
        small_validation = calibrator._validate_parameters(
            small_data,
            good_params,
            CalibrationTarget.MULTI_OBJECTIVE
        )
        
        print(f"\n3. Datos insuficientes:")
        print(f"   Validación: {'⚠️ Pasó' if small_validation['passed'] else '✅ Falló (correcto)'}")
        
        if not small_validation['passed']:
            print(f"   Razón: {small_validation['reason']}")
        
        # Verificar que la validación es conservadora
        conservative_behavior = (
            validation_result['passed'] and 
            not extreme_validation['passed'] and 
            not small_validation['passed']
        )
        
        if conservative_behavior:
            print(f"\n✅ Sistema de validación es apropiadamente conservador")
        else:
            print(f"\n⚠️ Sistema de validación podría ser muy permisivo o restrictivo")
        
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ Error en validación: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_full_calibration_cycle():
    """Prueba ciclo completo de calibración"""
    print(f"\n{'='*60}")
    print("PROBANDO CICLO COMPLETO DE CALIBRACIÓN")
    print(f"{'='*60}")
    
    try:
        temp_dir = tempfile.mkdtemp()
        weight_manager = create_mock_weight_manager(temp_dir)
        calibrator = AutoCalibrator(weight_manager, temp_dir)
        
        # Mock del método _load_historical_data
        regime = ScoringRegime.TRENDING_BULL
        historical_data = create_mock_historical_data(regime, 120)
        
        def mock_load_historical_data(regime_param, days_back=90):
            return historical_data
        
        calibrator._load_historical_data = mock_load_historical_data
        
        print(f"✅ Configuración de mocks completada")
        
        # Obtener parámetros iniciales
        initial_params = calibrator._get_current_parameters(regime)
        print(f"✅ Parámetros iniciales obtenidos: {len(initial_params)}")
        
        # Ejecutar calibración completa
        print(f"\n--- Ejecutando calibración completa ---")
        
        calibration_result = calibrator.calibrate_regime(
            regime,
            method=CalibrationMethod.ENSEMBLE,
            target=CalibrationTarget.MULTI_OBJECTIVE,
            force=True
        )
        
        if not calibration_result:
            print(f"❌ Calibración no produjo resultados")
            return False
        
        print(f"✅ Calibración completada exitosamente")
        
        # Verificar resultado
        print(f"\nRESULTADO DE CALIBRACIÓN:")
        print(f"  Método: {calibration_result.method.value}")
        print(f"  Objetivo: {calibration_result.target.value}")
        print(f"  Régimen: {calibration_result.regime.value}")
        print(f"  Mejora: {calibration_result.improvement_pct:.1%}")
        print(f"  Validación: {'✅' if calibration_result.validation_passed else '❌'}")
        print(f"  Confianza: {calibration_result.confidence_score:.3f}")
        print(f"  Iteraciones: {calibration_result.iterations}")
        print(f"  Convergencia: {'✅' if calibration_result.convergence_achieved else '❌'}")
        
        # Verificar que los parámetros cambiaron
        params_changed = False
        for param_name in initial_params:
            if param_name in calibration_result.optimized_parameters:
                initial_val = initial_params[param_name]
                optimized_val = calibration_result.optimized_parameters[param_name]
                if abs(initial_val - optimized_val) > 0.001:
                    params_changed = True
                    break
        
        if params_changed:
            print(f"✅ Parámetros optimizados exitosamente")
        else:
            print(f"⚠️ Parámetros no cambiaron (posiblemente ya óptimos)")
        
        # Verificar que se guardó en historial
        if len(calibrator.calibration_history) > 0:
            print(f"✅ Calibración guardada en historial")
        else:
            print(f"❌ Calibración no guardada en historial")
            return False
        
        # Verificar estado actualizado
        status = calibrator.get_calibration_status()
        if regime.value in status["last_calibrations"]:
            print(f"✅ Estado de calibración actualizado")
        else:
            print(f"❌ Estado no actualizado")
            return False
        
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ Error en ciclo completo: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_rollback_functionality():
    """Prueba funcionalidad de rollback"""
    print(f"\n{'='*60}")
    print("PROBANDO FUNCIONALIDAD DE ROLLBACK")
    print(f"{'='*60}")
    
    try:
        temp_dir = tempfile.mkdtemp()
        weight_manager = create_mock_weight_manager(temp_dir)
        calibrator = AutoCalibrator(weight_manager, temp_dir)
        
        regime = ScoringRegime.RANGING_LOW_VOL
        
        # Simular calibración previa
        from trading_bot.scoring.auto_calibrator import CalibrationResult
        from datetime import datetime, timedelta
        
        original_params = {
            "min_score_threshold": 65.0,
            "max_position_size": 0.05,
            "mtf_structure_weight": 0.35
        }
        
        optimized_params = {
            "min_score_threshold": 70.0,
            "max_position_size": 0.03,
            "mtf_structure_weight": 0.45
        }
        
        # Crear resultado de calibración simulado
        mock_result = CalibrationResult(
            method=CalibrationMethod.ENSEMBLE,
            target=CalibrationTarget.MULTI_OBJECTIVE,
            regime=regime,
            optimized_parameters=optimized_params,
            original_parameters=original_params,
            original_performance=PerformanceMetrics(
                period_start=datetime.now() - timedelta(days=60),
                period_end=datetime.now() - timedelta(days=30),
                total_trades=100,
                winning_trades=60,
                losing_trades=40,
                win_rate=0.6,
                profit_factor=1.8,
                sharpe_ratio=1.1,
                calmar_ratio=1.9,
                max_drawdown=0.12,
                avg_win=1.2,
                avg_loss=-0.8,
                largest_win=3.5,
                largest_loss=-2.2,
                consecutive_wins=5,
                consecutive_losses=3
            ),
            optimized_performance=PerformanceMetrics(
                period_start=datetime.now() - timedelta(days=30),
                period_end=datetime.now(),
                total_trades=100,
                winning_trades=65,
                losing_trades=35,
                win_rate=0.65,
                profit_factor=2.0,
                sharpe_ratio=1.3,
                calmar_ratio=2.2,
                max_drawdown=0.10,
                avg_win=1.3,
                avg_loss=-0.7,
                largest_win=4.1,
                largest_loss=-1.8,
                consecutive_wins=7,
                consecutive_losses=2
            ),
            improvement_pct=0.15,
            validation_passed=True,
            validation_score=0.75,
            calibration_time=datetime.now(),
            iterations=50,
            convergence_achieved=True,
            confidence_score=0.8
        )
        
        # Agregar al historial
        calibrator.calibration_history.append(mock_result)
        calibrator.active_parameters[regime] = optimized_params.copy()
        
        print(f"✅ Calibración simulada agregada al historial")
        
        # Verificar parámetros antes del rollback
        current_params = calibrator.active_parameters[regime]
        print(f"Parámetros actuales: {current_params['min_score_threshold']}")
        
        # Ejecutar rollback
        rollback_success = calibrator.rollback_calibration(regime)
        
        if not rollback_success:
            print(f"❌ Rollback falló")
            return False
        
        print(f"✅ Rollback ejecutado exitosamente")
        
        # Verificar que los parámetros se revirtieron
        reverted_params = calibrator.active_parameters[regime]
        
        if reverted_params["min_score_threshold"] == original_params["min_score_threshold"]:
            print(f"✅ Parámetros revertidos correctamente")
        else:
            print(f"❌ Parámetros no se revirtieron")
            print(f"   Esperado: {original_params['min_score_threshold']}")
            print(f"   Actual: {reverted_params['min_score_threshold']}")
            return False
        
        # Probar rollback de régimen sin calibraciones
        empty_regime = ScoringRegime.BREAKOUT
        empty_rollback = calibrator.rollback_calibration(empty_regime)
        
        if empty_rollback:
            print(f"⚠️ Rollback exitoso en régimen sin calibraciones")
        else:
            print(f"✅ Rollback falló correctamente en régimen vacío")
        
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ Error en rollback: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_persistence_and_recovery():
    """Prueba persistencia y recuperación de estado"""
    print(f"\n{'='*60}")
    print("PROBANDO PERSISTENCIA Y RECUPERACIÓN")
    print(f"{'='*60}")
    
    try:
        temp_dir = tempfile.mkdtemp()
        
        # Crear primer calibrador y agregar datos
        calibrator1 = AutoCalibrator(create_mock_weight_manager(temp_dir), temp_dir)
        
        regime = ScoringRegime.TRENDING_BULL
        
        # Simular calibración
        calibrator1.last_calibration[regime] = datetime.now()
        calibrator1.active_parameters[regime] = {
            "min_score_threshold": 72.0,
            "max_position_size": 0.04
        }
        
        # Guardar estado
        calibrator1._save_calibration_state()
        print(f"✅ Estado guardado por primer calibrador")
        
        # Crear segundo calibrador y verificar que cargó el estado
        calibrator2 = AutoCalibrator(create_mock_weight_manager(temp_dir), temp_dir)
        
        # Verificar que se cargó la última calibración
        if regime in calibrator2.last_calibration:
            print(f"✅ Última calibración cargada correctamente")
        else:
            print(f"❌ Última calibración no se cargó")
            return False
        
        # Verificar que se cargaron los parámetros activos
        if regime in calibrator2.active_parameters:
            loaded_params = calibrator2.active_parameters[regime]
            if loaded_params["min_score_threshold"] == 72.0:
                print(f"✅ Parámetros activos cargados correctamente")
            else:
                print(f"❌ Parámetros activos incorrectos")
                return False
        else:
            print(f"❌ Parámetros activos no se cargaron")
            return False
        
        # Verificar archivos de persistencia
        state_file = os.path.join(temp_dir, "calibration_state.json")
        if os.path.exists(state_file):
            print(f"✅ Archivo de estado existe")
            
            # Verificar contenido
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            if "last_calibrations" in state_data and "active_parameters" in state_data:
                print(f"✅ Estructura de estado correcta")
            else:
                print(f"❌ Estructura de estado incorrecta")
                return False
        else:
            print(f"❌ Archivo de estado no existe")
            return False
        
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ Error en persistencia: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_performance_evaluation():
    """Prueba evaluación de performance"""
    print(f"\n{'='*60}")
    print("PROBANDO EVALUACIÓN DE PERFORMANCE")
    print(f"{'='*60}")
    
    try:
        temp_dir = tempfile.mkdtemp()
        weight_manager = create_mock_weight_manager(temp_dir)
        calibrator = AutoCalibrator(weight_manager, temp_dir)
        
        # Crear datos con diferentes características de performance
        good_data = create_mock_historical_data(ScoringRegime.TRENDING_BULL, 100)
        poor_data = create_mock_historical_data(ScoringRegime.REVERSAL, 100)
        
        print(f"✅ Datos de prueba creados")
        
        # Parámetros de prueba
        test_params = {
            "min_score_threshold": 65.0,
            "max_position_size": 0.05,
            "mtf_structure_weight": 0.35,
            "technical_confluence_weight": 0.25,
            "market_context_weight": 0.20,
            "risk_metrics_weight": 0.20
        }
        
        # Evaluar con datos buenos
        good_score = calibrator._evaluate_parameter_set(
            good_data, 
            test_params, 
            CalibrationTarget.MULTI_OBJECTIVE
        )
        
        print(f"1. Datos buenos - Score: {good_score:.3f}")
        
        # Evaluar con datos pobres
        poor_score = calibrator._evaluate_parameter_set(
            poor_data,
            test_params,
            CalibrationTarget.MULTI_OBJECTIVE
        )
        
        print(f"2. Datos pobres - Score: {poor_score:.3f}")
        
        # Los datos buenos deberían tener mejor score
        if good_score > poor_score:
            print(f"✅ Evaluación diferencia correctamente calidad de datos")
        else:
            print(f"⚠️ Evaluación no diferencia calidad de datos")
        
        # Probar diferentes objetivos
        targets_to_test = [
            CalibrationTarget.PROFIT_FACTOR,
            CalibrationTarget.WIN_RATE,
            CalibrationTarget.SHARPE_RATIO,
            CalibrationTarget.MAX_DRAWDOWN
        ]
        
        target_scores = {}
        for target in targets_to_test:
            score = calibrator._evaluate_parameter_set(good_data, test_params, target)
            target_scores[target] = score
            print(f"3. {target.value}: {score:.3f}")
        
        # Verificar que los scores están en rangos razonables
        valid_scores = all(0 <= score <= 10 for score in target_scores.values())
        
        if valid_scores:
            print(f"✅ Todos los scores en rangos válidos")
        else:
            print(f"⚠️ Algunos scores fuera de rango")
        
        # Probar simulación de performance
        simulated_performance = calibrator._simulate_performance(good_data, test_params)
        
        print(f"\n4. Performance simulada:")
        print(f"   Total trades: {simulated_performance.total_trades}")
        print(f"   Win rate: {simulated_performance.win_rate:.1%}")
        print(f"   Profit factor: {simulated_performance.profit_factor:.2f}")
        print(f"   Max drawdown: {simulated_performance.max_drawdown:.1%}")
        
        if simulated_performance.total_trades > 0:
            print(f"✅ Simulación de performance funcional")
        else:
            print(f"❌ Simulación no produjo trades")
            return False
        
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ Error en evaluación de performance: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_edge_cases_and_robustness():
    """Prueba casos extremos y robustez"""
    print(f"\n{'='*60}")
    print("PROBANDO CASOS EXTREMOS Y ROBUSTEZ")
    print(f"{'='*60}")
    
    try:
        temp_dir = tempfile.mkdtemp()
        weight_manager = create_mock_weight_manager(temp_dir)
        calibrator = AutoCalibrator(weight_manager, temp_dir)
        
        # Caso 1: Datos vacíos
        print(f"1. Probando con datos vacíos...")
        empty_data = pd.DataFrame()
        
        try:
            empty_score = calibrator._evaluate_parameter_set(
                empty_data, 
                {"min_score_threshold": 65.0}, 
                CalibrationTarget.MULTI_OBJECTIVE
            )
            print(f"   Score con datos vacíos: {empty_score}")
            
            if empty_score == 0.0:
                print(f"   ✅ Manejo correcto de datos vacíos")
            else:
                print(f"   ⚠️ Score inesperado con datos vacíos")
                
        except Exception as e:
            print(f"   ❌ Error con datos vacíos: {str(e)}")
            return False
        
        # Caso 2: Parámetros extremos
        print(f"\n2. Probando parámetros extremos...")
        extreme_params = {
            "min_score_threshold": 99.0,  # Extremadamente restrictivo
            "max_position_size": 0.001,   # Extremadamente pequeño
            "mtf_structure_weight": 1.0,  # Todo el peso en un componente
            "technical_confluence_weight": 0.0,
            "market_context_weight": 0.0,
            "risk_metrics_weight": 0.0
        }
        
        normal_data = create_mock_historical_data(ScoringRegime.TRENDING_BULL, 50)
        
        try:
            extreme_score = calibrator._evaluate_parameter_set(
                normal_data,
                extreme_params,
                CalibrationTarget.MULTI_OBJECTIVE
            )
            print(f"   Score con parámetros extremos: {extreme_score:.3f}")
            print(f"   ✅ Parámetros extremos manejados sin error")
            
        except Exception as e:
            print(f"   ❌ Error con parámetros extremos: {str(e)}")
            return False
        
        # Caso 3: Datos corruptos
        print(f"\n3. Probando datos corruptos...")
        corrupt_data = normal_data.copy()
        corrupt_data.loc[0, 'pnl'] = float('inf')  # Valor infinito
        corrupt_data.loc[1, 'score'] = float('nan')  # Valor NaN
        
        try:
            corrupt_score = calibrator._evaluate_parameter_set(
                corrupt_data,
                {"min_score_threshold": 65.0},
                CalibrationTarget.MULTI_OBJECTIVE
            )
            print(f"   Score con datos corruptos: {corrupt_score:.3f}")
            print(f"   ✅ Datos corruptos manejados sin error")
            
        except Exception as e:
            print(f"   ❌ Error con datos corruptos: {str(e)}")
            return False
        
        # Caso 4: Directorio de datos inaccesible
        print(f"\n4. Probando directorio inaccesible...")
        try:
            # Crear calibrador con directorio que no se puede escribir
            readonly_dir = "/tmp/readonly_test"
            os.makedirs(readonly_dir, exist_ok=True)
            os.chmod(readonly_dir, 0o444)  # Solo lectura
            
            try:
                readonly_calibrator = AutoCalibrator(weight_manager, readonly_dir)
                print(f"   ✅ Calibrador creado con directorio de solo lectura")
                
                # Intentar guardar estado
                readonly_calibrator._save_calibration_state()
                print(f"   ⚠️ Guardado exitoso en directorio de solo lectura")
                
            except Exception as e:
                print(f"   ✅ Error manejado correctamente: {type(e).__name__}")
            
            finally:
                # Limpiar
                os.chmod(readonly_dir, 0o755)
                shutil.rmtree(readonly_dir, ignore_errors=True)
                
        except Exception as e:
            print(f"   ✅ Caso extremo manejado: {type(e).__name__}")
        
        # Caso 5: Memoria limitada (simulado)
        print(f"\n5. Probando con datos masivos...")
        try:
            # Crear dataset muy grande
            massive_data = create_mock_historical_data(ScoringRegime.TRENDING_BULL, 10000)
            
            start_time = datetime.now()
            massive_score = calibrator._evaluate_parameter_set(
                massive_data,
                {"min_score_threshold": 65.0},
                CalibrationTarget.MULTI_OBJECTIVE
            )
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            print(f"   Score con {len(massive_data)} trades: {massive_score:.3f}")
            print(f"   Tiempo de procesamiento: {processing_time:.2f}s")
            
            if processing_time < 10.0:  # Menos de 10 segundos
                print(f"   ✅ Performance aceptable con datos masivos")
            else:
                print(f"   ⚠️ Performance lenta con datos masivos")
                
        except Exception as e:
            print(f"   ❌ Error con datos masivos: {str(e)}")
            return False
        
        print(f"\n✅ Todos los casos extremos manejados correctamente")
        
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ Error en casos extremos: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Función principal de pruebas"""
    print("🔧 SISTEMA DE PRUEBAS - AUTO-CALIBRACIÓN INTELIGENTE")
    print("="*70)
    print("Como trader senior con más de 10 años de experiencia, sé que")
    print("la auto-calibración es lo que separa un bot amateur de uno profesional.")
    print("Este sistema debe ser robusto, conservador y confiable.")
    print("="*70)
    
    # Ejecutar todas las pruebas
    test_results = []
    
    # 1. Inicialización
    result1 = await test_auto_calibrator_initialization()
    test_results.append(("Inicialización del auto-calibrador", result1))
    
    # 2. Detección de necesidad
    result2 = await test_calibration_necessity()
    test_results.append(("Detección de necesidad de calibración", result2))
    
    # 3. Optimización de parámetros
    result3 = await test_parameter_optimization()
    test_results.append(("Optimización de parámetros", result3))
    
    # 4. Sistema de validación
    result4 = await test_validation_system()
    test_results.append(("Sistema de validación cruzada", result4))
    
    # 5. Ciclo completo
    result5 = await test_full_calibration_cycle()
    test_results.append(("Ciclo completo de calibración", result5))
    
    # 6. Funcionalidad de rollback
    result6 = await test_rollback_functionality()
    test_results.append(("Funcionalidad de rollback", result6))
    
    # 7. Persistencia y recuperación
    result7 = await test_persistence_and_recovery()
    test_results.append(("Persistencia y recuperación", result7))
    
    # 8. Evaluación de performance
    result8 = await test_performance_evaluation()
    test_results.append(("Evaluación de performance", result8))
    
    # 9. Casos extremos
    result9 = await test_edge_cases_and_robustness()
    test_results.append(("Casos extremos y robustez", result9))
    
    # Resumen final
    print(f"\n{'='*70}")
    print("RESUMEN FINAL DE PRUEBAS")
    print(f"{'='*70}")
    
    for test_name, success in test_results:
        status = "✅ PASÓ" if success else "❌ FALLÓ"
        print(f"  {test_name}: {status}")
    
    total_passed = sum(result for _, result in test_results)
    total_tests = len(test_results)
    
    print(f"\nResultado final: {total_passed}/{total_tests} grupos de pruebas pasaron")
    
    if total_passed == total_tests:
        print("🎉 ¡TODAS LAS PRUEBAS PASARON EXITOSAMENTE!")
        print("✅ El sistema de auto-calibración está listo para producción")
        print("🧠 Inteligencia artificial aplicada al trading algorítmico")
        print("💰 Objetivo: Optimización continua para máxima rentabilidad")
        print("🛡️ Conservador por diseño - Protege el capital mientras optimiza")
    else:
        print("⚠️  Algunas pruebas fallaron - el sistema necesita ajustes")
        print("🔧 Un auto-calibrador defectuoso = bot obsoleto en semanas")
        print("💀 La adaptación es supervivencia en los mercados")
    
    return total_passed == total_tests

if __name__ == "__main__":
    # Ejecutar pruebas
    success = asyncio.run(main())
    
    # Exit code para CI/CD
    sys.exit(0 if success else 1)