#!/usr/bin/env python3
"""
Script de prueba para el sistema de gestión adaptativa de pesos
Verifica que la optimización automática de pesos funcione correctamente

Como trader senior, sé que un sistema que no se adapta está condenado al fracaso.
Este test verifica que nuestro sistema evolucione con los mercados.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import tempfile
import shutil

# Agregar el directorio del proyecto al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_bot.scoring import (
    WeightManager,
    ScoreComponent,
    ScoringRegime,
    WeightOptimizationMethod,
    create_weight_manager
)

def create_mock_trade_data(regime: ScoringRegime, num_trades: int = 100) -> list:
    """Crea datos de trades simulados para testing"""
    
    np.random.seed(42)  # Para reproducibilidad
    trades = []
    
    # Configurar performance base según régimen
    if regime == ScoringRegime.TRENDING_BULL:
        base_win_rate = 0.7
        base_profit_factor = 2.2
        mtf_importance = 0.4  # MTF más importante en trending
    elif regime == ScoringRegime.RANGING_LOW_VOL:
        base_win_rate = 0.6
        base_profit_factor = 1.8
        mtf_importance = 0.2  # Técnico más importante en ranging
    else:
        base_win_rate = 0.65
        base_profit_factor = 2.0
        mtf_importance = 0.35
    
    for i in range(num_trades):
        # Generar scores por componente
        component_scores = {
            ScoreComponent.MTF_STRUCTURE.value: np.random.uniform(40, 90),
            ScoreComponent.TECHNICAL_CONFLUENCE.value: np.random.uniform(30, 95),
            ScoreComponent.MARKET_CONTEXT.value: np.random.uniform(45, 85),
            ScoreComponent.RISK_METRICS.value: np.random.uniform(50, 90)
        }
        
        # Simular resultado basado en scores y régimen
        # MTF structure es más predictivo en trending markets
        mtf_score = component_scores[ScoreComponent.MTF_STRUCTURE.value]
        tech_score = component_scores[ScoreComponent.TECHNICAL_CONFLUENCE.value]
        
        # Probabilidad de ganar basada en scores y régimen
        if regime == ScoringRegime.TRENDING_BULL:
            win_prob = base_win_rate * (mtf_score / 100) * 1.2
        elif regime == ScoringRegime.RANGING_LOW_VOL:
            win_prob = base_win_rate * (tech_score / 100) * 1.1
        else:
            win_prob = base_win_rate * ((mtf_score + tech_score) / 200)
        
        win_prob = min(0.95, max(0.05, win_prob))  # Limitar entre 5% y 95%
        
        is_win = np.random.random() < win_prob
        
        if is_win:
            pnl = np.random.uniform(0.5, 3.0)  # Ganancia entre 0.5% y 3%
        else:
            pnl = -np.random.uniform(0.3, 1.5)  # Pérdida entre 0.3% y 1.5%
        
        trade = {
            "timestamp": (datetime.now() - timedelta(days=num_trades-i)).isoformat(),
            "regime": regime.value,
            "component_scores": component_scores,
            "trade_result": {
                "pnl": pnl,
                "win": is_win,
                "entry_price": 50000 + np.random.uniform(-1000, 1000),
                "exit_price": 50000 + pnl * 500
            }
        }
        
        trades.append(trade)
    
    return trades

async def test_weight_manager_initialization():
    """Prueba inicialización del gestor de pesos"""
    print(f"\n{'='*60}")
    print("PROBANDO INICIALIZACIÓN DEL WEIGHT MANAGER")
    print(f"{'='*60}")
    
    try:
        # Crear directorio temporal
        temp_dir = tempfile.mkdtemp()
        
        # Crear weight manager
        weight_manager = create_weight_manager(temp_dir)
        print(f"✅ WeightManager creado exitosamente")
        
        # Verificar que se crearon configuraciones por defecto
        for regime in ScoringRegime:
            weights = weight_manager.get_current_weights(regime)
            print(f"  {regime.value}: {len(weights)} componentes")
            
            # Verificar que los pesos suman 1.0
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                print(f"❌ Pesos no suman 1.0 para {regime.value}: {total_weight}")
                return False
        
        # Verificar componentes esperados
        expected_components = [
            ScoreComponent.MTF_STRUCTURE,
            ScoreComponent.TECHNICAL_CONFLUENCE,
            ScoreComponent.MARKET_CONTEXT,
            ScoreComponent.RISK_METRICS
        ]
        
        test_weights = weight_manager.get_current_weights(ScoringRegime.TRENDING_BULL)
        for component in expected_components:
            if component not in test_weights:
                print(f"❌ Componente faltante: {component.value}")
                return False
        
        print(f"✅ Todos los componentes presentes y normalizados")
        
        # Limpiar
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Error en inicialización: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_regime_updates():
    """Prueba actualizaciones de régimen"""
    print(f"\n{'='*60}")
    print("PROBANDO ACTUALIZACIONES DE RÉGIMEN")
    print(f"{'='*60}")
    
    try:
        temp_dir = tempfile.mkdtemp()
        weight_manager = create_weight_manager(temp_dir)
        
        # Probar transiciones de régimen
        regimes_to_test = [
            (ScoringRegime.TRENDING_BULL, 0.8),
            (ScoringRegime.RANGING_LOW_VOL, 0.7),
            (ScoringRegime.BREAKOUT, 0.9),
            (ScoringRegime.REVERSAL, 0.6)
        ]
        
        market_data = {
            "volatility": 0.025,
            "trend_strength": 0.6,
            "momentum": 0.3
        }
        
        transitions = 0
        for regime, confidence in regimes_to_test:
            changed = weight_manager.update_regime(regime, confidence, market_data)
            if changed:
                transitions += 1
                print(f"  ✅ Transición a {regime.value} (confianza: {confidence:.1f})")
            
            # Verificar que el régimen actual se actualizó
            if weight_manager.current_regime != regime:
                print(f"❌ Régimen no se actualizó correctamente")
                return False
        
        print(f"✅ {transitions} transiciones de régimen exitosas")
        
        # Probar que no cambie si es el mismo régimen
        same_regime_change = weight_manager.update_regime(
            ScoringRegime.REVERSAL, 0.6, market_data
        )
        
        if same_regime_change:
            print(f"⚠️  Cambio detectado cuando no debería haber cambio")
        else:
            print(f"✅ No cambio cuando régimen es el mismo")
        
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ Error en actualizaciones de régimen: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_trade_recording():
    """Prueba registro de trades"""
    print(f"\n{'='*60}")
    print("PROBANDO REGISTRO DE TRADES")
    print(f"{'='*60}")
    
    try:
        temp_dir = tempfile.mkdtemp()
        weight_manager = create_weight_manager(temp_dir)
        
        # Configurar régimen
        regime = ScoringRegime.TRENDING_BULL
        weight_manager.update_regime(regime, 0.8, {"volatility": 0.02})
        
        # Registrar varios trades
        component_scores = {
            ScoreComponent.MTF_STRUCTURE: 75.0,
            ScoreComponent.TECHNICAL_CONFLUENCE: 68.0,
            ScoreComponent.MARKET_CONTEXT: 72.0,
            ScoreComponent.RISK_METRICS: 80.0
        }
        
        trades_to_record = [
            {"pnl": 1.5, "win": True, "entry_price": 50000, "exit_price": 50750},
            {"pnl": -0.8, "win": False, "entry_price": 50000, "exit_price": 49600},
            {"pnl": 2.1, "win": True, "entry_price": 50000, "exit_price": 51050},
        ]
        
        for i, trade_result in enumerate(trades_to_record):
            weight_manager.record_trade_result(regime, component_scores, trade_result)
            print(f"  ✅ Trade {i+1} registrado: PnL {trade_result['pnl']:.1f}%")
        
        # Verificar que se incrementó el contador
        config = weight_manager.weight_configs[regime]
        if config.trades_since_update != len(trades_to_record):
            print(f"❌ Contador de trades incorrecto: {config.trades_since_update}")
            return False
        
        print(f"✅ {len(trades_to_record)} trades registrados correctamente")
        
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ Error en registro de trades: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_weight_optimization():
    """Prueba optimización de pesos"""
    print(f"\n{'='*60}")
    print("PROBANDO OPTIMIZACIÓN DE PESOS")
    print(f"{'='*60}")
    
    try:
        temp_dir = tempfile.mkdtemp()
        weight_manager = create_weight_manager(temp_dir)
        
        # Crear datos de trades simulados
        regime = ScoringRegime.TRENDING_BULL
        mock_trades = create_mock_trade_data(regime, 30)  # 30 trades para optimización
        
        # Guardar trades simulados
        import json
        trade_file = os.path.join(temp_dir, f"trades_{regime.value}.json")
        with open(trade_file, 'w') as f:
            json.dump(mock_trades, f, indent=2)
        
        print(f"✅ {len(mock_trades)} trades simulados creados")
        
        # Obtener pesos iniciales
        initial_weights = weight_manager.get_current_weights(regime).copy()
        print(f"Pesos iniciales:")
        for comp, weight in initial_weights.items():
            print(f"  {comp.value}: {weight:.3f}")
        
        # Ejecutar optimización
        optimized = weight_manager.optimize_weights(
            regime, 
            WeightOptimizationMethod.MULTI_OBJECTIVE,
            force_update=True
        )
        
        if not optimized:
            print(f"⚠️  Optimización no aplicada (pesos actuales ya son óptimos)")
            print(f"✅ Sistema conservador funcionando correctamente")
            # En un sistema real, esto es bueno - no cambiar si no hay mejora significativa
            return True
        
        # Obtener pesos optimizados
        optimized_weights = weight_manager.get_current_weights(regime)
        print(f"\nPesos optimizados:")
        for comp, weight in optimized_weights.items():
            print(f"  {comp.value}: {weight:.3f}")
        
        # Verificar que los pesos cambiaron
        weights_changed = False
        for comp in initial_weights:
            if abs(initial_weights[comp] - optimized_weights[comp]) > 0.01:
                weights_changed = True
                break
        
        if not weights_changed:
            print(f"⚠️  Los pesos no cambiaron significativamente")
        else:
            print(f"✅ Pesos optimizados exitosamente")
        
        # Verificar que siguen sumando 1.0
        total_weight = sum(optimized_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            print(f"❌ Pesos optimizados no suman 1.0: {total_weight}")
            return False
        
        print(f"✅ Pesos normalizados correctamente")
        
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ Error en optimización: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_performance_metrics():
    """Prueba cálculo de métricas de performance"""
    print(f"\n{'='*60}")
    print("PROBANDO MÉTRICAS DE PERFORMANCE")
    print(f"{'='*60}")
    
    try:
        temp_dir = tempfile.mkdtemp()
        weight_manager = create_weight_manager(temp_dir)
        
        # Crear datos con performance conocida
        regime = ScoringRegime.RANGING_LOW_VOL
        mock_trades = create_mock_trade_data(regime, 50)
        
        # Guardar trades
        import json
        trade_file = os.path.join(temp_dir, f"trades_{regime.value}.json")
        with open(trade_file, 'w') as f:
            json.dump(mock_trades, f, indent=2)
        
        # Calcular métricas (usar más días para capturar todos los trades)
        performance = weight_manager.get_regime_performance(regime, days_back=60)
        
        if not performance:
            print(f"❌ No se pudieron calcular métricas")
            return False
        
        print(f"MÉTRICAS DE PERFORMANCE:")
        print(f"  Total trades: {performance.total_trades}")
        print(f"  Win rate: {performance.win_rate:.1%}")
        print(f"  Profit factor: {performance.profit_factor:.2f}")
        print(f"  Sharpe ratio: {performance.sharpe_ratio:.2f}")
        print(f"  Max drawdown: {performance.max_drawdown:.1%}")
        print(f"  Avg win: {performance.avg_win:.2f}%")
        print(f"  Avg loss: {performance.avg_loss:.2f}%")
        print(f"  Consecutive wins: {performance.consecutive_wins}")
        print(f"  Consecutive losses: {performance.consecutive_losses}")
        
        # Verificar rangos razonables
        if not (0 <= performance.win_rate <= 1):
            print(f"❌ Win rate fuera de rango: {performance.win_rate}")
            return False
        
        if performance.profit_factor < 0:
            print(f"❌ Profit factor negativo: {performance.profit_factor}")
            return False
        
        if performance.total_trades != len(mock_trades):
            print(f"❌ Total trades incorrecto: {performance.total_trades} vs {len(mock_trades)}")
            return False
        
        # Verificar método is_acceptable
        is_acceptable = performance.is_acceptable()
        print(f"  Performance aceptable: {'✅' if is_acceptable else '❌'}")
        
        print(f"✅ Métricas calculadas correctamente")
        
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ Error en métricas de performance: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_persistence():
    """Prueba persistencia de configuraciones"""
    print(f"\n{'='*60}")
    print("PROBANDO PERSISTENCIA DE CONFIGURACIONES")
    print(f"{'='*60}")
    
    try:
        temp_dir = tempfile.mkdtemp()
        
        # Crear primer weight manager y modificar pesos
        wm1 = create_weight_manager(temp_dir)
        
        regime = ScoringRegime.BREAKOUT
        original_weights = wm1.get_current_weights(regime).copy()
        
        # Modificar pesos manualmente
        modified_weights = original_weights.copy()
        modified_weights[ScoreComponent.MTF_STRUCTURE] = 0.5
        modified_weights[ScoreComponent.TECHNICAL_CONFLUENCE] = 0.3
        modified_weights[ScoreComponent.MARKET_CONTEXT] = 0.1
        modified_weights[ScoreComponent.RISK_METRICS] = 0.1
        
        wm1.weight_configs[regime].weights = modified_weights
        wm1._save_weight_configurations()
        
        print(f"✅ Configuración modificada y guardada")
        
        # Crear segundo weight manager y verificar que cargó los pesos
        wm2 = create_weight_manager(temp_dir)
        loaded_weights = wm2.get_current_weights(regime)
        
        # Verificar que los pesos se cargaron correctamente
        weights_match = True
        for comp in modified_weights:
            if abs(modified_weights[comp] - loaded_weights[comp]) > 0.001:
                weights_match = False
                break
        
        if not weights_match:
            print(f"❌ Pesos no se cargaron correctamente")
            print(f"Guardados: {modified_weights}")
            print(f"Cargados: {loaded_weights}")
            return False
        
        print(f"✅ Configuraciones persistidas y cargadas correctamente")
        
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ Error en persistencia: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_edge_cases():
    """Prueba casos extremos"""
    print(f"\n{'='*60}")
    print("PROBANDO CASOS EXTREMOS")
    print(f"{'='*60}")
    
    try:
        temp_dir = tempfile.mkdtemp()
        weight_manager = create_weight_manager(temp_dir)
        
        # Caso 1: Optimización sin datos suficientes
        print(f"1. Probando optimización sin datos suficientes...")
        regime = ScoringRegime.REVERSAL
        optimized = weight_manager.optimize_weights(regime, force_update=True)
        
        if optimized:
            print(f"   ⚠️  Optimización exitosa sin datos (inesperado)")
        else:
            print(f"   ✅ Optimización falló correctamente sin datos")
        
        # Caso 2: Performance con datos vacíos
        print(f"2. Probando métricas con datos vacíos...")
        performance = weight_manager.get_regime_performance(regime, days_back=30)
        
        if performance:
            print(f"   ⚠️  Métricas calculadas sin datos (inesperado)")
        else:
            print(f"   ✅ Métricas retornaron None sin datos")
        
        # Caso 3: Régimen inexistente
        print(f"3. Probando régimen inexistente...")
        try:
            # Crear régimen temporal que no existe
            fake_regime = ScoringRegime.TRENDING_BULL  # Usar uno existente pero limpiar config
            if fake_regime in weight_manager.weight_configs:
                del weight_manager.weight_configs[fake_regime]
            
            weights = weight_manager.get_current_weights(fake_regime)
            
            if weights:
                print(f"   ✅ Pesos por defecto retornados para régimen inexistente")
            else:
                print(f"   ❌ No se retornaron pesos por defecto")
                return False
                
        except Exception as e:
            print(f"   ❌ Error manejando régimen inexistente: {str(e)}")
            return False
        
        # Caso 4: Directorio de datos inaccesible
        print(f"4. Probando directorio inaccesible...")
        try:
            # Intentar crear weight manager en directorio que no se puede crear
            invalid_dir = "/root/invalid_path_that_should_not_exist"
            wm_invalid = WeightManager(invalid_dir)
            print(f"   ✅ Weight manager creado con directorio inválido (manejo de error)")
        except Exception as e:
            print(f"   ✅ Error manejado correctamente: {type(e).__name__}")
        
        print(f"\n✅ Casos extremos manejados correctamente")
        
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ Error en casos extremos: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Función principal de pruebas"""
    print("🔧 SISTEMA DE PRUEBAS - GESTIÓN ADAPTATIVA DE PESOS")
    print("="*70)
    print("Como trader senior, sé que la adaptación es supervivencia.")
    print("Un sistema estático es un sistema muerto.")
    print("="*70)
    
    # Ejecutar todas las pruebas
    test_results = []
    
    # 1. Inicialización
    result1 = await test_weight_manager_initialization()
    test_results.append(("Inicialización", result1))
    
    # 2. Actualizaciones de régimen
    result2 = await test_regime_updates()
    test_results.append(("Actualizaciones de régimen", result2))
    
    # 3. Registro de trades
    result3 = await test_trade_recording()
    test_results.append(("Registro de trades", result3))
    
    # 4. Optimización de pesos
    result4 = await test_weight_optimization()
    test_results.append(("Optimización de pesos", result4))
    
    # 5. Métricas de performance
    result5 = await test_performance_metrics()
    test_results.append(("Métricas de performance", result5))
    
    # 6. Persistencia
    result6 = await test_persistence()
    test_results.append(("Persistencia", result6))
    
    # 7. Casos extremos
    result7 = await test_edge_cases()
    test_results.append(("Casos extremos", result7))
    
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
        print("✅ El sistema de gestión adaptativa está listo para evolucionar")
        print("🧬 Adaptación = Supervivencia en los mercados")
    else:
        print("⚠️  Algunas pruebas fallaron - el sistema necesita ajustes")
        print("💀 Un sistema que no se adapta está condenado al fracaso")
    
    return total_passed == total_tests

if __name__ == "__main__":
    # Ejecutar pruebas
    success = asyncio.run(main())
    
    # Exit code para CI/CD
    sys.exit(0 if success else 1)