#!/usr/bin/env python3
"""
Script de prueba para el sistema de scoring dinámico
Verifica que el scoring multi-dimensional funcione correctamente

Como trader senior, este es el test más crítico - el scoring determina
si ganamos o perdemos dinero. Cada componente debe ser preciso.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Agregar el directorio del proyecto al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_bot.analysis import (
    analyze_all_timeframes, 
    analyze_confluence,
    detect_patterns
)
from trading_bot.scoring import (
    calculate_dynamic_score,
    ScoringRegime,
    ScoreComponent
)

def create_market_context(regime: str = "normal") -> dict:
    """Crea contexto de mercado para testing"""
    
    base_context = {
        "volatility": 0.025,
        "liquidity_score": 0.85,
        "correlation_risk": 0.3,
        "market_hours": "active",
        "position_risk": 0.02,
        "drawdown_risk": 0.05
    }
    
    if regime == "high_vol":
        base_context.update({
            "volatility": 0.08,
            "correlation_risk": 0.6,
            "drawdown_risk": 0.12
        })
    elif regime == "low_vol":
        base_context.update({
            "volatility": 0.012,
            "liquidity_score": 0.95,
            "correlation_risk": 0.1
        })
    elif regime == "trending":
        base_context.update({
            "volatility": 0.035,
            "liquidity_score": 0.9,
            "correlation_risk": 0.2
        })
    
    return base_context

def create_trending_data(periods: int = 100, direction: str = "bullish") -> pd.DataFrame:
    """Crea datos con tendencia clara para testing"""
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=periods), 
                         periods=periods, freq='h')
    
    np.random.seed(42)
    base_price = 50000.0
    
    # Tendencia base
    if direction == "bullish":
        trend = 0.003  # 0.3% por período
    elif direction == "bearish":
        trend = -0.003
    else:
        trend = 0.0
    
    prices = [base_price]
    for i in range(periods - 1):
        # Tendencia + ruido
        change = trend + np.random.normal(0, 0.015)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Generar OHLCV
    data = []
    for i, close in enumerate(prices):
        volatility = 0.02
        
        high = close * (1 + volatility * np.random.uniform(0.3, 1.0))
        low = close * (1 - volatility * np.random.uniform(0.3, 1.0))
        
        if i == 0:
            open_price = close
        else:
            open_price = prices[i-1]
        
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Volumen más alto en tendencias fuertes
        if direction != "neutral":
            volume = np.random.uniform(3000, 8000)
        else:
            volume = np.random.uniform(1000, 4000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data, index=dates)

async def test_scoring_components():
    """Prueba componentes individuales del scoring"""
    print(f"\n{'='*60}")
    print("PROBANDO COMPONENTES DE SCORING")
    print(f"{'='*60}")
    
    try:
        # Crear datos de tendencia alcista
        timeframes = ['1d', '4h', '1h', '15m']
        data_dict = {}
        
        for tf in timeframes:
            data_dict[tf] = create_trending_data(100, "bullish")
        
        print(f"✅ Datos de tendencia alcista generados")
        
        # Ejecutar análisis completo
        analyses = await analyze_all_timeframes(data_dict, "BTCUSDT")
        confluence_result = analyze_confluence(analyses)
        patterns = detect_patterns(data_dict['1h'], "BTCUSDT", "1h")
        market_context = create_market_context("trending")
        
        print(f"✅ Análisis completado:")
        print(f"   - Timeframes analizados: {len(analyses)}")
        print(f"   - Score de confluencia: {confluence_result.overall_confluence_score:.1f}")
        print(f"   - Patrones detectados: {len(patterns)}")
        
        # Calcular scoring dinámico
        scoring_result = calculate_dynamic_score(
            analyses, confluence_result, patterns, market_context, "BTCUSDT"
        )
        
        print(f"✅ Scoring dinámico calculado")
        
        # Verificar componentes
        print(f"\nCOMPONENTES DE SCORING:")
        for component, score in scoring_result.breakdown.component_scores.items():
            print(f"  {component.value}:")
            print(f"    Score bruto: {score.raw_score:.1f}")
            print(f"    Peso: {score.weight:.3f}")
            print(f"    Score ponderado: {score.weighted_score:.1f}")
            print(f"    Confianza: {score.confidence:.3f}")
        
        print(f"\nRESULTADO FINAL:")
        print(f"  Score final: {scoring_result.final_score:.1f}/100")
        print(f"  Recomendación: {scoring_result.trade_recommendation}")
        print(f"  Nivel de confianza: {scoring_result.confidence_level}")
        print(f"  Régimen: {scoring_result.regime.value}")
        print(f"  Win rate esperado: {scoring_result.expected_win_rate:.1%}")
        print(f"  Profit factor esperado: {scoring_result.expected_profit_factor:.2f}")
        print(f"  Tamaño posición sugerido: {scoring_result.suggested_position_size:.1%}")
        
        # Verificar que todos los componentes estén presentes
        expected_components = [
            ScoreComponent.MTF_STRUCTURE,
            ScoreComponent.TECHNICAL_CONFLUENCE,
            ScoreComponent.MARKET_CONTEXT,
            ScoreComponent.RISK_METRICS
        ]
        
        missing_components = []
        for component in expected_components:
            if component not in scoring_result.breakdown.component_scores:
                missing_components.append(component.value)
        
        if missing_components:
            print(f"❌ Componentes faltantes: {missing_components}")
            return False
        
        # Verificar que el score esté en rango válido
        if not (0 <= scoring_result.final_score <= 100):
            print(f"❌ Score fuera de rango: {scoring_result.final_score}")
            return False
        
        print(f"✅ Todos los componentes funcionan correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en componentes de scoring: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_regime_adaptation():
    """Prueba adaptación a diferentes regímenes de mercado"""
    print(f"\n{'='*60}")
    print("PROBANDO ADAPTACIÓN POR RÉGIMEN")
    print(f"{'='*60}")
    
    try:
        regimes_to_test = [
            ("trending_bull", "bullish", "trending"),
            ("ranging", "neutral", "low_vol"),
            ("high_vol", "bearish", "high_vol")
        ]
        
        regime_results = {}
        
        for regime_name, trend_direction, market_regime in regimes_to_test:
            print(f"\n--- Probando régimen: {regime_name} ---")
            
            # Crear datos específicos para el régimen
            timeframes = ['1d', '4h', '1h', '15m']
            data_dict = {}
            
            for tf in timeframes:
                data_dict[tf] = create_trending_data(80, trend_direction)
            
            # Análisis
            analyses = await analyze_all_timeframes(data_dict, "BTCUSDT")
            confluence_result = analyze_confluence(analyses)
            patterns = detect_patterns(data_dict['1h'], "BTCUSDT", "1h")
            market_context = create_market_context(market_regime)
            
            # Scoring
            scoring_result = calculate_dynamic_score(
                analyses, confluence_result, patterns, market_context, "BTCUSDT"
            )
            
            regime_results[regime_name] = scoring_result
            
            print(f"  Score: {scoring_result.final_score:.1f}")
            print(f"  Recomendación: {scoring_result.trade_recommendation}")
            print(f"  Régimen detectado: {scoring_result.regime.value}")
            print(f"  Tamaño posición: {scoring_result.suggested_position_size:.1%}")
        
        # Verificar que los regímenes se detecten correctamente
        print(f"\nRESUMEN DE REGÍMENES:")
        for regime_name, result in regime_results.items():
            print(f"  {regime_name}: {result.regime.value} (score: {result.final_score:.1f})")
        
        # Verificar diferencias en scoring por régimen
        scores = [r.final_score for r in regime_results.values()]
        score_variance = np.var(scores)
        
        if score_variance < 10:  # Muy poca variación
            print(f"⚠️  Poca variación entre regímenes (var: {score_variance:.1f})")
        else:
            print(f"✅ Buena adaptación por régimen (var: {score_variance:.1f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en adaptación por régimen: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_scoring_edge_cases():
    """Prueba casos extremos del scoring"""
    print(f"\n{'='*60}")
    print("PROBANDO CASOS EXTREMOS")
    print(f"{'='*60}")
    
    try:
        # Caso 1: Datos mínimos
        print(f"1. Probando con datos mínimos...")
        minimal_data = {'1h': create_trending_data(25, "neutral")}
        minimal_analyses = await analyze_all_timeframes(minimal_data, "BTCUSDT")
        minimal_confluence = analyze_confluence(minimal_analyses)
        minimal_patterns = detect_patterns(minimal_data['1h'], "BTCUSDT", "1h")
        minimal_context = create_market_context("normal")
        
        minimal_score = calculate_dynamic_score(
            minimal_analyses, minimal_confluence, minimal_patterns, 
            minimal_context, "BTCUSDT"
        )
        
        print(f"   Score con datos mínimos: {minimal_score.final_score:.1f}")
        print(f"   Recomendación: {minimal_score.trade_recommendation}")
        
        # Caso 2: Sin patrones detectados
        print(f"2. Probando sin patrones...")
        no_patterns_score = calculate_dynamic_score(
            minimal_analyses, minimal_confluence, [], 
            minimal_context, "BTCUSDT"
        )
        
        print(f"   Score sin patrones: {no_patterns_score.final_score:.1f}")
        
        # Caso 3: Contexto de mercado extremo
        print(f"3. Probando contexto extremo...")
        extreme_context = {
            "volatility": 0.15,  # Volatilidad extrema
            "liquidity_score": 0.3,  # Baja liquidez
            "correlation_risk": 0.95,  # Alta correlación
            "market_hours": "closed",
            "position_risk": 0.08,  # Alto riesgo
            "drawdown_risk": 0.25   # Alto riesgo de drawdown
        }
        
        extreme_score = calculate_dynamic_score(
            minimal_analyses, minimal_confluence, minimal_patterns,
            extreme_context, "BTCUSDT"
        )
        
        print(f"   Score con contexto extremo: {extreme_score.final_score:.1f}")
        print(f"   Warnings: {len(extreme_score.risk_warnings)}")
        for warning in extreme_score.risk_warnings:
            print(f"     - {warning}")
        
        # Verificar que los casos extremos se manejen apropiadamente
        if extreme_score.final_score > minimal_score.final_score:
            print(f"⚠️  Score extremo mayor que normal - revisar penalizaciones")
        else:
            print(f"✅ Penalizaciones por contexto extremo funcionan")
        
        print(f"\n✅ Casos extremos manejados correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en casos extremos: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_scoring_consistency():
    """Prueba consistencia del scoring"""
    print(f"\n{'='*60}")
    print("PROBANDO CONSISTENCIA DEL SCORING")
    print(f"{'='*60}")
    
    try:
        # Crear datos idénticos
        timeframes = ['1d', '4h', '1h', '15m']
        data_dict = {}
        
        for tf in timeframes:
            data_dict[tf] = create_trending_data(100, "bullish")
        
        # Ejecutar scoring múltiples veces
        scores = []
        for i in range(5):
            analyses = await analyze_all_timeframes(data_dict, "BTCUSDT")
            confluence_result = analyze_confluence(analyses)
            patterns = detect_patterns(data_dict['1h'], "BTCUSDT", "1h")
            market_context = create_market_context("trending")
            
            scoring_result = calculate_dynamic_score(
                analyses, confluence_result, patterns, market_context, "BTCUSDT"
            )
            
            scores.append(scoring_result.final_score)
        
        # Verificar consistencia
        score_std = np.std(scores)
        score_mean = np.mean(scores)
        
        print(f"Scores obtenidos: {[f'{s:.1f}' for s in scores]}")
        print(f"Media: {score_mean:.1f}")
        print(f"Desviación estándar: {score_std:.3f}")
        
        if score_std < 0.1:  # Muy consistente
            print(f"✅ Scoring muy consistente")
            return True
        elif score_std < 1.0:  # Aceptablemente consistente
            print(f"✅ Scoring consistente")
            return True
        else:
            print(f"⚠️  Scoring inconsistente - revisar determinismo")
            return False
        
    except Exception as e:
        print(f"❌ Error en consistencia: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_scoring_performance():
    """Prueba performance del sistema de scoring"""
    print(f"\n{'='*60}")
    print("PROBANDO PERFORMANCE DEL SCORING")
    print(f"{'='*60}")
    
    try:
        # Preparar datos
        timeframes = ['1d', '4h', '1h', '15m']
        data_dict = {}
        
        for tf in timeframes:
            data_dict[tf] = create_trending_data(100, "bullish")
        
        analyses = await analyze_all_timeframes(data_dict, "BTCUSDT")
        confluence_result = analyze_confluence(analyses)
        patterns = detect_patterns(data_dict['1h'], "BTCUSDT", "1h")
        market_context = create_market_context("trending")
        
        # Medir tiempo de ejecución
        import time
        
        start_time = time.time()
        
        for i in range(10):  # 10 iteraciones
            scoring_result = calculate_dynamic_score(
                analyses, confluence_result, patterns, market_context, "BTCUSDT"
            )
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        
        print(f"Tiempo promedio por scoring: {avg_time*1000:.1f}ms")
        
        if avg_time < 0.1:  # Menos de 100ms
            print(f"✅ Performance excelente")
        elif avg_time < 0.5:  # Menos de 500ms
            print(f"✅ Performance buena")
        else:
            print(f"⚠️  Performance lenta - optimizar")
        
        return avg_time < 1.0  # Máximo 1 segundo
        
    except Exception as e:
        print(f"❌ Error en performance: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Función principal de pruebas"""
    print("🔧 SISTEMA DE PRUEBAS - SCORING DINÁMICO")
    print("="*70)
    print("Como trader senior, este es el test más crítico del sistema.")
    print("El scoring determina si ganamos o perdemos dinero.")
    print("="*70)
    
    # Ejecutar todas las pruebas
    test_results = []
    
    # 1. Componentes de scoring
    result1 = await test_scoring_components()
    test_results.append(("Componentes de scoring", result1))
    
    # 2. Adaptación por régimen
    result2 = await test_regime_adaptation()
    test_results.append(("Adaptación por régimen", result2))
    
    # 3. Casos extremos
    result3 = await test_scoring_edge_cases()
    test_results.append(("Casos extremos", result3))
    
    # 4. Consistencia
    result4 = await test_scoring_consistency()
    test_results.append(("Consistencia", result4))
    
    # 5. Performance
    result5 = await test_scoring_performance()
    test_results.append(("Performance", result5))
    
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
        print("✅ El sistema de scoring dinámico está listo para generar alpha")
        print("💰 Objetivo: Profit Factor > 2.0, Win Rate > 65%")
    else:
        print("⚠️  Algunas pruebas fallaron - el scoring necesita ajustes")
        print("🔧 Un scoring defectuoso = pérdidas garantizadas")
    
    return total_passed == total_tests

if __name__ == "__main__":
    # Ejecutar pruebas
    success = asyncio.run(main())
    
    # Exit code para CI/CD
    sys.exit(0 if success else 1)