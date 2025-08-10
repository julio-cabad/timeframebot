#!/usr/bin/env python3
"""
Script de prueba para los analizadores de timeframe
Verifica que todos los analizadores funcionen correctamente
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Agregar el directorio del proyecto al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_bot.analysis import create_analyzer, analyze_all_timeframes
from trading_bot.data.fetcher import fetch_symbol_data

def create_sample_data(periods: int = 100) -> pd.DataFrame:
    """Crea datos de muestra para testing"""
    
    # Generar fechas
    dates = pd.date_range(start=datetime.now() - timedelta(days=periods), 
                         periods=periods, freq='H')
    
    # Generar precios simulados (random walk)
    np.random.seed(42)  # Para resultados reproducibles
    
    # Precio inicial
    initial_price = 50000.0
    
    # Generar cambios de precio (random walk con tendencia)
    price_changes = np.random.normal(0.001, 0.02, periods)  # Media 0.1%, std 2%
    
    # Calcular precios de cierre
    close_prices = [initial_price]
    for change in price_changes[1:]:
        new_price = close_prices[-1] * (1 + change)
        close_prices.append(new_price)
    
    close_prices = np.array(close_prices)
    
    # Generar OHLC basado en close
    data = []
    for i, close in enumerate(close_prices):
        # Generar high/low alrededor del close
        volatility = np.random.uniform(0.005, 0.02)  # 0.5% a 2% de volatilidad
        
        high = close * (1 + volatility * np.random.uniform(0.3, 1.0))
        low = close * (1 - volatility * np.random.uniform(0.3, 1.0))
        
        # Open es el close anterior (excepto para el primero)
        if i == 0:
            open_price = close
        else:
            open_price = close_prices[i-1]
        
        # Asegurar que OHLC sea consistente
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generar volumen
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df

async def test_single_analyzer(timeframe: str, symbol: str = "BTCUSDT"):
    """Prueba un analizador específico"""
    print(f"\n{'='*50}")
    print(f"PROBANDO ANALIZADOR {timeframe.upper()}")
    print(f"{'='*50}")
    
    try:
        # Crear analizador
        analyzer = create_analyzer(timeframe)
        print(f"✅ Analizador {timeframe} creado exitosamente")
        
        # Crear datos de muestra
        data = create_sample_data(200)  # 200 períodos
        print(f"✅ Datos de muestra generados: {len(data)} velas")
        
        # Ejecutar análisis
        analysis = analyzer.analyze(data, symbol)
        print(f"✅ Análisis completado exitosamente")
        
        # Mostrar resultados
        print(f"\nRESULTADOS DEL ANÁLISIS:")
        print(f"  Timeframe: {analysis.timeframe}")
        print(f"  Tendencia: {analysis.trend_direction} (fuerza: {analysis.trend_strength:.3f})")
        print(f"  Momentum: {analysis.momentum_score:.3f}")
        print(f"  Estructura: {analysis.market_structure.value}")
        print(f"  RSI: {analysis.rsi:.1f}")
        print(f"  MACD: {analysis.macd_signal}")
        print(f"  Score General: {analysis.overall_score:.1f}/100")
        print(f"  Confianza: {analysis.confidence:.3f}")
        print(f"  Patrones detectados: {len(analysis.patterns)}")
        print(f"  Niveles de soporte: {len(analysis.support_levels)}")
        print(f"  Niveles de resistencia: {len(analysis.resistance_levels)}")
        print(f"  Volatilidad: {analysis.volatility:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en analizador {timeframe}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_all_analyzers():
    """Prueba todos los analizadores"""
    print("🚀 INICIANDO PRUEBAS DE ANALIZADORES")
    print("="*60)
    
    timeframes = ['1d', '4h', '1h', '15m']
    results = {}
    
    for timeframe in timeframes:
        success = await test_single_analyzer(timeframe)
        results[timeframe] = success
    
    # Resumen final
    print(f"\n{'='*60}")
    print("RESUMEN DE PRUEBAS")
    print(f"{'='*60}")
    
    total_tests = len(timeframes)
    passed_tests = sum(results.values())
    
    for timeframe, success in results.items():
        status = "✅ PASÓ" if success else "❌ FALLÓ"
        print(f"  {timeframe.upper()}: {status}")
    
    print(f"\nResultado: {passed_tests}/{total_tests} pruebas pasaron")
    
    if passed_tests == total_tests:
        print("🎉 ¡TODAS LAS PRUEBAS PASARON!")
    else:
        print("⚠️  Algunas pruebas fallaron")
    
    return passed_tests == total_tests

async def test_analyze_all_timeframes():
    """Prueba la función de análisis completo"""
    print(f"\n{'='*60}")
    print("PROBANDO ANÁLISIS MULTI-TIMEFRAME")
    print(f"{'='*60}")
    
    try:
        # Crear datos para todos los timeframes
        timeframes = ['1d', '4h', '1h', '15m']
        data_dict = {}
        
        for tf in timeframes:
            data_dict[tf] = create_sample_data(100)
        
        print(f"✅ Datos generados para {len(timeframes)} timeframes")
        
        # Ejecutar análisis completo
        results = await analyze_all_timeframes(data_dict, "BTCUSDT")
        
        print(f"✅ Análisis multi-timeframe completado")
        print(f"  Timeframes analizados: {len(results)}")
        
        # Mostrar resumen
        print(f"\nRESUMEN MULTI-TIMEFRAME:")
        for tf, analysis in results.items():
            print(f"  {tf.upper()}: {analysis.trend_direction} "
                  f"(score: {analysis.overall_score:.1f}, "
                  f"confianza: {analysis.confidence:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en análisis multi-timeframe: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_with_real_data():
    """Prueba con datos reales de Binance (si están disponibles)"""
    print(f"\n{'='*60}")
    print("PROBANDO CON DATOS REALES (OPCIONAL)")
    print(f"{'='*60}")
    
    try:
        # Intentar obtener datos reales
        print("Intentando obtener datos reales de Binance...")
        
        real_data = await fetch_symbol_data("BTCUSDT", ["1h"], limit=100)
        
        if real_data and "1h" in real_data:
            print("✅ Datos reales obtenidos exitosamente")
            
            # Probar con datos reales
            analyzer = create_analyzer("1h")
            analysis = analyzer.analyze(real_data["1h"].data, "BTCUSDT")
            
            print(f"✅ Análisis con datos reales completado")
            print(f"  Precio actual: ${real_data['1h'].data['close'].iloc[-1]:.2f}")
            print(f"  Tendencia: {analysis.trend_direction}")
            print(f"  Score: {analysis.overall_score:.1f}/100")
            print(f"  Confianza: {analysis.confidence:.3f}")
            
            return True
        else:
            print("⚠️  No se pudieron obtener datos reales (normal en testing)")
            return True  # No es un error crítico
            
    except Exception as e:
        print(f"⚠️  Error obteniendo datos reales: {str(e)}")
        print("   (Esto es normal si no hay configuración de API)")
        return True  # No es un error crítico

async def main():
    """Función principal de pruebas"""
    print("🔧 SISTEMA DE PRUEBAS - ANALIZADORES DE TIMEFRAME")
    print("="*70)
    
    # Ejecutar todas las pruebas
    test_results = []
    
    # 1. Probar analizadores individuales
    result1 = await test_all_analyzers()
    test_results.append(("Analizadores individuales", result1))
    
    # 2. Probar análisis multi-timeframe
    result2 = await test_analyze_all_timeframes()
    test_results.append(("Análisis multi-timeframe", result2))
    
    # 3. Probar con datos reales (opcional)
    result3 = await test_with_real_data()
    test_results.append(("Datos reales (opcional)", result3))
    
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
        print("✅ Los analizadores están listos para usar")
    else:
        print("⚠️  Algunas pruebas fallaron - revisar implementación")
    
    return total_passed == total_tests

if __name__ == "__main__":
    # Ejecutar pruebas
    success = asyncio.run(main())
    
    # Exit code para CI/CD
    sys.exit(0 if success else 1)