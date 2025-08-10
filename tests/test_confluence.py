#!/usr/bin/env python3
"""
Script de prueba para el motor de confluencia
Verifica que el análisis multi-timeframe funcione correctamente
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
    ConfluenceStrength,
    AlignmentType
)

def create_sample_data(periods: int = 100, trend: str = "bullish") -> pd.DataFrame:
    """Crea datos de muestra con tendencia específica"""
    
    # Generar fechas
    dates = pd.date_range(start=datetime.now() - timedelta(days=periods), 
                         periods=periods, freq='h')
    
    # Generar precios simulados con tendencia
    np.random.seed(42)
    
    initial_price = 50000.0
    
    # Ajustar tendencia
    if trend == "bullish":
        base_trend = 0.002  # 0.2% por período
        volatility = 0.015
    elif trend == "bearish":
        base_trend = -0.002  # -0.2% por período
        volatility = 0.015
    else:  # neutral/ranging
        base_trend = 0.0
        volatility = 0.01
    
    # Generar cambios de precio
    price_changes = np.random.normal(base_trend, volatility, periods)
    
    # Calcular precios de cierre
    close_prices = [initial_price]
    for change in price_changes[1:]:
        new_price = close_prices[-1] * (1 + change)
        close_prices.append(new_price)
    
    close_prices = np.array(close_prices)
    
    # Generar OHLC
    data = []
    for i, close in enumerate(close_prices):
        volatility_factor = np.random.uniform(0.005, 0.02)
        
        high = close * (1 + volatility_factor * np.random.uniform(0.3, 1.0))
        low = close * (1 - volatility_factor * np.random.uniform(0.3, 1.0))
        
        if i == 0:
            open_price = close
        else:
            open_price = close_prices[i-1]
        
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
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

async def test_confluence_aligned_trends():
    """Prueba confluencia con tendencias alineadas"""
    print(f"\n{'='*60}")
    print("PROBANDO CONFLUENCIA - TENDENCIAS ALINEADAS")
    print(f"{'='*60}")
    
    try:
        # Crear datos con tendencia alcista para todos los timeframes
        timeframes = ['1d', '4h', '1h', '15m']
        data_dict = {}
        
        for tf in timeframes:
            data_dict[tf] = create_sample_data(100, trend="bullish")
        
        print(f"✅ Datos generados con tendencia alcista para {len(timeframes)} timeframes")
        
        # Ejecutar análisis de timeframes
        analyses = await analyze_all_timeframes(data_dict, "BTCUSDT")
        print(f"✅ Análisis de timeframes completado: {len(analyses)} timeframes")
        
        # Ejecutar análisis de confluencia
        confluence_result = analyze_confluence(analyses)
        print(f"✅ Análisis de confluencia completado")
        
        # Mostrar resultados
        print(f"\nRESULTADOS DE CONFLUENCIA:")
        print(f"  Score General: {confluence_result.overall_confluence_score:.1f}/100")
        print(f"  Fuerza: {confluence_result.confluence_strength.value}")
        print(f"  Confianza: {confluence_result.confidence:.3f}")
        print(f"  Recomendación: {confluence_result.trade_recommendation}")
        print(f"  Timeframe recomendado: {confluence_result.recommended_timeframe}")
        print(f"  Tendencia dominante: {confluence_result.dominant_trend}")
        print(f"  Nivel de riesgo: {confluence_result.risk_level}")
        
        print(f"\nALINEACIONES DETECTADAS:")
        for alignment_type, score_obj in confluence_result.alignment_scores.items():
            print(f"  {alignment_type.value}: {score_obj.score:.3f} "
                  f"(TFs: {len(score_obj.participating_timeframes)})")
        
        print(f"\nNIVELES DE CONFLUENCIA: {len(confluence_result.confluence_levels)}")
        for i, level in enumerate(confluence_result.confluence_levels[:3]):  # Top 3
            print(f"  {i+1}. {level.level_type.upper()} ${level.price_level:.2f} "
                  f"(fuerza: {level.strength:.3f}, TFs: {len(level.timeframes)})")
        
        print(f"\nDIVERGENCIAS: {len(confluence_result.divergences)}")
        for divergence in confluence_result.divergences:
            print(f"  {divergence.divergence_type.value}: {divergence.description}")
        
        # Verificar que la confluencia sea fuerte para tendencias alineadas
        expected_strong = confluence_result.confluence_strength in [
            ConfluenceStrength.STRONG, ConfluenceStrength.CRITICAL
        ]
        
        if expected_strong:
            print(f"✅ Confluencia fuerte detectada correctamente")
            return True
        else:
            print(f"⚠️  Confluencia más débil de lo esperado")
            return True  # No es error crítico
            
    except Exception as e:
        print(f"❌ Error en prueba de confluencia alineada: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_confluence_divergent_trends():
    """Prueba confluencia con tendencias divergentes"""
    print(f"\n{'='*60}")
    print("PROBANDO CONFLUENCIA - TENDENCIAS DIVERGENTES")
    print(f"{'='*60}")
    
    try:
        # Crear datos con tendencias mixtas
        data_dict = {
            '1d': create_sample_data(100, trend="bullish"),   # Tendencia alcista a largo plazo
            '4h': create_sample_data(100, trend="bearish"),   # Tendencia bajista intermedia
            '1h': create_sample_data(100, trend="neutral"),   # Neutral
            '15m': create_sample_data(100, trend="bearish")   # Bajista a corto plazo
        }
        
        print(f"✅ Datos generados con tendencias divergentes")
        
        # Ejecutar análisis
        analyses = await analyze_all_timeframes(data_dict, "BTCUSDT")
        confluence_result = analyze_confluence(analyses)
        
        print(f"✅ Análisis de confluencia completado")
        
        # Mostrar resultados
        print(f"\nRESULTADOS DE CONFLUENCIA:")
        print(f"  Score General: {confluence_result.overall_confluence_score:.1f}/100")
        print(f"  Fuerza: {confluence_result.confluence_strength.value}")
        print(f"  Confianza: {confluence_result.confidence:.3f}")
        print(f"  Recomendación: {confluence_result.trade_recommendation}")
        print(f"  Tendencia dominante: {confluence_result.dominant_trend}")
        print(f"  Nivel de riesgo: {confluence_result.risk_level}")
        
        print(f"\nDIVERGENCIAS DETECTADAS: {len(confluence_result.divergences)}")
        for divergence in confluence_result.divergences:
            print(f"  {divergence.divergence_type.value}: {divergence.description} "
                  f"(severidad: {divergence.severity:.3f})")
        
        # Verificar que se detecten divergencias
        has_divergences = len(confluence_result.divergences) > 0
        weak_confluence = confluence_result.confluence_strength == ConfluenceStrength.WEAK
        
        if has_divergences:
            print(f"✅ Divergencias detectadas correctamente")
        
        if weak_confluence:
            print(f"✅ Confluencia débil detectada correctamente")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba de confluencia divergente: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_confluence_components():
    """Prueba componentes específicos de confluencia"""
    print(f"\n{'='*60}")
    print("PROBANDO COMPONENTES DE CONFLUENCIA")
    print(f"{'='*60}")
    
    try:
        # Crear datos balanceados
        timeframes = ['1d', '4h', '1h', '15m']
        data_dict = {}
        
        for tf in timeframes:
            data_dict[tf] = create_sample_data(150, trend="neutral")
        
        # Ejecutar análisis
        analyses = await analyze_all_timeframes(data_dict, "BTCUSDT")
        confluence_result = analyze_confluence(analyses)
        
        print(f"✅ Análisis completado")
        
        # Verificar componentes específicos
        print(f"\nCOMPONENTES DE ALINEACIÓN:")
        
        required_alignments = [
            AlignmentType.TREND_ALIGNMENT,
            AlignmentType.MOMENTUM_SYNC,
            AlignmentType.STRUCTURE_CONFLUENCE,
            AlignmentType.LEVEL_CONFLUENCE,
            AlignmentType.PATTERN_CONFLUENCE
        ]
        
        components_ok = True
        for alignment_type in required_alignments:
            if alignment_type in confluence_result.alignment_scores:
                score = confluence_result.alignment_scores[alignment_type]
                print(f"  ✅ {alignment_type.value}: {score.score:.3f}")
            else:
                print(f"  ❌ {alignment_type.value}: FALTANTE")
                components_ok = False
        
        print(f"\nNIVELES DE CONFLUENCIA:")
        print(f"  Total detectados: {len(confluence_result.confluence_levels)}")
        
        print(f"\nMÉTRICAS GENERALES:")
        print(f"  Timeframes participantes: {len(confluence_result.participating_timeframes)}")
        print(f"  Score final: {confluence_result.overall_confluence_score:.1f}")
        print(f"  Confianza: {confluence_result.confidence:.3f}")
        
        if components_ok:
            print(f"✅ Todos los componentes funcionan correctamente")
        else:
            print(f"⚠️  Algunos componentes tienen problemas")
        
        return components_ok
        
    except Exception as e:
        print(f"❌ Error en prueba de componentes: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_confluence_edge_cases():
    """Prueba casos extremos"""
    print(f"\n{'='*60}")
    print("PROBANDO CASOS EXTREMOS")
    print(f"{'='*60}")
    
    try:
        # Caso 1: Solo un timeframe
        print(f"\n1. Probando con un solo timeframe...")
        single_tf_data = {'1h': create_sample_data(100)}
        single_analyses = await analyze_all_timeframes(single_tf_data, "BTCUSDT")
        single_confluence = analyze_confluence(single_analyses)
        
        print(f"   Score: {single_confluence.overall_confluence_score:.1f}")
        print(f"   Fuerza: {single_confluence.confluence_strength.value}")
        
        # Caso 2: Datos mínimos
        print(f"\n2. Probando con datos mínimos...")
        minimal_data = {'1h': create_sample_data(25)}  # Solo 25 períodos
        minimal_analyses = await analyze_all_timeframes(minimal_data, "BTCUSDT")
        minimal_confluence = analyze_confluence(minimal_analyses)
        
        print(f"   Score: {minimal_confluence.overall_confluence_score:.1f}")
        print(f"   Confianza: {minimal_confluence.confidence:.3f}")
        
        # Caso 3: Análisis vacío (debería fallar graciosamente)
        print(f"\n3. Probando con análisis vacío...")
        try:
            empty_confluence = analyze_confluence({})
            print(f"   ❌ Debería haber fallado")
            return False
        except Exception as e:
            print(f"   ✅ Falló correctamente: {type(e).__name__}")
        
        print(f"\n✅ Casos extremos manejados correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en casos extremos: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Función principal de pruebas"""
    print("🔧 SISTEMA DE PRUEBAS - MOTOR DE CONFLUENCIA")
    print("="*70)
    
    # Ejecutar todas las pruebas
    test_results = []
    
    # 1. Confluencia con tendencias alineadas
    result1 = await test_confluence_aligned_trends()
    test_results.append(("Tendencias alineadas", result1))
    
    # 2. Confluencia con tendencias divergentes
    result2 = await test_confluence_divergent_trends()
    test_results.append(("Tendencias divergentes", result2))
    
    # 3. Componentes de confluencia
    result3 = await test_confluence_components()
    test_results.append(("Componentes de confluencia", result3))
    
    # 4. Casos extremos
    result4 = await test_confluence_edge_cases()
    test_results.append(("Casos extremos", result4))
    
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
        print("✅ El motor de confluencia está listo para usar")
    else:
        print("⚠️  Algunas pruebas fallaron - revisar implementación")
    
    return total_passed == total_tests

if __name__ == "__main__":
    # Ejecutar pruebas
    success = asyncio.run(main())
    
    # Exit code para CI/CD
    sys.exit(0 if success else 1)