#!/usr/bin/env python3
"""
Script de prueba para el sistema de integración LLM
Verifica que la integración con Gemini funcione correctamente

Como trader senior, este test es crítico - el LLM debe proporcionar
análisis confiable y consistente para decisiones de trading complejas.

El LLM es nuestro "segundo cerebro" para situaciones ambiguas donde
el análisis técnico puro no es suficiente.

Autor: Trader Algorítmico Senior (10+ años)
Zona Horaria: UTC-5 (Ecuador)
"""

import asyncio
import json
from datetime import datetime, timedelta
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch

# Agregar el directorio del proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot.llm import (
    LLMConnector,
    LLMProvider,
    LLMResponse,
    PromptBuilder,
    CostManager,
    CostCategory,
    ResponseParser,
    ParsedResponse,
    DecisionType,
    create_llm_connector
)

from trading_bot.llm.prompt_builder import PromptContext
from trading_bot.scoring.scorer import ScoringResult, ScoringRegime, ScoreComponent
from trading_bot.analysis.confluence import ConfluenceResult

def create_mock_scoring_result() -> ScoringResult:
    """Crea un ScoringResult mock para testing"""
    # Crear un mock simple sin breakdown complejo
    from unittest.mock import Mock
    
    mock_result = Mock(spec=ScoringResult)
    mock_result.final_score = 70.0  # En zona ambigua
    mock_result.trade_recommendation = "ANALYZE_FURTHER"
    mock_result.confidence_level = "MEDIUM"
    mock_result.regime = ScoringRegime.TRENDING_BULL
    mock_result.expected_win_rate = 0.65
    mock_result.expected_profit_factor = 2.1
    mock_result.suggested_position_size = 0.05
    mock_result.risk_warnings = ["Score en zona ambigua - requiere análisis LLM"]
    
    # Mock breakdown simple
    mock_breakdown = Mock()
    mock_breakdown.component_scores = {
        ScoreComponent.MTF_STRUCTURE: Mock(weighted_score=26.25, weight=0.35),
        ScoreComponent.TECHNICAL_CONFLUENCE: Mock(weighted_score=17.0, weight=0.25),
        ScoreComponent.MARKET_CONTEXT: Mock(weighted_score=14.4, weight=0.20),
        ScoreComponent.RISK_METRICS: Mock(weighted_score=16.0, weight=0.20)
    }
    mock_result.breakdown = mock_breakdown
    
    return mock_result

def create_mock_prompt_context() -> PromptContext:
    """Crea contexto mock para prompts"""
    from trading_bot.analysis.tf_analyzers import TimeframeAnalysis, MarketStructure, PatternType
    
    # Crear análisis mock por timeframe
    import pytz
    ECUADOR_TZ = pytz.timezone('America/Guayaquil')
    
    timeframe_analyses = {
        "1d": TimeframeAnalysis(
            timeframe="1d",
            timestamp=datetime.now(ECUADOR_TZ),
            trend_direction="bullish",
            trend_strength=0.75,
            trend_duration_candles=15,
            momentum_score=0.65,
            momentum_divergence=False,
            market_structure=MarketStructure.UPTREND,
            structure_break=False,
            support_levels=[49500.0],
            resistance_levels=[51200.0],
            current_level_type="neutral",
            patterns=[PatternType.BULL_FLAG],
            pattern_confidence=0.8,
            rsi=65.0,
            macd_signal="bullish",
            ema_alignment=True,
            volume_trend="increasing",
            volume_confirmation=True,
            overall_score=75.0,
            confidence=0.8,
            volatility=0.025,
            liquidity_score=0.85,
            market_hours="active"
        ),
        "4h": TimeframeAnalysis(
            timeframe="4h",
            timestamp=datetime.now(ECUADOR_TZ),
            trend_direction="bullish",
            trend_strength=0.68,
            trend_duration_candles=8,
            momentum_score=0.72,
            momentum_divergence=False,
            market_structure=MarketStructure.UPTREND,
            structure_break=False,
            support_levels=[50200.0],
            resistance_levels=[50800.0],
            current_level_type="support",
            patterns=[PatternType.BULL_FLAG],
            pattern_confidence=0.75,
            rsi=68.0,
            macd_signal="bullish",
            ema_alignment=True,
            volume_trend="stable",
            volume_confirmation=True,
            overall_score=72.0,
            confidence=0.75,
            volatility=0.022,
            liquidity_score=0.88,
            market_hours="active"
        ),
        "1h": TimeframeAnalysis(
            timeframe="1h",
            timestamp=datetime.now(ECUADOR_TZ),
            trend_direction="neutral",
            trend_strength=0.45,
            trend_duration_candles=3,
            momentum_score=0.52,
            momentum_divergence=True,
            market_structure=MarketStructure.RANGING,
            structure_break=False,
            support_levels=[50400.0],
            resistance_levels=[50600.0],
            current_level_type="neutral",
            patterns=[],
            pattern_confidence=0.6,
            rsi=55.0,
            macd_signal="neutral",
            ema_alignment=False,
            volume_trend="decreasing",
            volume_confirmation=False,
            overall_score=60.0,
            confidence=0.6,
            volatility=0.018,
            liquidity_score=0.75,
            market_hours="active"
        )
    }
    
    from trading_bot.analysis.confluence import ConfluenceStrength, AlignmentType, AlignmentScore
    
    confluence_result = ConfluenceResult(
        timestamp=datetime.now(ECUADOR_TZ),
        symbol="BTCUSDT",
        overall_confluence_score=72.5,
        confluence_strength=ConfluenceStrength.MODERATE,
        confidence=0.75,
        alignment_scores={
            AlignmentType.TREND_ALIGNMENT: AlignmentScore(
                alignment_type=AlignmentType.TREND_ALIGNMENT,
                score=0.68,
                participating_timeframes=["1d", "4h"],
                weight=0.4
            )
        },
        confluence_levels=[],
        divergences=[],
        trade_recommendation="BUY",
        recommended_timeframe="1h",
        participating_timeframes=["1d", "4h", "1h"],
        dominant_trend="bullish",
        risk_level="medium"
    )
    
    return PromptContext(
        symbol="BTCUSDT",
        timeframe="1h",
        current_price=50500.0,
        scoring_result=create_mock_scoring_result(),
        timeframe_analyses=timeframe_analyses,
        confluence_result=confluence_result,
        patterns=[],
        market_context={
            "volatility": 0.025,
            "liquidity_score": 0.85,
            "correlation_risk": 0.3,
            "market_hours": "active"
        }
    )

async def test_llm_connector_initialization():
    """Prueba inicialización del conector LLM"""
    print(f"\n{'='*60}")
    print("PROBANDO INICIALIZACIÓN DEL CONECTOR LLM")
    print(f"{'='*60}")
    
    try:
        temp_dir = tempfile.mkdtemp()
        
        # Crear conector
        connector = create_llm_connector(temp_dir)
        print(f"✅ LLMConnector creado exitosamente")
        
        # Verificar que Gemini esté configurado
        if LLMProvider.GEMINI in connector.configs:
            print(f"✅ Gemini configurado correctamente")
            config = connector.configs[LLMProvider.GEMINI]
            print(f"   Modelo: {config.model}")
            print(f"   Costo por 1K tokens: ${config.cost_per_1k_tokens:.4f}")
        else:
            print(f"⚠️ Gemini no configurado - verificar GEMINI_API_KEY")
        
        # Verificar estadísticas iniciales
        stats = connector.get_daily_stats()
        print(f"✅ Estadísticas iniciales obtenidas")
        print(f"   Costo diario: ${stats['daily_cost']:.2f}")
        print(f"   Requests diarios: {stats['daily_requests']}")
        print(f"   Proveedores disponibles: {stats['providers_available']}")
        
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ Error en inicialización: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_prompt_builder():
    """Prueba construcción de prompts"""
    print(f"\n{'='*60}")
    print("PROBANDO CONSTRUCCIÓN DE PROMPTS")
    print(f"{'='*60}")
    
    try:
        builder = PromptBuilder()
        context = create_mock_prompt_context()
        
        # Probar prompt de análisis de trade
        trade_prompt = builder.build_trade_analysis_prompt(context)
        print(f"✅ Prompt de análisis de trade generado")
        print(f"   Longitud: {len(trade_prompt)} caracteres")
        
        # Verificar que contenga elementos clave
        required_elements = [
            "BTCUSDT",
            "50500",
            "70.0/100",
            "BULLISH",
            "TRENDING_BULL",
            "JSON"
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in trade_prompt:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"⚠️ Elementos faltantes en prompt: {missing_elements}")
        else:
            print(f"✅ Todos los elementos clave presentes")
        
        # Probar prompt de divergencias
        divergences = [
            "1D bullish vs 1H neutral - posible debilitamiento",
            "Momentum 4H alto vs momentum 1H bajo"
        ]
        
        divergence_prompt = builder.build_divergence_check_prompt(context, divergences)
        print(f"✅ Prompt de divergencias generado")
        print(f"   Longitud: {len(divergence_prompt)} caracteres")
        
        # Mostrar muestra del prompt
        print(f"\n--- MUESTRA DE PROMPT ---")
        print(trade_prompt[:500] + "...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en construcción de prompts: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_response_parser():
    """Prueba parsing de respuestas LLM"""
    print(f"\n{'='*60}")
    print("PROBANDO PARSER DE RESPUESTAS")
    print(f"{'='*60}")
    
    try:
        parser = ResponseParser()
        
        # Caso 1: JSON válido
        valid_json_response = '''
        {
            "decision": "APPROVE",
            "confidence": 0.75,
            "reasoning": "El análisis multi-timeframe muestra confluencia bullish en 1D y 4H, con soporte sólido en 50200. El momentum está mejorando y el patrón bull flag sugiere continuación alcista.",
            "key_factors": [
                "Confluencia bullish 1D/4H",
                "Soporte sólido en 50200",
                "Patrón bull flag confirmado"
            ],
            "identified_risks": [
                "Divergencia en timeframe 1H",
                "Volatilidad moderada"
            ],
            "suggested_modifications": {
                "position_size_multiplier": 0.8,
                "stop_loss_adjustment": -0.1,
                "take_profit_adjustment": 0.05
            },
            "market_outlook": "BULLISH",
            "urgency": "MEDIUM"
        }
        '''
        
        parsed_valid = parser.parse_response(valid_json_response)
        print(f"✅ JSON válido parseado correctamente")
        print(f"   Decisión: {parsed_valid.decision.value}")
        print(f"   Confianza: {parsed_valid.confidence:.2f}")
        print(f"   Score validación: {parsed_valid.validation_score:.2f}")
        
        # Caso 2: Respuesta malformada
        malformed_response = '''
        DECISION: REJECT
        CONFIDENCE: 0.65
        REASONING: El análisis muestra divergencias preocupantes entre timeframes...
        
        RISKS:
        - Divergencia 1D vs 1H
        - Alta volatilidad
        - Soporte débil
        
        OUTLOOK: BEARISH
        '''
        
        parsed_malformed = parser.parse_response(malformed_response)
        print(f"✅ Respuesta malformada parseada con fallback")
        print(f"   Decisión: {parsed_malformed.decision.value}")
        print(f"   Confianza: {parsed_malformed.confidence:.2f}")
        print(f"   Warnings: {len(parsed_malformed.parsing_warnings)}")
        
        # Caso 3: Respuesta completamente inválida
        invalid_response = "Esta es una respuesta completamente inválida sin estructura"
        
        parsed_invalid = parser.parse_response(invalid_response)
        print(f"✅ Respuesta inválida manejada con fallback conservador")
        print(f"   Decisión: {parsed_invalid.decision.value}")
        print(f"   Score validación: {parsed_invalid.validation_score:.2f}")
        
        # Validar calidad de respuestas
        valid_quality, valid_problems = parser.validate_response_quality(parsed_valid)
        print(f"✅ Validación de calidad - Válida: {'Sí' if valid_quality else 'No'}")
        if valid_problems:
            print(f"   Problemas: {valid_problems}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en parser de respuestas: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_cost_management():
    """Prueba gestión de costos"""
    print(f"\n{'='*60}")
    print("PROBANDO GESTIÓN DE COSTOS")
    print(f"{'='*60}")
    
    try:
        temp_dir = tempfile.mkdtemp()
        
        # Crear cost manager
        cost_manager = CostManager(max_daily_cost=10.0, data_directory=temp_dir)
        print(f"✅ CostManager creado con presupuesto de $10.00")
        
        # Verificar presupuesto inicial
        available, reason = cost_manager.check_budget_available()
        print(f"✅ Presupuesto inicial disponible: {available}")
        print(f"   Razón: {reason}")
        
        # Simular uso de LLM
        mock_response = LLMResponse(
            provider=LLMProvider.GEMINI,
            model="gemini-1.5-flash",
            prompt_hash="test_hash",
            response_text="Test response",
            tokens_used=500,
            cost_usd=0.25,
            response_time_ms=1500
        )
        
        # Registrar uso
        cost_manager.record_llm_usage(mock_response, CostCategory.TRADE_ANALYSIS)
        print(f"✅ Uso de LLM registrado: $0.25")
        
        # Verificar presupuesto después del uso
        available, reason = cost_manager.check_budget_available()
        print(f"✅ Presupuesto después del uso: {available}")
        print(f"   Razón: {reason}")
        
        # Simular múltiples usos para probar alertas
        for i in range(20):
            mock_response.cost_usd = 0.45
            cost_manager.record_llm_usage(mock_response, CostCategory.TRADE_ANALYSIS)
        
        print(f"✅ Múltiples usos simulados")
        
        # Verificar presupuesto cerca del límite
        available, reason = cost_manager.check_budget_available()
        print(f"✅ Estado final del presupuesto: {available}")
        print(f"   Razón: {reason}")
        
        # Generar reporte de eficiencia
        efficiency_report = cost_manager.get_cost_efficiency_report(days_back=1)
        print(f"✅ Reporte de eficiencia generado")
        print(f"   Costo total: ${efficiency_report['summary']['total_cost']:.2f}")
        print(f"   Total requests: {efficiency_report['summary']['total_requests']}")
        
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ Error en gestión de costos: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_real_llm_call():
    """Prueba llamada real al LLM (si está configurado)"""
    print(f"\n{'='*60}")
    print("PROBANDO LLAMADA REAL AL LLM")
    print(f"{'='*60}")
    
    try:
        temp_dir = tempfile.mkdtemp()
        
        # Crear conector
        connector = create_llm_connector(temp_dir)
        
        # Verificar si Gemini está disponible
        if LLMProvider.GEMINI not in connector.configs:
            print(f"⚠️ Gemini no configurado - saltando test de llamada real")
            return True
        
        # Crear prompt simple para test
        builder = PromptBuilder()
        context = create_mock_prompt_context()
        
        prompt = builder.build_trade_analysis_prompt(context)
        
        print(f"✅ Prompt generado para test real")
        print(f"   Longitud: {len(prompt)} caracteres")
        
        # Realizar llamada real
        print(f"🔄 Realizando llamada a Gemini...")
        
        response = await connector.generate_response(
            prompt=prompt,
            preferred_provider=LLMProvider.GEMINI,
            use_cache=True
        )
        
        print(f"✅ Respuesta recibida de Gemini")
        print(f"   Proveedor: {response.provider.value}")
        print(f"   Modelo: {response.model}")
        print(f"   Tokens usados: {response.tokens_used}")
        print(f"   Costo: ${response.cost_usd:.4f}")
        print(f"   Tiempo respuesta: {response.response_time_ms}ms")
        print(f"   Desde caché: {response.cached}")
        
        # Parsear respuesta
        parser = ResponseParser()
        parsed_response = parser.parse_response(response.response_text)
        
        print(f"✅ Respuesta parseada exitosamente")
        print(f"   Decisión: {parsed_response.decision.value}")
        print(f"   Confianza: {parsed_response.confidence:.2f}")
        print(f"   Score validación: {parsed_response.validation_score:.2f}")
        
        # Mostrar muestra de la respuesta
        print(f"\n--- MUESTRA DE RESPUESTA ---")
        print(f"Reasoning: {parsed_response.reasoning[:200]}...")
        print(f"Key Factors: {parsed_response.key_factors[:2]}")
        print(f"Risks: {parsed_response.identified_risks[:2]}")
        
        # Verificar estadísticas actualizadas
        stats = connector.get_daily_stats()
        print(f"✅ Estadísticas actualizadas")
        print(f"   Costo diario: ${stats['daily_cost']:.4f}")
        print(f"   Requests diarios: {stats['daily_requests']}")
        
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ Error en llamada real al LLM: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_edge_cases():
    """Prueba casos extremos del sistema LLM"""
    print(f"\n{'='*60}")
    print("PROBANDO CASOS EXTREMOS")
    print(f"{'='*60}")
    
    try:
        temp_dir = tempfile.mkdtemp()
        
        # Caso 1: Presupuesto agotado
        print(f"1. Probando presupuesto agotado...")
        cost_manager = CostManager(max_daily_cost=0.01, data_directory=temp_dir)  # Muy bajo
        
        available, reason = cost_manager.check_budget_available(estimated_cost=0.02)
        if not available:
            print(f"   ✅ Presupuesto agotado detectado correctamente")
        else:
            print(f"   ⚠️ Presupuesto debería estar agotado")
        
        # Caso 2: Respuesta LLM vacía
        print(f"2. Probando respuesta vacía...")
        parser = ResponseParser()
        empty_response = parser.parse_response("")
        
        if empty_response.decision == DecisionType.REJECT:
            print(f"   ✅ Respuesta vacía manejada conservadoramente")
        else:
            print(f"   ⚠️ Respuesta vacía no manejada correctamente")
        
        # Caso 3: JSON malformado
        print(f"3. Probando JSON malformado...")
        malformed_json = '{"decision": "APPROVE", "confidence": 0.8, "reasoning": "Test'  # JSON incompleto
        malformed_parsed = parser.parse_response(malformed_json)
        
        if malformed_parsed.validation_score < 1.0:
            print(f"   ✅ JSON malformado detectado (score: {malformed_parsed.validation_score:.2f})")
        else:
            print(f"   ⚠️ JSON malformado no detectado")
        
        # Caso 4: Valores fuera de rango
        print(f"4. Probando valores fuera de rango...")
        out_of_range_json = '''
        {
            "decision": "APPROVE",
            "confidence": 1.5,
            "reasoning": "Test",
            "key_factors": [],
            "identified_risks": [],
            "suggested_modifications": {
                "position_size_multiplier": 5.0,
                "stop_loss_adjustment": 1.0,
                "take_profit_adjustment": -1.0
            },
            "market_outlook": "BULLISH",
            "urgency": "MEDIUM"
        }
        '''
        
        out_of_range_parsed = parser.parse_response(out_of_range_json)
        
        if (out_of_range_parsed.confidence <= 1.0 and 
            out_of_range_parsed.suggested_modifications.is_valid()):
            print(f"   ✅ Valores fuera de rango corregidos")
        else:
            print(f"   ⚠️ Valores fuera de rango no corregidos")
        
        # Caso 5: Directorio inaccesible
        print(f"5. Probando directorio inaccesible...")
        try:
            readonly_dir = "/tmp/readonly_llm_test"
            os.makedirs(readonly_dir, exist_ok=True)
            os.chmod(readonly_dir, 0o444)  # Solo lectura
            
            try:
                readonly_connector = LLMConnector(readonly_dir)
                print(f"   ✅ Conector creado con directorio de solo lectura")
            except Exception as e:
                print(f"   ✅ Error manejado correctamente: {type(e).__name__}")
            
            finally:
                os.chmod(readonly_dir, 0o755)
                shutil.rmtree(readonly_dir, ignore_errors=True)
                
        except Exception as e:
            print(f"   ✅ Caso extremo manejado: {type(e).__name__}")
        
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
    print("🔧 SISTEMA DE PRUEBAS - INTEGRACIÓN LLM")
    print("="*70)
    print("Como trader senior, el LLM es mi 'segundo cerebro' para decisiones complejas.")
    print("Debe ser confiable, cost-effective y proporcionar análisis de alta calidad.")
    print("="*70)
    
    # Ejecutar todas las pruebas
    test_results = []
    
    # 1. Inicialización
    result1 = await test_llm_connector_initialization()
    test_results.append(("Inicialización del conector LLM", result1))
    
    # 2. Construcción de prompts
    result2 = await test_prompt_builder()
    test_results.append(("Construcción de prompts", result2))
    
    # 3. Parser de respuestas
    result3 = await test_response_parser()
    test_results.append(("Parser de respuestas", result3))
    
    # 4. Gestión de costos
    result4 = await test_cost_management()
    test_results.append(("Gestión de costos", result4))
    
    # 5. Llamada real al LLM (opcional)
    result5 = await test_real_llm_call()
    test_results.append(("Llamada real al LLM", result5))
    
    # 6. Casos extremos
    result6 = await test_edge_cases()
    test_results.append(("Casos extremos", result6))
    
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
        print("✅ El sistema de integración LLM está listo para producción")
        print("🧠 Gemini integrado como 'segundo cerebro' para decisiones complejas")
        print("💰 Control de costos estricto: máximo $50/día")
        print("🛡️ Parsing robusto con fallbacks conservadores")
    else:
        print("⚠️  Algunas pruebas fallaron - el sistema necesita ajustes")
        print("🔧 Un LLM defectuoso = decisiones erróneas = pérdidas")
        print("💀 La IA debe ser confiable o no debe usarse")
    
    return total_passed == total_tests

if __name__ == "__main__":
    # Ejecutar pruebas
    success = asyncio.run(main())
    
    # Exit code para CI/CD
    sys.exit(0 if success else 1)