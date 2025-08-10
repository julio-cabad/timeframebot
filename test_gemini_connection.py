#!/usr/bin/env python3
"""
Test rápido para verificar conexión con Gemini
"""

import asyncio
import os
from trading_bot.llm import create_llm_connector, LLMProvider

async def test_gemini_connection():
    """Prueba conexión real con Gemini"""
    
    # Verificar que existe la API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY no encontrada en variables de entorno")
        print("💡 Configura tu API key:")
        print("   export GEMINI_API_KEY='tu_api_key_aqui'")
        return False
    
    print(f"✅ API Key encontrada: {api_key[:10]}...")
    
    try:
        # Crear conector
        connector = create_llm_connector(
            primary_provider=LLMProvider.GEMINI,
            primary_api_key=api_key
        )
        
        print("✅ Conector LLM creado exitosamente")
        
        # Prompt simple de prueba
        test_prompt = """
        Analiza esta oportunidad de trading:
        - Símbolo: BTCUSDT
        - Precio: $50,000
        - Tendencia: Alcista
        
        Responde en formato JSON:
        {
            "decision": "APPROVE",
            "confidence": 0.75,
            "reasoning": "Análisis técnico favorable",
            "identified_risks": ["Volatilidad del mercado"]
        }
        """
        
        print("🔄 Enviando prompt de prueba a Gemini...")
        
        # Llamada real a Gemini
        response = await connector.generate_response(test_prompt)
        
        print("✅ Respuesta recibida de Gemini:")
        print(f"   - Proveedor: {response.provider.value}")
        print(f"   - Modelo: {response.model}")
        print(f"   - Tokens: {response.total_tokens}")
        print(f"   - Costo: ${response.cost_usd:.4f}")
        print(f"   - Tiempo: {response.response_time_ms}ms")
        print(f"   - Decisión: {response.decision}")
        print(f"   - Confianza: {response.confidence}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error conectando con Gemini: {str(e)}")
        print("💡 Verifica que tu API key sea válida")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_gemini_connection())
    if success:
        print("\n🎉 ¡Conexión con Gemini exitosa!")
        print("✅ El sistema LLM está listo para usar")
    else:
        print("\n⚠️ Configura tu API key de Gemini para continuar")