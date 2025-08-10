"""
Sistema de Construcción de Prompts para Trading Algorítmico
==========================================================

Como trader senior, he diseñado estos prompts para obtener análisis de alta
calidad del LLM. Cada prompt está optimizado para:
- Obtener respuestas estructuradas y consistentes
- Minimizar alucinaciones con contexto específico
- Maximizar valor analítico para decisiones de trading
- Mantener costos bajos con prompts concisos pero efectivos

Los prompts están diseñados para situaciones específicas:
- Análisis de trades en zona ambigua (score 60-75%)
- Validación de divergencias críticas
- Análisis de contexto macro complejo
- Evaluación de riesgo en situaciones inusuales

Autor: Trader Algorítmico Senior (10+ años)
Zona Horaria: UTC-5 (Ecuador)
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import pytz

from ..scoring.scorer import ScoringResult, ScoreComponent
from ..analysis.tf_analyzers import TimeframeAnalysis
from ..analysis.confluence import ConfluenceResult
from ..analysis.patterns import DetectedPattern

# Zona horaria de Ecuador (UTC-5)
ECUADOR_TZ = pytz.timezone('America/Guayaquil')

class PromptType(Enum):
    """Tipos de prompts disponibles"""
    TRADE_ANALYSIS = "trade_analysis"
    DIVERGENCE_CHECK = "divergence_check"
    MACRO_CONTEXT = "macro_context"
    RISK_ASSESSMENT = "risk_assessment"
    PATTERN_VALIDATION = "pattern_validation"

@dataclass
class PromptContext:
    """Contexto para construcción de prompts"""
    symbol: str
    timeframe: str
    current_price: float
    scoring_result: ScoringResult
    timeframe_analyses: Dict[str, TimeframeAnalysis]
    confluence_result: ConfluenceResult
    patterns: List[DetectedPattern]
    market_context: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(ECUADOR_TZ))

class PromptBuilder:
    """
    Constructor de prompts estructurados para análisis de trading
    
    Como trader senior, he optimizado estos prompts para:
    1. Obtener análisis objetivo y cuantificable
    2. Minimizar sesgos y alucinaciones
    3. Maximizar valor analítico por token
    4. Mantener consistencia en las respuestas
    """
    
    def __init__(self):
        self.base_instructions = self._get_base_instructions()
        self.response_schema = self._get_response_schema()
    
    def _get_base_instructions(self) -> str:
        """Instrucciones base para todos los prompts"""
        return """Eres un analista de trading algorítmico senior con más de 10 años de experiencia en mercados de criptomonedas. Tu objetivo es proporcionar análisis objetivo, cuantificable y accionable.

PRINCIPIOS CLAVE:
1. Sé objetivo y basado en datos - no especules
2. Considera TODOS los timeframes y su confluencia
3. Evalúa riesgo/recompensa de manera conservadora
4. Identifica divergencias y señales contradictorias
5. Proporciona reasoning claro y específico
6. Mantén respuestas concisas pero completas

RESPONDE SIEMPRE EN FORMATO JSON VÁLIDO con la estructura exacta solicitada."""
    
    def _get_response_schema(self) -> Dict[str, Any]:
        """Schema de respuesta estándar"""
        return {
            "decision": "string (APPROVE/REJECT/MODIFY)",
            "confidence": "float (0.0-1.0)",
            "reasoning": "string (máximo 200 palabras)",
            "key_factors": ["array de strings con factores clave"],
            "identified_risks": ["array de strings con riesgos identificados"],
            "suggested_modifications": {
                "position_size_multiplier": "float (0.5-1.5)",
                "stop_loss_adjustment": "float (-0.2 to 0.2)",
                "take_profit_adjustment": "float (-0.2 to 0.2)"
            },
            "market_outlook": "string (BULLISH/BEARISH/NEUTRAL/UNCERTAIN)",
            "urgency": "string (LOW/MEDIUM/HIGH)"
        }
    
    def build_trade_analysis_prompt(self, context: PromptContext) -> str:
        """
        Construye prompt para análisis completo de trade
        
        Usado cuando el score está en zona ambigua (60-75%)
        """
        prompt_parts = [
            self.base_instructions,
            "",
            "ANÁLISIS DE OPORTUNIDAD DE TRADING",
            "=" * 40,
            "",
            f"SÍMBOLO: {context.symbol}",
            f"PRECIO ACTUAL: ${context.current_price:,.2f}",
            f"TIMESTAMP: {context.timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            "",
            "SCORING ACTUAL:",
            f"- Score Final: {context.scoring_result.final_score:.1f}/100",
            f"- Recomendación: {context.scoring_result.trade_recommendation}",
            f"- Confianza: {context.scoring_result.confidence_level}",
            f"- Régimen: {context.scoring_result.regime.value}",
            "",
            "BREAKDOWN POR COMPONENTE:"
        ]
        
        # Agregar breakdown de scoring
        for component, score in context.scoring_result.breakdown.component_scores.items():
            prompt_parts.append(f"- {component.value}: {score.weighted_score:.1f} (peso: {score.weight:.2f})")
        
        prompt_parts.extend([
            "",
            "ANÁLISIS MULTI-TIMEFRAME:"
        ])
        
        # Agregar análisis por timeframe
        for tf, analysis in context.timeframe_analyses.items():
            prompt_parts.extend([
                f"",
                f"{tf.upper()}:",
                f"- Tendencia: {analysis.trend_direction} (fuerza: {analysis.trend_strength:.2f})",
                f"- Momentum: {analysis.momentum_score:.2f}",
                f"- Confianza: {analysis.confidence:.2f}"
            ])
            
            if analysis.patterns:
                prompt_parts.append(f"- Patrones: {[p.value for p in analysis.patterns]}")
        
        # Agregar confluencia
        prompt_parts.extend([
            "",
            "CONFLUENCIA:",
            f"- Score General: {context.confluence_result.overall_confluence_score:.1f}",
            f"- Fuerza: {context.confluence_result.confluence_strength.value}",
            f"- Confianza: {context.confluence_result.confidence:.2f}"
        ])
        
        # Agregar patrones detectados
        if context.patterns:
            prompt_parts.extend([
                "",
                "PATRONES DETECTADOS:"
            ])
            for pattern in context.patterns[:3]:  # Máximo 3 patrones
                prompt_parts.append(f"- {pattern.pattern_type}: {pattern.confidence:.2f} confianza")
        
        # Agregar contexto de mercado
        prompt_parts.extend([
            "",
            "CONTEXTO DE MERCADO:",
            f"- Volatilidad: {context.market_context.get('volatility', 0):.3f}",
            f"- Liquidez: {context.market_context.get('liquidity_score', 0):.2f}",
            f"- Riesgo Correlación: {context.market_context.get('correlation_risk', 0):.2f}",
            f"- Horas Mercado: {context.market_context.get('market_hours', 'unknown')}"
        ])
        
        # Instrucciones específicas
        prompt_parts.extend([
            "",
            "INSTRUCCIONES ESPECÍFICAS:",
            "1. El score está en zona AMBIGUA (60-75%) - requiere análisis profundo",
            "2. Evalúa si hay divergencias entre timeframes que justifiquen precaución",
            "3. Considera el contexto macro y condiciones de mercado actuales",
            "4. Si apruebas, sugiere ajustes para optimizar riesgo/recompensa",
            "5. Si rechazas, explica claramente los factores de riesgo",
            "",
            f"RESPONDE EN JSON con esta estructura exacta:",
            json.dumps(self.response_schema, indent=2)
        ])
        
        return "\n".join(prompt_parts)
    
    def build_divergence_check_prompt(self, context: PromptContext, 
                                    divergences: List[str]) -> str:
        """
        Construye prompt para validación de divergencias críticas
        
        Usado cuando se detectan divergencias importantes entre timeframes
        """
        prompt_parts = [
            self.base_instructions,
            "",
            "VALIDACIÓN DE DIVERGENCIAS CRÍTICAS",
            "=" * 40,
            "",
            f"SÍMBOLO: {context.symbol}",
            f"PRECIO ACTUAL: ${context.current_price:,.2f}",
            "",
            "DIVERGENCIAS DETECTADAS:"
        ]
        
        for i, divergence in enumerate(divergences, 1):
            prompt_parts.append(f"{i}. {divergence}")
        
        prompt_parts.extend([
            "",
            "ANÁLISIS MULTI-TIMEFRAME ACTUAL:"
        ])
        
        # Análisis condensado por timeframe
        for tf, analysis in context.timeframe_analyses.items():
            direction_emoji = "🟢" if analysis.trend_direction == "bullish" else "🔴" if analysis.trend_direction == "bearish" else "🟡"
            prompt_parts.append(
                f"- {tf}: {direction_emoji} {analysis.trend_direction} "
                f"(fuerza: {analysis.trend_strength:.2f}, momentum: {analysis.momentum_score:.2f})"
            )
        
        prompt_parts.extend([
            "",
            "PREGUNTAS CLAVE:",
            "1. ¿Son estas divergencias señales de reversión inminente?",
            "2. ¿Qué timeframe tiene más peso en la situación actual?",
            "3. ¿Las divergencias invalidan la oportunidad de trading?",
            "4. ¿Hay forma de ajustar la estrategia para manejar las divergencias?",
            "",
            "INSTRUCCIONES:",
            "- Evalúa la severidad de cada divergencia",
            "- Determina si las divergencias son temporales o estructurales",
            "- Recomienda REJECT si las divergencias son demasiado riesgosas",
            "- Si es APPROVE/MODIFY, sugiere ajustes específicos",
            "",
            f"RESPONDE EN JSON:",
            json.dumps(self.response_schema, indent=2)
        ])
        
        return "\n".join(prompt_parts)
    
    def build_macro_context_prompt(self, context: PromptContext,
                                 macro_events: List[Dict[str, Any]]) -> str:
        """
        Construye prompt para análisis de contexto macro
        
        Usado cuando hay eventos macro importantes en las próximas 4 horas
        """
        prompt_parts = [
            self.base_instructions,
            "",
            "ANÁLISIS DE CONTEXTO MACRO",
            "=" * 40,
            "",
            f"SÍMBOLO: {context.symbol}",
            f"PRECIO ACTUAL: ${context.current_price:,.2f}",
            f"SCORE TÉCNICO: {context.scoring_result.final_score:.1f}/100",
            "",
            "EVENTOS MACRO PRÓXIMOS:"
        ]
        
        for event in macro_events:
            prompt_parts.extend([
                f"",
                f"EVENTO: {event.get('title', 'Unknown')}",
                f"- Tiempo: {event.get('time', 'Unknown')}",
                f"- Impacto: {event.get('impact', 'Unknown')}",
                f"- Descripción: {event.get('description', 'N/A')}"
            ])
        
        prompt_parts.extend([
            "",
            "ANÁLISIS TÉCNICO ACTUAL:",
            f"- Tendencia Principal: {context.timeframe_analyses.get('1d').trend_direction if '1d' in context.timeframe_analyses else 'Unknown'}",
            f"- Momentum 4H: {context.timeframe_analyses.get('4h').momentum_score:.2f if '4h' in context.timeframe_analyses else 'N/A'}",
            f"- Setup 1H: {context.timeframe_analyses.get('1h').confidence:.2f if '1h' in context.timeframe_analyses else 'N/A'}",
            "",
            "PREGUNTAS CLAVE:",
            "1. ¿Los eventos macro pueden invalidar el análisis técnico?",
            "2. ¿Es mejor esperar después de los eventos para operar?",
            "3. ¿El timing del trade coincide con volatilidad esperada?",
            "4. ¿Hay que ajustar position size por el riesgo macro?",
            "",
            "INSTRUCCIONES:",
            "- Evalúa la probabilidad de volatilidad extrema",
            "- Considera si el análisis técnico es confiable con eventos macro",
            "- Recomienda timing óptimo para la entrada",
            "- Sugiere ajustes de riesgo específicos",
            "",
            f"RESPONDE EN JSON:",
            json.dumps(self.response_schema, indent=2)
        ])
        
        return "\n".join(prompt_parts)
    
    def build_risk_assessment_prompt(self, context: PromptContext,
                                   risk_factors: List[str]) -> str:
        """
        Construye prompt para evaluación de riesgo específica
        
        Usado cuando se detectan factores de riesgo inusuales
        """
        prompt_parts = [
            self.base_instructions,
            "",
            "EVALUACIÓN DE RIESGO ESPECÍFICA",
            "=" * 40,
            "",
            f"SÍMBOLO: {context.symbol}",
            f"PRECIO ACTUAL: ${context.current_price:,.2f}",
            "",
            "FACTORES DE RIESGO IDENTIFICADOS:"
        ]
        
        for i, risk in enumerate(risk_factors, 1):
            prompt_parts.append(f"{i}. {risk}")
        
        prompt_parts.extend([
            "",
            "MÉTRICAS DE RIESGO ACTUALES:",
            f"- Volatilidad: {context.market_context.get('volatility', 0):.3f}",
            f"- Drawdown Risk: {context.market_context.get('drawdown_risk', 0):.2f}",
            f"- Position Risk: {context.market_context.get('position_risk', 0):.2f}",
            f"- Correlation Risk: {context.market_context.get('correlation_risk', 0):.2f}",
            "",
            "SETUP TÉCNICO:",
            f"- Score: {context.scoring_result.final_score:.1f}/100",
            f"- Win Rate Esperado: {context.scoring_result.expected_win_rate:.1%}",
            f"- Profit Factor Esperado: {context.scoring_result.expected_profit_factor:.2f}",
            "",
            "PREGUNTAS CLAVE:",
            "1. ¿Los factores de riesgo superan el potencial de ganancia?",
            "2. ¿Hay forma de mitigar los riesgos identificados?",
            "3. ¿El timing es apropiado considerando los riesgos?",
            "4. ¿Qué ajustes específicos minimizarían el riesgo?",
            "",
            "INSTRUCCIONES:",
            "- Evalúa cada factor de riesgo individualmente",
            "- Considera el riesgo combinado de todos los factores",
            "- Sugiere mitigaciones específicas si es posible",
            "- Recomienda REJECT si el riesgo es inaceptable",
            "",
            f"RESPONDE EN JSON:",
            json.dumps(self.response_schema, indent=2)
        ])
        
        return "\n".join(prompt_parts)
    
    def build_pattern_validation_prompt(self, context: PromptContext,
                                      suspicious_patterns: List[DetectedPattern]) -> str:
        """
        Construye prompt para validación de patrones sospechosos
        
        Usado cuando los patrones detectados tienen baja confianza o son contradictorios
        """
        prompt_parts = [
            self.base_instructions,
            "",
            "VALIDACIÓN DE PATRONES TÉCNICOS",
            "=" * 40,
            "",
            f"SÍMBOLO: {context.symbol}",
            f"PRECIO ACTUAL: ${context.current_price:,.2f}",
            "",
            "PATRONES DETECTADOS (REQUIEREN VALIDACIÓN):"
        ]
        
        for i, pattern in enumerate(suspicious_patterns, 1):
            target_str = f"${pattern.targets[0].price:.2f}" if pattern.targets else "N/A"
            prompt_parts.extend([
                f"",
                f"PATRÓN {i}: {pattern.pattern_type}",
                f"- Confianza: {pattern.confidence:.2f}",
                f"- Timeframe: {pattern.timeframe}",
                f"- Categoría: {pattern.category.value}",
                f"- Target: {target_str}",
                f"- Stop: ${pattern.stop_loss:.2f}"
            ])
        
        prompt_parts.extend([
            "",
            "CONTEXTO MULTI-TIMEFRAME:",
            f"- Tendencia 1D: {context.timeframe_analyses.get('1d').trend_direction if '1d' in context.timeframe_analyses else 'Unknown'}",
            f"- Momentum 4H: {context.timeframe_analyses.get('4h').momentum_score:.2f if '4h' in context.timeframe_analyses else 'N/A'}",
            f"- Setup 1H: {[p.value for p in context.timeframe_analyses.get('1h').patterns] if '1h' in context.timeframe_analyses else 'None'}",
            "",
            "PREGUNTAS CLAVE:",
            "1. ¿Los patrones son válidos en el contexto actual?",
            "2. ¿Hay confluencia entre patrones de diferentes timeframes?",
            "3. ¿Los patrones contradicen el análisis de tendencia?",
            "4. ¿Qué patrón tiene más probabilidad de cumplirse?",
            "",
            "INSTRUCCIONES:",
            "- Evalúa la validez de cada patrón individualmente",
            "- Considera la confluencia entre patrones",
            "- Identifica el patrón más confiable si hay múltiples",
            "- Recomienda ignorar patrones de baja calidad",
            "",
            f"RESPONDE EN JSON:",
            json.dumps(self.response_schema, indent=2)
        ])
        
        return "\n".join(prompt_parts)

# Clases de conveniencia para tipos específicos de prompts
@dataclass
class TradeAnalysisPrompt:
    """Prompt específico para análisis de trades"""
    context: PromptContext
    
    def build(self) -> str:
        builder = PromptBuilder()
        return builder.build_trade_analysis_prompt(self.context)

@dataclass 
class MarketContextPrompt:
    """Prompt específico para contexto de mercado"""
    context: PromptContext
    macro_events: List[Dict[str, Any]]
    
    def build(self) -> str:
        builder = PromptBuilder()
        return builder.build_macro_context_prompt(self.context, self.macro_events)

@dataclass
class RiskAssessmentPrompt:
    """Prompt específico para evaluación de riesgo"""
    context: PromptContext
    risk_factors: List[str]
    
    def build(self) -> str:
        builder = PromptBuilder()
        return builder.build_risk_assessment_prompt(self.context, self.risk_factors)