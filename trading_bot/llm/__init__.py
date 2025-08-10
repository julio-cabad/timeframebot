"""
Sistema de Integración LLM para Trading Algorítmico
==================================================

Como trader senior con más de 10 años de experiencia, he aprendido que hay
situaciones donde el análisis técnico puro no es suficiente. Los LLMs pueden
analizar contexto complejo, detectar patrones sutiles y tomar decisiones
inteligentes en zonas ambiguas.

Este módulo implementa:
- Conectores escalables para múltiples proveedores LLM
- Sistema de prompts estructurados para análisis de trading
- Control de costos inteligente con presupuesto diario
- Parsing robusto de respuestas JSON
- Caché inteligente para reducir costos
- Fallback automático entre proveedores

Proveedores soportados:
- Google Gemini (recomendado para trading)
- OpenAI GPT-4
- Anthropic Claude
- Extensible para otros proveedores

Autor: Trader Algorítmico Senior (10+ años)
Zona Horaria: UTC-5 (Ecuador)
"""

from .connector import (
    LLMConnector,
    LLMProvider,
    LLMResponse,
    LLMError,
    create_llm_connector
)

from .prompt_builder import (
    PromptBuilder,
    TradeAnalysisPrompt,
    MarketContextPrompt,
    RiskAssessmentPrompt
)

from .cost_manager import (
    CostManager,
    CostTracker,
    CostCategory,
    BudgetExceededError
)

from .response_parser import (
    ResponseParser,
    ParsedResponse,
    DecisionType,
    ValidationError
)

__all__ = [
    'LLMConnector',
    'LLMProvider', 
    'LLMResponse',
    'LLMError',
    'create_llm_connector',
    'PromptBuilder',
    'TradeAnalysisPrompt',
    'MarketContextPrompt', 
    'RiskAssessmentPrompt',
    'CostManager',
    'CostTracker',
    'CostCategory',
    'BudgetExceededError',
    'ResponseParser',
    'ParsedResponse',
    'DecisionType',
    'ValidationError'
]