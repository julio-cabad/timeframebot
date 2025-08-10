"""
Parser de Respuestas LLM para Trading Algorítmico
================================================

Como trader senior, he aprendido que los LLMs pueden ser inconsistentes.
Este parser asegura que siempre obtengamos respuestas estructuradas y
válidas, sin importar las variaciones en el formato de salida.

Características:
- Parsing robusto de JSON con fallbacks
- Validación estricta de schema
- Limpieza automática de respuestas malformadas
- Extracción inteligente de datos clave
- Logging detallado para debugging

El objetivo: Convertir cualquier respuesta LLM en datos estructurados confiables.

Autor: Trader Algorítmico Senior (10+ años)
Zona Horaria: UTC-5 (Ecuador)
"""

import json
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pytz
from datetime import datetime

from ..utils.logger import get_logger, LogContext
from ..utils.exceptions import TradingBotException

# Zona horaria de Ecuador (UTC-5)
ECUADOR_TZ = pytz.timezone('America/Guayaquil')

class ValidationError(TradingBotException):
    """Error de validación de respuesta"""
    pass

class DecisionType(Enum):
    """Tipos de decisión del LLM"""
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    MODIFY = "MODIFY"

class MarketOutlook(Enum):
    """Perspectiva de mercado"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    UNCERTAIN = "UNCERTAIN"

class UrgencyLevel(Enum):
    """Nivel de urgencia"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

@dataclass
class SuggestedModifications:
    """Modificaciones sugeridas por el LLM"""
    position_size_multiplier: float = 1.0
    stop_loss_adjustment: float = 0.0
    take_profit_adjustment: float = 0.0
    
    def is_valid(self) -> bool:
        """Verifica si las modificaciones están en rangos válidos"""
        return (
            0.1 <= self.position_size_multiplier <= 2.0 and
            -0.5 <= self.stop_loss_adjustment <= 0.5 and
            -0.5 <= self.take_profit_adjustment <= 0.5
        )

@dataclass
class ParsedResponse:
    """Respuesta LLM parseada y validada"""
    decision: DecisionType
    confidence: float
    reasoning: str
    key_factors: List[str]
    identified_risks: List[str]
    suggested_modifications: SuggestedModifications
    market_outlook: MarketOutlook
    urgency: UrgencyLevel
    
    # Metadata
    raw_response: str = ""
    parsing_warnings: List[str] = field(default_factory=list)
    validation_score: float = 1.0  # 0.0 = inválida, 1.0 = perfecta
    
    def is_valid(self) -> bool:
        """Verifica si la respuesta parseada es válida"""
        return (
            0.0 <= self.confidence <= 1.0 and
            len(self.reasoning.strip()) > 10 and
            self.suggested_modifications.is_valid() and
            self.validation_score >= 0.7
        )
    
    def get_summary(self) -> str:
        """Obtiene resumen de la decisión"""
        return f"{self.decision.value} (conf: {self.confidence:.2f}) - {self.reasoning[:100]}..."

class ResponseParser:
    """
    Parser robusto para respuestas LLM
    
    Como trader senior, he diseñado este parser para ser:
    1. Robusto - Maneja cualquier formato de respuesta
    2. Inteligente - Extrae información incluso de respuestas malformadas
    3. Validador - Asegura que los datos sean confiables
    4. Flexible - Se adapta a diferentes estilos de LLM
    """
    
    def __init__(self):
        self.logger = get_logger("ResponseParser")
        
        # Patrones regex para extracción de datos
        self.decision_patterns = [
            r'"decision"\s*:\s*"([^"]+)"',
            r'decision["\']?\s*:\s*["\']?([^"\']+)["\']?',
            r'DECISION:\s*([A-Z]+)',
            r'(APPROVE|REJECT|MODIFY)'
        ]
        
        self.confidence_patterns = [
            r'"confidence"\s*:\s*([0-9.]+)',
            r'confidence["\']?\s*:\s*([0-9.]+)',
            r'CONFIDENCE:\s*([0-9.]+)',
            r'([0-9.]+)\s*confidence'
        ]
        
        self.reasoning_patterns = [
            r'"reasoning"\s*:\s*"([^"]+)"',
            r'reasoning["\']?\s*:\s*["\']([^"\']+)["\']',
            r'REASONING:\s*([^\n]+)',
            r'ANALYSIS:\s*([^\n]+)'
        ]
    
    def parse_response(self, raw_response: str) -> ParsedResponse:
        """
        Parsea una respuesta LLM cruda
        
        Args:
            raw_response: Respuesta cruda del LLM
            
        Returns:
            ParsedResponse parseada y validada
        """
        context = LogContext(component="response_parser")
        
        self.logger.debug("Iniciando parsing de respuesta LLM")
        
        # Limpiar respuesta
        cleaned_response = self._clean_response(raw_response)
        
        # Intentar parsing JSON primero
        parsed_data = self._try_json_parsing(cleaned_response)
        
        if not parsed_data:
            # Fallback a parsing por patrones
            parsed_data = self._pattern_parsing(cleaned_response)
        
        # Validar y construir respuesta
        try:
            response = self._build_parsed_response(parsed_data, raw_response)
            
            self.logger.info(
                f"Respuesta parseada exitosamente: {response.decision.value}",
                context=context,
                extra_fields={
                    "confidence": response.confidence,
                    "validation_score": response.validation_score,
                    "warnings": len(response.parsing_warnings)
                }
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error construyendo respuesta parseada: {e}")
            # Retornar respuesta por defecto conservadora
            return self._create_fallback_response(raw_response, str(e))
    
    def _clean_response(self, response: str) -> str:
        """Limpia la respuesta removiendo caracteres problemáticos"""
        # Remover caracteres de control
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', response)
        
        # Remover markdown code blocks
        cleaned = re.sub(r'```json\s*', '', cleaned)
        cleaned = re.sub(r'```\s*', '', cleaned)
        
        # Remover espacios extra
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _try_json_parsing(self, response: str) -> Optional[Dict[str, Any]]:
        """Intenta parsear como JSON válido"""
        try:
            # Buscar JSON en la respuesta
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            
            # Intentar parsear toda la respuesta
            return json.loads(response)
            
        except json.JSONDecodeError as e:
            self.logger.debug(f"JSON parsing falló: {e}")
            return None
    
    def _pattern_parsing(self, response: str) -> Dict[str, Any]:
        """Parsing usando patrones regex como fallback"""
        parsed_data = {}
        
        # Extraer decisión
        decision = self._extract_with_patterns(response, self.decision_patterns)
        if decision:
            parsed_data["decision"] = decision.upper()
        
        # Extraer confianza
        confidence = self._extract_with_patterns(response, self.confidence_patterns)
        if confidence:
            try:
                conf_value = float(confidence)
                # Convertir a rango 0-1 si está en 0-100
                if conf_value > 1.0:
                    conf_value = conf_value / 100.0
                parsed_data["confidence"] = conf_value
            except ValueError:
                pass
        
        # Extraer reasoning
        reasoning = self._extract_with_patterns(response, self.reasoning_patterns)
        if reasoning:
            parsed_data["reasoning"] = reasoning
        
        # Extraer factores clave (buscar listas)
        key_factors = self._extract_list_items(response, ["key_factors", "factors", "reasons"])
        if key_factors:
            parsed_data["key_factors"] = key_factors
        
        # Extraer riesgos
        risks = self._extract_list_items(response, ["risks", "identified_risks", "warnings"])
        if risks:
            parsed_data["identified_risks"] = risks
        
        # Extraer outlook
        outlook_match = re.search(r'(BULLISH|BEARISH|NEUTRAL|UNCERTAIN)', response, re.IGNORECASE)
        if outlook_match:
            parsed_data["market_outlook"] = outlook_match.group(1).upper()
        
        # Extraer urgencia
        urgency_match = re.search(r'urgency["\']?\s*:\s*["\']?(LOW|MEDIUM|HIGH)["\']?', response, re.IGNORECASE)
        if urgency_match:
            parsed_data["urgency"] = urgency_match.group(1).upper()
        
        return parsed_data
    
    def _extract_with_patterns(self, text: str, patterns: List[str]) -> Optional[str]:
        """Extrae texto usando una lista de patrones"""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return None
    
    def _extract_list_items(self, text: str, field_names: List[str]) -> List[str]:
        """Extrae elementos de lista del texto"""
        items = []
        
        for field_name in field_names:
            # Buscar arrays JSON
            pattern = rf'"{field_name}"\s*:\s*\[(.*?)\]'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    array_content = match.group(1)
                    # Extraer strings del array
                    string_matches = re.findall(r'"([^"]+)"', array_content)
                    items.extend(string_matches)
                except:
                    pass
            
            # Buscar listas con bullets
            bullet_pattern = rf'{field_name}[:\s]*\n?((?:\s*[-*•]\s*[^\n]+\n?)+)'
            match = re.search(bullet_pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                bullet_text = match.group(1)
                bullet_items = re.findall(r'[-*•]\s*([^\n]+)', bullet_text)
                items.extend([item.strip() for item in bullet_items])
        
        return list(set(items))  # Remover duplicados
    
    def _build_parsed_response(self, data: Dict[str, Any], raw_response: str) -> ParsedResponse:
        """Construye ParsedResponse desde datos extraídos"""
        warnings = []
        validation_score = 1.0
        
        # Parsear decisión
        try:
            decision_str = data.get("decision", "REJECT").upper()
            decision = DecisionType(decision_str)
        except ValueError:
            decision = DecisionType.REJECT
            warnings.append(f"Decisión inválida: {data.get('decision')}")
            validation_score -= 0.2
        
        # Parsear confianza
        confidence = data.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            confidence = 0.5
            warnings.append(f"Confianza inválida: {data.get('confidence')}")
            validation_score -= 0.1
        
        # Parsear reasoning
        reasoning = data.get("reasoning", "No reasoning provided")
        if len(reasoning.strip()) < 10:
            reasoning = "Análisis insuficiente proporcionado"
            warnings.append("Reasoning muy corto")
            validation_score -= 0.2
        
        # Parsear factores clave
        key_factors = data.get("key_factors", [])
        if not isinstance(key_factors, list):
            key_factors = []
            warnings.append("Key factors no es una lista")
            validation_score -= 0.1
        
        # Parsear riesgos
        identified_risks = data.get("identified_risks", [])
        if not isinstance(identified_risks, list):
            identified_risks = []
            warnings.append("Identified risks no es una lista")
            validation_score -= 0.1
        
        # Parsear modificaciones sugeridas
        modifications_data = data.get("suggested_modifications", {})
        suggested_modifications = SuggestedModifications(
            position_size_multiplier=modifications_data.get("position_size_multiplier", 1.0),
            stop_loss_adjustment=modifications_data.get("stop_loss_adjustment", 0.0),
            take_profit_adjustment=modifications_data.get("take_profit_adjustment", 0.0)
        )
        
        if not suggested_modifications.is_valid():
            suggested_modifications = SuggestedModifications()
            warnings.append("Modificaciones sugeridas fuera de rango")
            validation_score -= 0.1
        
        # Parsear market outlook
        try:
            outlook_str = data.get("market_outlook", "UNCERTAIN").upper()
            market_outlook = MarketOutlook(outlook_str)
        except ValueError:
            market_outlook = MarketOutlook.UNCERTAIN
            warnings.append(f"Market outlook inválido: {data.get('market_outlook')}")
            validation_score -= 0.1
        
        # Parsear urgencia
        try:
            urgency_str = data.get("urgency", "MEDIUM").upper()
            urgency = UrgencyLevel(urgency_str)
        except ValueError:
            urgency = UrgencyLevel.MEDIUM
            warnings.append(f"Urgency inválida: {data.get('urgency')}")
            validation_score -= 0.1
        
        return ParsedResponse(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=key_factors,
            identified_risks=identified_risks,
            suggested_modifications=suggested_modifications,
            market_outlook=market_outlook,
            urgency=urgency,
            raw_response=raw_response,
            parsing_warnings=warnings,
            validation_score=max(0.0, validation_score)
        )
    
    def _create_fallback_response(self, raw_response: str, error: str) -> ParsedResponse:
        """Crea respuesta fallback conservadora cuando el parsing falla completamente"""
        self.logger.warning(f"Creando respuesta fallback: {error}")
        
        return ParsedResponse(
            decision=DecisionType.REJECT,  # Conservador por defecto
            confidence=0.0,
            reasoning=f"Error parsing respuesta LLM: {error}",
            key_factors=["Parsing failed"],
            identified_risks=["Unable to parse LLM response"],
            suggested_modifications=SuggestedModifications(),
            market_outlook=MarketOutlook.UNCERTAIN,
            urgency=UrgencyLevel.LOW,
            raw_response=raw_response,
            parsing_warnings=[f"Parsing completamente fallido: {error}"],
            validation_score=0.0
        )
    
    def validate_response_quality(self, response: ParsedResponse) -> Tuple[bool, List[str]]:
        """
        Valida la calidad de una respuesta parseada
        
        Args:
            response: Respuesta a validar
            
        Returns:
            Tuple de (es_válida, lista_de_problemas)
        """
        problems = []
        
        # Verificar campos obligatorios
        if not response.reasoning or len(response.reasoning.strip()) < 20:
            problems.append("Reasoning insuficiente (< 20 caracteres)")
        
        if response.confidence < 0.1:
            problems.append("Confianza muy baja (< 0.1)")
        
        if not response.key_factors:
            problems.append("No se proporcionaron factores clave")
        
        if response.decision == DecisionType.APPROVE and not response.identified_risks:
            problems.append("Aprobación sin identificar riesgos")
        
        if response.validation_score < 0.7:
            problems.append(f"Score de validación bajo: {response.validation_score:.2f}")
        
        # Verificar consistencia
        if response.decision == DecisionType.APPROVE and response.confidence < 0.6:
            problems.append("Aprobación con baja confianza")
        
        if response.decision == DecisionType.REJECT and response.confidence > 0.8:
            problems.append("Rechazo con alta confianza (posible inconsistencia)")
        
        is_valid = len(problems) == 0
        
        return is_valid, problems
    
    def extract_key_metrics(self, response: ParsedResponse) -> Dict[str, Any]:
        """Extrae métricas clave de la respuesta para análisis"""
        return {
            "decision": response.decision.value,
            "confidence": response.confidence,
            "validation_score": response.validation_score,
            "reasoning_length": len(response.reasoning),
            "key_factors_count": len(response.key_factors),
            "risks_count": len(response.identified_risks),
            "warnings_count": len(response.parsing_warnings),
            "market_outlook": response.market_outlook.value,
            "urgency": response.urgency.value,
            "has_modifications": (
                response.suggested_modifications.position_size_multiplier != 1.0 or
                response.suggested_modifications.stop_loss_adjustment != 0.0 or
                response.suggested_modifications.take_profit_adjustment != 0.0
            )
        }