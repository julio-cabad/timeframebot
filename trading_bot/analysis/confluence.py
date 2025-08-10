"""
Motor de Confluencia Multi-Timeframe
===================================

Este módulo implementa el sistema de confluencia que combina análisis de múltiples timeframes
para identificar oportunidades de alta probabilidad. La confluencia es el corazón del sistema:
cuando múltiples timeframes se alinean, la probabilidad de éxito aumenta significativamente.

El sistema evalúa:
- Alineación de tendencias entre timeframes
- Confluencia de niveles de soporte/resistencia
- Sincronización de momentum
- Detección de divergencias críticas
- Scoring ponderado por importancia de timeframe

Filosofía: "Una señal en un timeframe es información, múltiples timeframes alineados es convicción"

Autor: Sistema de Trading Algorítmico
Zona Horaria: UTC-5 (Ecuador)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pytz

from .tf_analyzers import TimeframeAnalysis, MarketStructure, PatternType
from ..config.settings import config
from ..utils.logger import get_logger, LogContext
from ..utils.exceptions import (
    AnalysisException,
    ErrorCodes,
    create_exception
)

# Zona horaria de Ecuador (UTC-5)
ECUADOR_TZ = pytz.timezone('America/Guayaquil')

class ConfluenceStrength(Enum):
    """Niveles de fuerza de confluencia"""
    WEAK = "weak"           # 1-2 timeframes alineados
    MODERATE = "moderate"   # 3 timeframes alineados
    STRONG = "strong"       # 4+ timeframes alineados
    CRITICAL = "critical"   # Confluencia excepcional con múltiples factores

class AlignmentType(Enum):
    """Tipos de alineación entre timeframes"""
    TREND_ALIGNMENT = "trend_alignment"         # Tendencias en la misma dirección
    MOMENTUM_SYNC = "momentum_sync"             # Momentum sincronizado
    STRUCTURE_CONFLUENCE = "structure_confluence"  # Estructuras alineadas
    LEVEL_CONFLUENCE = "level_confluence"       # Niveles coincidentes
    PATTERN_CONFLUENCE = "pattern_confluence"   # Patrones complementarios

class DivergenceType(Enum):
    """Tipos de divergencias detectadas"""
    TREND_DIVERGENCE = "trend_divergence"       # Tendencias opuestas
    MOMENTUM_DIVERGENCE = "momentum_divergence" # Momentum divergente
    STRUCTURE_DIVERGENCE = "structure_divergence" # Estructuras conflictivas
    CRITICAL_DIVERGENCE = "critical_divergence"  # Divergencia crítica

@dataclass
class ConfluenceLevel:
    """Representa un nivel de confluencia entre timeframes"""
    price_level: float
    timeframes: List[str]
    level_type: str  # "support" o "resistance"
    strength: float  # 0.0 a 1.0
    distance_from_current: float  # Distancia del precio actual

@dataclass
class Divergence:
    """Representa una divergencia detectada"""
    divergence_type: DivergenceType
    timeframes_involved: List[str]
    severity: float  # 0.0 a 1.0
    description: str
    impact_on_confidence: float  # Impacto negativo en confianza

@dataclass
class AlignmentScore:
    """Score de alineación para un tipo específico"""
    alignment_type: AlignmentType
    score: float  # 0.0 a 1.0
    participating_timeframes: List[str]
    weight: float  # Peso en el score final

@dataclass
class ConfluenceResult:
    """Resultado completo del análisis de confluencia"""
    timestamp: datetime
    symbol: str
    
    # Scores principales
    overall_confluence_score: float  # 0.0 a 100.0
    confluence_strength: ConfluenceStrength
    confidence: float  # 0.0 a 1.0
    
    # Alineaciones detectadas
    alignment_scores: Dict[AlignmentType, AlignmentScore]
    
    # Niveles de confluencia
    confluence_levels: List[ConfluenceLevel]
    
    # Divergencias detectadas
    divergences: List[Divergence]
    
    # Recomendación final
    trade_recommendation: str  # "STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"
    recommended_timeframe: str  # Mejor timeframe para entrada
    
    # Contexto adicional
    participating_timeframes: List[str]
    dominant_trend: str  # "bullish", "bearish", "neutral"
    risk_level: str  # "low", "medium", "high"

class ConfluenceEngine:
    """
    Motor principal de confluencia multi-timeframe
    
    Combina análisis de múltiples timeframes para identificar oportunidades
    de alta probabilidad basadas en alineación y confluencia.
    """
    
    def __init__(self):
        self.logger = get_logger("ConfluenceEngine")
        
        # Pesos por timeframe (más peso = mayor importancia)
        self.timeframe_weights = {
            '1d': 0.35,   # Mayor peso - tendencia principal
            '4h': 0.30,   # Alto peso - momentum intermedio
            '1h': 0.25,   # Peso medio - setups de entrada
            '15m': 0.10   # Menor peso - timing preciso
        }
        
        # Umbrales de confluencia
        self.confluence_thresholds = {
            ConfluenceStrength.WEAK: 40.0,
            ConfluenceStrength.MODERATE: 60.0,
            ConfluenceStrength.STRONG: 80.0,
            ConfluenceStrength.CRITICAL: 90.0
        }
        
        # Tolerancia para niveles de confluencia (porcentaje)
        self.level_tolerance = 0.005  # 0.5%
    
    def calculate_confluence(self, analyses: Dict[str, TimeframeAnalysis]) -> ConfluenceResult:
        """
        Calcula confluencia completa entre múltiples análisis de timeframe
        
        Args:
            analyses: Diccionario con análisis por timeframe {timeframe: TimeframeAnalysis}
            
        Returns:
            ConfluenceResult con análisis completo de confluencia
        """
        context = LogContext(
            component="confluence_engine",
            symbol=analyses[list(analyses.keys())[0]].timeframe if analyses else "unknown"
        )
        
        if not analyses:
            raise AnalysisException(
                "No hay análisis disponibles para calcular confluencia",
                ErrorCodes.ANALYSIS_INSUFFICIENT_DATA
            )
        
        symbol = self._extract_symbol_from_analyses(analyses)
        
        self.logger.debug(
            f"Calculando confluencia para {symbol} con {len(analyses)} timeframes",
            context=context
        )
        
        try:
            # 1. Calcular alineaciones por tipo
            alignment_scores = self._calculate_alignment_scores(analyses)
            
            # 2. Detectar niveles de confluencia
            confluence_levels = self._detect_confluence_levels(analyses)
            
            # 3. Detectar divergencias
            divergences = self._detect_divergences(analyses)
            
            # 4. Calcular score general de confluencia
            overall_score = self._calculate_overall_confluence_score(
                alignment_scores, confluence_levels, divergences
            )
            
            # 5. Determinar fuerza de confluencia
            confluence_strength = self._determine_confluence_strength(overall_score)
            
            # 6. Calcular confianza
            confidence = self._calculate_confluence_confidence(
                alignment_scores, divergences, len(analyses)
            )
            
            # 7. Generar recomendación
            trade_recommendation, recommended_timeframe = self._generate_trade_recommendation(
                analyses, alignment_scores, overall_score
            )
            
            # 8. Determinar tendencia dominante y riesgo
            dominant_trend = self._determine_dominant_trend(analyses)
            risk_level = self._assess_risk_level(divergences, confidence)
            
            result = ConfluenceResult(
                timestamp=datetime.now(ECUADOR_TZ),
                symbol=symbol,
                overall_confluence_score=overall_score,
                confluence_strength=confluence_strength,
                confidence=confidence,
                alignment_scores=alignment_scores,
                confluence_levels=confluence_levels,
                divergences=divergences,
                trade_recommendation=trade_recommendation,
                recommended_timeframe=recommended_timeframe,
                participating_timeframes=list(analyses.keys()),
                dominant_trend=dominant_trend,
                risk_level=risk_level
            )
            
            self.logger.info(
                f"Confluencia calculada: {confluence_strength.value} "
                f"(score: {overall_score:.1f}, confianza: {confidence:.3f})",
                context=context,
                extra_fields={
                    "confluence_score": overall_score,
                    "confluence_strength": confluence_strength.value,
                    "confidence": confidence,
                    "recommendation": trade_recommendation,
                    "timeframes_count": len(analyses)
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculando confluencia: {str(e)}", context=context)
            raise AnalysisException(
                f"Fallo en cálculo de confluencia: {str(e)}",
                ErrorCodes.ANALYSIS_CALCULATION_FAILED
            )
    
    def _calculate_alignment_scores(self, analyses: Dict[str, TimeframeAnalysis]) -> Dict[AlignmentType, AlignmentScore]:
        """Calcula scores de alineación por tipo"""
        alignment_scores = {}
        
        # 1. Alineación de tendencias
        trend_score = self._calculate_trend_alignment(analyses)
        alignment_scores[AlignmentType.TREND_ALIGNMENT] = trend_score
        
        # 2. Sincronización de momentum
        momentum_score = self._calculate_momentum_sync(analyses)
        alignment_scores[AlignmentType.MOMENTUM_SYNC] = momentum_score
        
        # 3. Confluencia de estructura
        structure_score = self._calculate_structure_confluence(analyses)
        alignment_scores[AlignmentType.STRUCTURE_CONFLUENCE] = structure_score
        
        # 4. Confluencia de niveles
        level_score = self._calculate_level_confluence(analyses)
        alignment_scores[AlignmentType.LEVEL_CONFLUENCE] = level_score
        
        # 5. Confluencia de patrones
        pattern_score = self._calculate_pattern_confluence(analyses)
        alignment_scores[AlignmentType.PATTERN_CONFLUENCE] = pattern_score
        
        return alignment_scores
    
    def _calculate_trend_alignment(self, analyses: Dict[str, TimeframeAnalysis]) -> AlignmentScore:
        """Calcula alineación de tendencias"""
        trend_directions = {}
        trend_strengths = {}
        
        for tf, analysis in analyses.items():
            trend_directions[tf] = analysis.trend_direction
            trend_strengths[tf] = analysis.trend_strength
        
        # Contar tendencias por dirección
        bullish_count = sum(1 for direction in trend_directions.values() if direction == "bullish")
        bearish_count = sum(1 for direction in trend_directions.values() if direction == "bearish")
        neutral_count = sum(1 for direction in trend_directions.values() if direction == "neutral")
        
        total_timeframes = len(analyses)
        
        # Calcular score basado en alineación
        if bullish_count >= total_timeframes * 0.75:  # 75% o más alcistas
            alignment_ratio = bullish_count / total_timeframes
            participating_tfs = [tf for tf, dir in trend_directions.items() if dir == "bullish"]
        elif bearish_count >= total_timeframes * 0.75:  # 75% o más bajistas
            alignment_ratio = bearish_count / total_timeframes
            participating_tfs = [tf for tf, dir in trend_directions.items() if dir == "bearish"]
        else:
            # Sin alineación clara
            alignment_ratio = max(bullish_count, bearish_count) / total_timeframes
            participating_tfs = []
        
        # Ponderar por fuerza de tendencias
        weighted_strength = 0.0
        for tf in participating_tfs:
            weight = self.timeframe_weights.get(tf, 0.1)
            weighted_strength += trend_strengths[tf] * weight
        
        # Score final combinando alineación y fuerza
        final_score = alignment_ratio * 0.7 + weighted_strength * 0.3
        
        return AlignmentScore(
            alignment_type=AlignmentType.TREND_ALIGNMENT,
            score=final_score,
            participating_timeframes=participating_tfs,
            weight=0.35  # Mayor peso para alineación de tendencias
        )
    
    def _calculate_momentum_sync(self, analyses: Dict[str, TimeframeAnalysis]) -> AlignmentScore:
        """Calcula sincronización de momentum"""
        momentum_scores = {}
        momentum_directions = {}
        
        for tf, analysis in analyses.items():
            momentum_scores[tf] = analysis.momentum_score
            momentum_directions[tf] = "bullish" if analysis.momentum_score > 0.1 else \
                                   "bearish" if analysis.momentum_score < -0.1 else "neutral"
        
        # Contar direcciones de momentum
        bullish_momentum = sum(1 for dir in momentum_directions.values() if dir == "bullish")
        bearish_momentum = sum(1 for dir in momentum_directions.values() if dir == "bearish")
        
        total_timeframes = len(analyses)
        
        # Calcular sincronización
        if bullish_momentum >= total_timeframes * 0.6:  # 60% o más con momentum alcista
            sync_ratio = bullish_momentum / total_timeframes
            participating_tfs = [tf for tf, dir in momentum_directions.items() if dir == "bullish"]
        elif bearish_momentum >= total_timeframes * 0.6:  # 60% o más con momentum bajista
            sync_ratio = bearish_momentum / total_timeframes
            participating_tfs = [tf for tf, dir in momentum_directions.items() if dir == "bearish"]
        else:
            sync_ratio = 0.5  # Sin sincronización clara
            participating_tfs = []
        
        # Ponderar por intensidad del momentum
        weighted_momentum = 0.0
        for tf in participating_tfs:
            weight = self.timeframe_weights.get(tf, 0.1)
            weighted_momentum += abs(momentum_scores[tf]) * weight
        
        final_score = sync_ratio * 0.6 + min(weighted_momentum, 1.0) * 0.4
        
        return AlignmentScore(
            alignment_type=AlignmentType.MOMENTUM_SYNC,
            score=final_score,
            participating_timeframes=participating_tfs,
            weight=0.25
        )
    
    def _calculate_structure_confluence(self, analyses: Dict[str, TimeframeAnalysis]) -> AlignmentScore:
        """Calcula confluencia de estructura de mercado"""
        structures = {}
        structure_breaks = {}
        
        for tf, analysis in analyses.items():
            structures[tf] = analysis.market_structure
            structure_breaks[tf] = analysis.structure_break
        
        # Contar estructuras por tipo
        uptrend_count = sum(1 for struct in structures.values() if struct == MarketStructure.UPTREND)
        downtrend_count = sum(1 for struct in structures.values() if struct == MarketStructure.DOWNTREND)
        ranging_count = sum(1 for struct in structures.values() if struct == MarketStructure.RANGING)
        
        total_timeframes = len(analyses)
        
        # Calcular confluencia estructural
        if uptrend_count >= total_timeframes * 0.7:
            confluence_ratio = uptrend_count / total_timeframes
            participating_tfs = [tf for tf, struct in structures.items() if struct == MarketStructure.UPTREND]
        elif downtrend_count >= total_timeframes * 0.7:
            confluence_ratio = downtrend_count / total_timeframes
            participating_tfs = [tf for tf, struct in structures.items() if struct == MarketStructure.DOWNTREND]
        else:
            confluence_ratio = 0.4  # Estructuras mixtas
            participating_tfs = []
        
        # Penalizar si hay muchas rupturas de estructura (inestabilidad)
        break_count = sum(1 for has_break in structure_breaks.values() if has_break)
        break_penalty = (break_count / total_timeframes) * 0.2
        
        final_score = max(0.0, confluence_ratio - break_penalty)
        
        return AlignmentScore(
            alignment_type=AlignmentType.STRUCTURE_CONFLUENCE,
            score=final_score,
            participating_timeframes=participating_tfs,
            weight=0.20
        )
    
    def _calculate_level_confluence(self, analyses: Dict[str, TimeframeAnalysis]) -> AlignmentScore:
        """Calcula confluencia de niveles de soporte/resistencia"""
        all_supports = []
        all_resistances = []
        participating_tfs = []
        
        # Recopilar todos los niveles
        for tf, analysis in analyses.items():
            weight = self.timeframe_weights.get(tf, 0.1)
            
            for support in analysis.support_levels:
                all_supports.append((support, tf, weight))
            
            for resistance in analysis.resistance_levels:
                all_resistances.append((resistance, tf, weight))
        
        # Encontrar confluencias en soportes
        support_confluences = self._find_level_confluences(all_supports)
        
        # Encontrar confluencias en resistencias
        resistance_confluences = self._find_level_confluences(all_resistances)
        
        # Calcular score basado en número y calidad de confluencias
        total_confluences = len(support_confluences) + len(resistance_confluences)
        
        if total_confluences > 0:
            # Score basado en número de confluencias y timeframes participantes
            confluence_score = min(1.0, total_confluences / 3.0)  # Máximo 3 confluencias esperadas
            
            # Identificar timeframes participantes
            for confluences in [support_confluences, resistance_confluences]:
                for confluence in confluences:
                    participating_tfs.extend(confluence['timeframes'])
            
            participating_tfs = list(set(participating_tfs))  # Eliminar duplicados
        else:
            confluence_score = 0.0
        
        return AlignmentScore(
            alignment_type=AlignmentType.LEVEL_CONFLUENCE,
            score=confluence_score,
            participating_timeframes=participating_tfs,
            weight=0.15
        )
    
    def _calculate_pattern_confluence(self, analyses: Dict[str, TimeframeAnalysis]) -> AlignmentScore:
        """Calcula confluencia de patrones"""
        pattern_types = {}
        pattern_confidences = {}
        
        for tf, analysis in analyses.items():
            if analysis.patterns:
                # Tomar el patrón más fuerte
                pattern_types[tf] = analysis.patterns[0]  # Primer patrón (más fuerte)
                pattern_confidences[tf] = analysis.pattern_confidence
        
        if not pattern_types:
            return AlignmentScore(
                alignment_type=AlignmentType.PATTERN_CONFLUENCE,
                score=0.0,
                participating_timeframes=[],
                weight=0.05
            )
        
        # Agrupar patrones por tipo
        pattern_groups = {}
        for tf, pattern in pattern_types.items():
            if pattern not in pattern_groups:
                pattern_groups[pattern] = []
            pattern_groups[pattern].append(tf)
        
        # Encontrar el grupo más grande
        largest_group = max(pattern_groups.values(), key=len)
        
        if len(largest_group) >= 2:  # Al menos 2 timeframes con el mismo patrón
            confluence_ratio = len(largest_group) / len(analyses)
            
            # Ponderar por confianza de patrones
            weighted_confidence = 0.0
            for tf in largest_group:
                weight = self.timeframe_weights.get(tf, 0.1)
                weighted_confidence += pattern_confidences[tf] * weight
            
            final_score = confluence_ratio * 0.6 + weighted_confidence * 0.4
            participating_tfs = largest_group
        else:
            final_score = 0.3  # Patrones diversos
            participating_tfs = []
        
        return AlignmentScore(
            alignment_type=AlignmentType.PATTERN_CONFLUENCE,
            score=final_score,
            participating_timeframes=participating_tfs,
            weight=0.05
        )
    
    def _find_level_confluences(self, levels: List[Tuple[float, str, float]]) -> List[Dict]:
        """Encuentra confluencias entre niveles de precio"""
        confluences = []
        
        if len(levels) < 2:
            return confluences
        
        # Ordenar niveles por precio
        sorted_levels = sorted(levels, key=lambda x: x[0])
        
        i = 0
        while i < len(sorted_levels):
            current_level = sorted_levels[i][0]
            confluence_group = [sorted_levels[i]]
            
            # Buscar niveles cercanos
            j = i + 1
            while j < len(sorted_levels):
                next_level = sorted_levels[j][0]
                
                # Verificar si están dentro de la tolerancia
                if abs(next_level - current_level) / current_level <= self.level_tolerance:
                    confluence_group.append(sorted_levels[j])
                    j += 1
                else:
                    break
            
            # Si hay confluencia (2+ niveles), agregarla
            if len(confluence_group) >= 2:
                avg_price = sum(level[0] for level in confluence_group) / len(confluence_group)
                timeframes = [level[1] for level in confluence_group]
                total_weight = sum(level[2] for level in confluence_group)
                
                confluences.append({
                    'price': avg_price,
                    'timeframes': timeframes,
                    'strength': min(1.0, total_weight),
                    'count': len(confluence_group)
                })
            
            i = j if j > i + 1 else i + 1
        
        return confluences
    
    def _detect_confluence_levels(self, analyses: Dict[str, TimeframeAnalysis]) -> List[ConfluenceLevel]:
        """Detecta niveles de confluencia específicos"""
        confluence_levels = []
        
        # Obtener precio actual (del primer análisis disponible)
        current_price = None
        for analysis in analyses.values():
            if hasattr(analysis, 'current_price'):
                current_price = analysis.current_price
                break
        
        if current_price is None:
            # Estimar precio actual basado en niveles
            all_levels = []
            for analysis in analyses.values():
                all_levels.extend(analysis.support_levels)
                all_levels.extend(analysis.resistance_levels)
            
            if all_levels:
                current_price = sum(all_levels) / len(all_levels)
            else:
                return confluence_levels
        
        # Recopilar todos los soportes
        all_supports = []
        for tf, analysis in analyses.items():
            for support in analysis.support_levels:
                all_supports.append((support, tf, self.timeframe_weights.get(tf, 0.1)))
        
        # Recopilar todas las resistencias
        all_resistances = []
        for tf, analysis in analyses.items():
            for resistance in analysis.resistance_levels:
                all_resistances.append((resistance, tf, self.timeframe_weights.get(tf, 0.1)))
        
        # Procesar soportes
        support_confluences = self._find_level_confluences(all_supports)
        for confluence in support_confluences:
            confluence_levels.append(ConfluenceLevel(
                price_level=confluence['price'],
                timeframes=confluence['timeframes'],
                level_type="support",
                strength=confluence['strength'],
                distance_from_current=abs(confluence['price'] - current_price) / current_price
            ))
        
        # Procesar resistencias
        resistance_confluences = self._find_level_confluences(all_resistances)
        for confluence in resistance_confluences:
            confluence_levels.append(ConfluenceLevel(
                price_level=confluence['price'],
                timeframes=confluence['timeframes'],
                level_type="resistance",
                strength=confluence['strength'],
                distance_from_current=abs(confluence['price'] - current_price) / current_price
            ))
        
        # Ordenar por fuerza (descendente)
        confluence_levels.sort(key=lambda x: x.strength, reverse=True)
        
        return confluence_levels[:10]  # Máximo 10 niveles más fuertes
    
    def _detect_divergences(self, analyses: Dict[str, TimeframeAnalysis]) -> List[Divergence]:
        """Detecta divergencias entre timeframes"""
        divergences = []
        
        # 1. Divergencias de tendencia
        trend_divergences = self._detect_trend_divergences(analyses)
        divergences.extend(trend_divergences)
        
        # 2. Divergencias de momentum
        momentum_divergences = self._detect_momentum_divergences(analyses)
        divergences.extend(momentum_divergences)
        
        # 3. Divergencias de estructura
        structure_divergences = self._detect_structure_divergences(analyses)
        divergences.extend(structure_divergences)
        
        return divergences
    
    def _detect_trend_divergences(self, analyses: Dict[str, TimeframeAnalysis]) -> List[Divergence]:
        """Detecta divergencias de tendencia entre timeframes"""
        divergences = []
        
        # Comparar tendencias entre timeframes importantes
        important_pairs = [('1d', '4h'), ('4h', '1h'), ('1d', '1h')]
        
        for tf1, tf2 in important_pairs:
            if tf1 in analyses and tf2 in analyses:
                trend1 = analyses[tf1].trend_direction
                trend2 = analyses[tf2].trend_direction
                
                # Detectar divergencia crítica (tendencias opuestas)
                if ((trend1 == "bullish" and trend2 == "bearish") or 
                    (trend1 == "bearish" and trend2 == "bullish")):
                    
                    # Calcular severidad basada en fuerza de tendencias
                    strength1 = analyses[tf1].trend_strength
                    strength2 = analyses[tf2].trend_strength
                    severity = (strength1 + strength2) / 2
                    
                    divergences.append(Divergence(
                        divergence_type=DivergenceType.TREND_DIVERGENCE,
                        timeframes_involved=[tf1, tf2],
                        severity=severity,
                        description=f"Tendencia {trend1} en {tf1} vs {trend2} en {tf2}",
                        impact_on_confidence=-0.3 * severity
                    ))
        
        return divergences
    
    def _detect_momentum_divergences(self, analyses: Dict[str, TimeframeAnalysis]) -> List[Divergence]:
        """Detecta divergencias de momentum"""
        divergences = []
        
        # Verificar si hay momentum divergente significativo
        momentum_values = {}
        for tf, analysis in analyses.items():
            momentum_values[tf] = analysis.momentum_score
        
        # Comparar momentum entre timeframes consecutivos
        tf_order = ['1d', '4h', '1h', '15m']
        available_tfs = [tf for tf in tf_order if tf in momentum_values]
        
        for i in range(len(available_tfs) - 1):
            tf1, tf2 = available_tfs[i], available_tfs[i + 1]
            momentum1, momentum2 = momentum_values[tf1], momentum_values[tf2]
            
            # Detectar divergencia significativa
            if abs(momentum1 - momentum2) > 0.5 and momentum1 * momentum2 < 0:
                severity = abs(momentum1 - momentum2) / 2.0
                
                divergences.append(Divergence(
                    divergence_type=DivergenceType.MOMENTUM_DIVERGENCE,
                    timeframes_involved=[tf1, tf2],
                    severity=severity,
                    description=f"Momentum divergente: {momentum1:.2f} ({tf1}) vs {momentum2:.2f} ({tf2})",
                    impact_on_confidence=-0.2 * severity
                ))
        
        return divergences
    
    def _detect_structure_divergences(self, analyses: Dict[str, TimeframeAnalysis]) -> List[Divergence]:
        """Detecta divergencias de estructura"""
        divergences = []
        
        # Verificar rupturas de estructura simultáneas (señal de cambio)
        structure_breaks = {}
        for tf, analysis in analyses.items():
            structure_breaks[tf] = analysis.structure_break
        
        break_count = sum(1 for has_break in structure_breaks.values() if has_break)
        total_tfs = len(analyses)
        
        # Si más del 50% tiene rupturas de estructura
        if break_count > total_tfs * 0.5:
            severity = break_count / total_tfs
            breaking_tfs = [tf for tf, has_break in structure_breaks.items() if has_break]
            
            divergences.append(Divergence(
                divergence_type=DivergenceType.STRUCTURE_DIVERGENCE,
                timeframes_involved=breaking_tfs,
                severity=severity,
                description=f"Rupturas de estructura simultáneas en {break_count}/{total_tfs} timeframes",
                impact_on_confidence=-0.15 * severity
            ))
        
        return divergences
    
    def _calculate_overall_confluence_score(self, alignment_scores: Dict[AlignmentType, AlignmentScore],
                                          confluence_levels: List[ConfluenceLevel],
                                          divergences: List[Divergence]) -> float:
        """Calcula el score general de confluencia"""
        
        # 1. Score base de alineaciones (70% del total)
        alignment_score = 0.0
        total_weight = 0.0
        
        for alignment_type, score_obj in alignment_scores.items():
            weighted_score = score_obj.score * score_obj.weight
            alignment_score += weighted_score
            total_weight += score_obj.weight
        
        if total_weight > 0:
            alignment_score = (alignment_score / total_weight) * 70
        
        # 2. Bonus por niveles de confluencia (20% del total)
        level_bonus = 0.0
        if confluence_levels:
            # Bonus basado en número y fuerza de niveles
            strong_levels = [level for level in confluence_levels if level.strength > 0.7]
            level_bonus = min(20.0, len(strong_levels) * 5.0)
        
        # 3. Penalización por divergencias (hasta -30%)
        divergence_penalty = 0.0
        for divergence in divergences:
            divergence_penalty += abs(divergence.impact_on_confidence) * 30
        
        # Score final
        final_score = alignment_score + level_bonus - divergence_penalty
        
        return max(0.0, min(100.0, final_score))
    
    def _determine_confluence_strength(self, score: float) -> ConfluenceStrength:
        """Determina la fuerza de confluencia basada en el score"""
        if score >= self.confluence_thresholds[ConfluenceStrength.CRITICAL]:
            return ConfluenceStrength.CRITICAL
        elif score >= self.confluence_thresholds[ConfluenceStrength.STRONG]:
            return ConfluenceStrength.STRONG
        elif score >= self.confluence_thresholds[ConfluenceStrength.MODERATE]:
            return ConfluenceStrength.MODERATE
        else:
            return ConfluenceStrength.WEAK
    
    def _calculate_confluence_confidence(self, alignment_scores: Dict[AlignmentType, AlignmentScore],
                                       divergences: List[Divergence], 
                                       timeframe_count: int) -> float:
        """Calcula la confianza en el análisis de confluencia"""
        
        base_confidence = 0.3  # Base
        
        # Bonus por alineaciones fuertes
        for alignment_type, score_obj in alignment_scores.items():
            if score_obj.score > 0.7:
                base_confidence += 0.15
        
        # Bonus por número de timeframes
        tf_bonus = min(0.2, (timeframe_count - 2) * 0.1)
        base_confidence += tf_bonus
        
        # Penalización por divergencias
        for divergence in divergences:
            base_confidence += divergence.impact_on_confidence
        
        return max(0.0, min(1.0, base_confidence))
    
    def _generate_trade_recommendation(self, analyses: Dict[str, TimeframeAnalysis],
                                     alignment_scores: Dict[AlignmentType, AlignmentScore],
                                     overall_score: float) -> Tuple[str, str]:
        """Genera recomendación de trading y timeframe recomendado"""
        
        # Determinar dirección dominante
        trend_alignment = alignment_scores.get(AlignmentType.TREND_ALIGNMENT)
        
        if trend_alignment and trend_alignment.score > 0.7:
            # Hay alineación fuerte de tendencias
            dominant_direction = self._determine_dominant_trend(analyses)
            
            if overall_score >= 80:
                recommendation = f"STRONG_{dominant_direction.upper()}"
            elif overall_score >= 60:
                recommendation = dominant_direction.upper()
            else:
                recommendation = "NEUTRAL"
        else:
            recommendation = "NEUTRAL"
        
        # Determinar mejor timeframe para entrada
        # Priorizar timeframes con mejor score individual
        best_tf = "1h"  # Default
        best_score = 0.0
        
        for tf, analysis in analyses.items():
            if analysis.overall_score > best_score:
                best_score = analysis.overall_score
                best_tf = tf
        
        return recommendation, best_tf
    
    def _determine_dominant_trend(self, analyses: Dict[str, TimeframeAnalysis]) -> str:
        """Determina la tendencia dominante ponderada"""
        weighted_trends = {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0}
        
        for tf, analysis in analyses.items():
            weight = self.timeframe_weights.get(tf, 0.1)
            trend_strength = analysis.trend_strength
            
            weighted_trends[analysis.trend_direction] += weight * trend_strength
        
        return max(weighted_trends, key=weighted_trends.get)
    
    def _assess_risk_level(self, divergences: List[Divergence], confidence: float) -> str:
        """Evalúa el nivel de riesgo"""
        
        # Contar divergencias críticas
        critical_divergences = [d for d in divergences if d.severity > 0.7]
        
        if len(critical_divergences) > 0 or confidence < 0.4:
            return "high"
        elif len(divergences) > 2 or confidence < 0.6:
            return "medium"
        else:
            return "low"
    
    def _extract_symbol_from_analyses(self, analyses: Dict[str, TimeframeAnalysis]) -> str:
        """Extrae el símbolo de los análisis (asumiendo que todos son del mismo símbolo)"""
        # Por ahora retornamos un placeholder, en implementación real
        # el símbolo vendría como parámetro o en los análisis
        return "UNKNOWN"


# Función de conveniencia
def analyze_confluence(analyses: Dict[str, TimeframeAnalysis]) -> ConfluenceResult:
    """
    Función de conveniencia para análisis de confluencia
    
    Args:
        analyses: Diccionario con análisis por timeframe
        
    Returns:
        ConfluenceResult con análisis completo
    """
    engine = ConfluenceEngine()
    return engine.calculate_confluence(analyses)