"""
Sistema Avanzado de Detección de Patrones
=========================================

Este módulo implementa un sistema sofisticado de detección de patrones chartistas
y técnicos para trading algorítmico. 

Autor: Sistema de Trading Algorítmico
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pytz

from ..config.settings import config
from ..utils.logger import get_logger, LogContext
from ..utils.exceptions import (
    PatternDetectionException,
    AnalysisException,
    ErrorCodes
)

# Zona horaria de Ecuador (UTC-5)
ECUADOR_TZ = pytz.timezone('America/Guayaquil')

class PatternCategory(Enum):
    """Categorías de patrones"""
    REVERSAL = "reversal"
    CONTINUATION = "continuation"
    CONSOLIDATION = "consolidation"
    BREAKOUT = "breakout"

class PatternReliability(Enum):
    """Niveles de confiabilidad de patrones"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class PatternTarget:
    """Objetivo de precio para un patrón"""
    price: float
    probability: float  # 0.0 a 1.0
    timeframe_estimate: int  # Velas estimadas para alcanzar

@dataclass
class DetectedPattern:
    """Patrón detectado con toda su información"""
    pattern_type: str
    category: PatternCategory
    reliability: PatternReliability
    confidence: float  # 0.0 a 1.0
    
    # Información del patrón
    start_index: int
    end_index: int
    formation_bars: int
    
    # Niveles clave
    entry_price: float
    stop_loss: float
    targets: List[PatternTarget]
    invalidation_level: float
    
    # Contexto
    trend_context: str  # "with_trend", "against_trend", "neutral"
    volume_confirmation: bool
    timeframe: str
    
    # Metadata
    detection_time: datetime
    description: str
    notes: Optional[str] = None

class PatternDetector:
    """
    Detector principal de patrones chartistas y técnicos
    """
    
    def __init__(self):
        self.logger = get_logger("PatternDetector")
        
        # Configuración de detección
        self.min_pattern_bars = 5
        self.max_pattern_bars = 50
        self.volume_confirmation_threshold = 1.2  # 20% más volumen
        
        # Umbrales de confianza por tipo de patrón (más permisivos para testing)
        self.confidence_thresholds = {
            "triangle": 0.5,
            "flag": 0.6,
            "wedge": 0.5,
            "head_shoulders": 0.6,
            "double_top": 0.6,
            "double_bottom": 0.6,
            "cup_handle": 0.5,
            "bull_flag": 0.6,
            "bear_flag": 0.6,
            "rising_wedge": 0.5,
            "falling_wedge": 0.5
        }
    
    def detect_patterns(self, data: pd.DataFrame, symbol: str, 
                       timeframe: str) -> List[DetectedPattern]:
        """
        Detecta todos los patrones en los datos proporcionados
        
        Args:
            data: DataFrame con datos OHLCV
            symbol: Símbolo del activo
            timeframe: Timeframe de los datos
            
        Returns:
            Lista de patrones detectados
        """
        context = LogContext(
            component="pattern_detector",
            symbol=symbol,
            timeframe=timeframe
        )
        
        self.logger.debug(
            f"Iniciando detección de patrones para {symbol} {timeframe}",
            context=context
        )
        
        if len(data) < self.min_pattern_bars:
            self.logger.warning(
                f"Datos insuficientes para detección: {len(data)} velas",
                context=context
            )
            return []
        
        detected_patterns = []
        
        try:
            # 1. Patrones de triángulos
            triangle_patterns = self._detect_triangles(data, symbol, timeframe)
            detected_patterns.extend(triangle_patterns)
            
            # 2. Patrones de banderas y banderines
            flag_patterns = self._detect_flags(data, symbol, timeframe)
            detected_patterns.extend(flag_patterns)
            
            # 3. Patrones de cuñas
            wedge_patterns = self._detect_wedges(data, symbol, timeframe)
            detected_patterns.extend(wedge_patterns)
            
            # 4. Patrones de hombro-cabeza-hombro
            hcs_patterns = self._detect_head_shoulders(data, symbol, timeframe)
            detected_patterns.extend(hcs_patterns)
            
            # 5. Patrones de doble techo/suelo
            double_patterns = self._detect_double_patterns(data, symbol, timeframe)
            detected_patterns.extend(double_patterns)
            
            # 6. Patrones de copa y asa
            cup_patterns = self._detect_cup_handle(data, symbol, timeframe)
            detected_patterns.extend(cup_patterns)
            
            # Filtrar patrones por confianza mínima
            filtered_patterns = [
                p for p in detected_patterns 
                if p.confidence >= self.confidence_thresholds.get(p.pattern_type, 0.6)
            ]
            
            self.logger.info(
                f"Detección completada: {len(filtered_patterns)} patrones válidos "
                f"de {len(detected_patterns)} detectados",
                context=context,
                extra_fields={
                    "patterns_detected": len(detected_patterns),
                    "patterns_valid": len(filtered_patterns),
                    "symbol": symbol,
                    "timeframe": timeframe
                }
            )
            
            return filtered_patterns
            
        except Exception as e:
            self.logger.error(f"Error en detección de patrones: {str(e)}", context=context)
            raise PatternDetectionException(
                f"Fallo en detección de patrones: {str(e)}",
                ErrorCodes.ANALYSIS_PATTERN_DETECTION_FAILED
            )
    
    def _detect_triangles(self, data: pd.DataFrame, symbol: str, 
                         timeframe: str) -> List[DetectedPattern]:
        """Detecta patrones de triángulos (ascendente, descendente, simétrico)"""
        patterns = []
        
        if len(data) < 20:
            return patterns
        
        # Buscar formaciones triangulares en ventanas deslizantes
        for window_size in range(15, min(40, len(data))):
            for start_idx in range(len(data) - window_size):
                end_idx = start_idx + window_size
                window_data = data.iloc[start_idx:end_idx]
                
                triangle_pattern = self._analyze_triangle_formation(
                    window_data, start_idx, end_idx, symbol, timeframe
                )
                
                if triangle_pattern:
                    patterns.append(triangle_pattern)
        
        # Eliminar patrones superpuestos (quedarse con el de mayor confianza)
        return self._remove_overlapping_patterns(patterns)
    
    def _analyze_triangle_formation(self, data: pd.DataFrame, start_idx: int,
                                   end_idx: int, symbol: str, 
                                   timeframe: str) -> Optional[DetectedPattern]:
        """Analiza si una ventana de datos forma un triángulo"""
        
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        volumes = data['volume'].values
        
        # Encontrar máximos y mínimos locales
        high_peaks = self._find_peaks(highs, min_distance=3)
        low_valleys = self._find_valleys(lows, min_distance=3)
        
        if len(high_peaks) < 2 or len(low_valleys) < 2:
            return None
        
        # Calcular líneas de tendencia
        high_slope, high_r2 = self._calculate_trendline(high_peaks, highs)
        low_slope, low_r2 = self._calculate_trendline(low_valleys, lows)
        
        # Verificar calidad de las líneas de tendencia (más permisivo)
        if high_r2 < 0.5 or low_r2 < 0.5:
            return None
        
        # Determinar tipo de triángulo
        triangle_type = self._classify_triangle(high_slope, low_slope)
        
        if triangle_type is None:
            return None
        
        # Calcular confianza
        confidence = self._calculate_triangle_confidence(
            high_r2, low_r2, len(high_peaks), len(low_valleys), volumes
        )
        
        # Calcular niveles clave
        current_price = closes[-1]
        entry_price = current_price
        
        # Stop loss basado en el patrón
        if triangle_type == "ascending":
            stop_loss = min(lows[-5:]) * 0.98
            target_price = max(highs) * 1.05
        elif triangle_type == "descending":
            stop_loss = max(highs[-5:]) * 1.02
            target_price = min(lows) * 0.95
        else:  # simétrico
            range_size = max(highs) - min(lows)
            if high_slope > 0:  # Probable ruptura alcista
                stop_loss = min(lows[-5:]) * 0.98
                target_price = current_price + range_size
            else:  # Probable ruptura bajista
                stop_loss = max(highs[-5:]) * 1.02
                target_price = current_price - range_size
        
        # Crear patrón
        pattern = DetectedPattern(
            pattern_type="triangle",
            category=PatternCategory.CONSOLIDATION,
            reliability=PatternReliability.MEDIUM,
            confidence=confidence,
            start_index=start_idx,
            end_index=end_idx,
            formation_bars=end_idx - start_idx,
            entry_price=entry_price,
            stop_loss=stop_loss,
            targets=[PatternTarget(target_price, 0.7, 10)],
            invalidation_level=stop_loss,
            trend_context="neutral",
            volume_confirmation=self._check_volume_confirmation(volumes),
            timeframe=timeframe,
            detection_time=datetime.now(ECUADOR_TZ),
            description=f"Triángulo {triangle_type} detectado",
            notes=f"R² superior: {high_r2:.3f}, R² inferior: {low_r2:.3f}"
        )
        
        return pattern
    
    def _detect_flags(self, data: pd.DataFrame, symbol: str, 
                     timeframe: str) -> List[DetectedPattern]:
        """Detecta patrones de banderas y banderines"""
        patterns = []
        
        if len(data) < 15:
            return patterns
        
        # Buscar movimientos fuertes seguidos de consolidación
        for i in range(10, len(data) - 5):
            # Verificar movimiento fuerte previo (mástil)
            pole_start = max(0, i - 10)
            pole_data = data.iloc[pole_start:i]
            
            if len(pole_data) < 5:
                continue
            
            pole_move = (pole_data['close'].iloc[-1] - pole_data['close'].iloc[0]) / pole_data['close'].iloc[0]
            
            # Debe ser un movimiento significativo (>3%)
            if abs(pole_move) < 0.03:
                continue
            
            # Verificar consolidación posterior (bandera)
            flag_end = min(len(data), i + 8)
            flag_data = data.iloc[i:flag_end]
            
            if len(flag_data) < 5:
                continue
            
            flag_pattern = self._analyze_flag_formation(
                pole_data, flag_data, pole_move, i, symbol, timeframe
            )
            
            if flag_pattern:
                patterns.append(flag_pattern)
        
        return self._remove_overlapping_patterns(patterns)
    
    def _analyze_flag_formation(self, pole_data: pd.DataFrame, flag_data: pd.DataFrame,
                               pole_move: float, flag_start: int, symbol: str,
                               timeframe: str) -> Optional[DetectedPattern]:
        """Analiza formación de bandera"""
        
        # La bandera debe ser una consolidación pequeña
        flag_range = (flag_data['high'].max() - flag_data['low'].min()) / flag_data['close'].iloc[0]
        
        # Rango de consolidación debe ser pequeño (<2%)
        if flag_range > 0.02:
            return None
        
        # Verificar dirección de la bandera (ligeramente contra la tendencia)
        flag_slope = (flag_data['close'].iloc[-1] - flag_data['close'].iloc[0]) / flag_data['close'].iloc[0]
        
        # Para movimiento alcista, bandera debe ser ligeramente bajista o plana
        if pole_move > 0 and flag_slope > 0.01:
            return None
        
        # Para movimiento bajista, bandera debe ser ligeramente alcista o plana
        if pole_move < 0 and flag_slope < -0.01:
            return None
        
        # Calcular confianza
        volume_confirmation = self._check_volume_confirmation(flag_data['volume'].values)
        confidence = 0.6 + (0.2 if volume_confirmation else 0) + min(0.2, abs(pole_move) * 5)
        
        # Calcular objetivos
        current_price = flag_data['close'].iloc[-1]
        pole_size = abs(pole_data['close'].iloc[-1] - pole_data['close'].iloc[0])
        
        if pole_move > 0:  # Bandera alcista
            target_price = current_price + pole_size
            stop_loss = flag_data['low'].min() * 0.99
            pattern_type = "bull_flag"
        else:  # Bandera bajista
            target_price = current_price - pole_size
            stop_loss = flag_data['high'].max() * 1.01
            pattern_type = "bear_flag"
        
        pattern = DetectedPattern(
            pattern_type=pattern_type,
            category=PatternCategory.CONTINUATION,
            reliability=PatternReliability.HIGH,
            confidence=confidence,
            start_index=flag_start - len(pole_data),
            end_index=flag_start + len(flag_data),
            formation_bars=len(pole_data) + len(flag_data),
            entry_price=current_price,
            stop_loss=stop_loss,
            targets=[PatternTarget(target_price, 0.8, 8)],
            invalidation_level=stop_loss,
            trend_context="with_trend",
            volume_confirmation=volume_confirmation,
            timeframe=timeframe,
            detection_time=datetime.now(ECUADOR_TZ),
            description=f"Bandera {'alcista' if pole_move > 0 else 'bajista'} detectada",
            notes=f"Movimiento mástil: {pole_move:.2%}, Rango bandera: {flag_range:.2%}"
        )
        
        return pattern
    
    def _detect_wedges(self, data: pd.DataFrame, symbol: str, 
                      timeframe: str) -> List[DetectedPattern]:
        """Detecta patrones de cuñas (ascendente/descendente)"""
        patterns = []
        
        # Implementación simplificada - buscar convergencia de líneas de tendencia
        for window_size in range(20, min(50, len(data))):
            for start_idx in range(len(data) - window_size):
                end_idx = start_idx + window_size
                window_data = data.iloc[start_idx:end_idx]
                
                wedge_pattern = self._analyze_wedge_formation(
                    window_data, start_idx, end_idx, symbol, timeframe
                )
                
                if wedge_pattern:
                    patterns.append(wedge_pattern)
        
        return self._remove_overlapping_patterns(patterns)
    
    def _analyze_wedge_formation(self, data: pd.DataFrame, start_idx: int,
                                end_idx: int, symbol: str, 
                                timeframe: str) -> Optional[DetectedPattern]:
        """Analiza formación de cuña"""
        
        highs = data['high'].values
        lows = data['low'].values
        
        # Encontrar picos y valles
        high_peaks = self._find_peaks(highs, min_distance=3)
        low_valleys = self._find_valleys(lows, min_distance=3)
        
        if len(high_peaks) < 3 or len(low_valleys) < 3:
            return None
        
        # Calcular líneas de tendencia
        high_slope, high_r2 = self._calculate_trendline(high_peaks, highs)
        low_slope, low_r2 = self._calculate_trendline(low_valleys, lows)
        
        # Verificar convergencia (ambas líneas deben tener la misma dirección general)
        if high_slope * low_slope <= 0:  # Direcciones opuestas
            return None
        
        # Verificar que las líneas converjan
        slope_diff = abs(high_slope - low_slope)
        if slope_diff < 0.001:  # Líneas demasiado paralelas
            return None
        
        # Determinar tipo de cuña
        if high_slope > 0 and low_slope > 0:
            wedge_type = "rising_wedge"  # Generalmente bajista
            expected_direction = "bearish"
        else:
            wedge_type = "falling_wedge"  # Generalmente alcista
            expected_direction = "bullish"
        
        # Calcular confianza
        confidence = min(0.9, (high_r2 + low_r2) / 2 + 0.1)
        
        # Calcular niveles
        current_price = data['close'].iloc[-1]
        price_range = max(highs) - min(lows)
        
        if expected_direction == "bullish":
            target_price = current_price + price_range * 0.8
            stop_loss = min(lows[-3:]) * 0.98
        else:
            target_price = current_price - price_range * 0.8
            stop_loss = max(highs[-3:]) * 1.02
        
        pattern = DetectedPattern(
            pattern_type=wedge_type,
            category=PatternCategory.REVERSAL,
            reliability=PatternReliability.MEDIUM,
            confidence=confidence,
            start_index=start_idx,
            end_index=end_idx,
            formation_bars=end_idx - start_idx,
            entry_price=current_price,
            stop_loss=stop_loss,
            targets=[PatternTarget(target_price, 0.65, 12)],
            invalidation_level=stop_loss,
            trend_context="against_trend",
            volume_confirmation=self._check_volume_confirmation(data['volume'].values),
            timeframe=timeframe,
            detection_time=datetime.now(ECUADOR_TZ),
            description=f"Cuña {wedge_type.replace('_', ' ')} detectada",
            notes=f"Dirección esperada: {expected_direction}"
        )
        
        return pattern
    
    def _detect_head_shoulders(self, data: pd.DataFrame, symbol: str, 
                              timeframe: str) -> List[DetectedPattern]:
        """Detecta patrones de hombro-cabeza-hombro"""
        patterns = []
        
        if len(data) < 25:
            return patterns
        
        # Buscar formaciones H-C-H en ventanas grandes
        for window_size in range(25, min(60, len(data))):
            for start_idx in range(len(data) - window_size):
                end_idx = start_idx + window_size
                window_data = data.iloc[start_idx:end_idx]
                
                hcs_pattern = self._analyze_hcs_formation(
                    window_data, start_idx, end_idx, symbol, timeframe
                )
                
                if hcs_pattern:
                    patterns.append(hcs_pattern)
        
        return self._remove_overlapping_patterns(patterns)
    
    def _analyze_hcs_formation(self, data: pd.DataFrame, start_idx: int,
                              end_idx: int, symbol: str, 
                              timeframe: str) -> Optional[DetectedPattern]:
        """Analiza formación hombro-cabeza-hombro"""
        
        highs = data['high'].values
        lows = data['low'].values
        
        # Encontrar picos significativos
        peaks = self._find_peaks(highs, min_distance=5, prominence=0.01)
        
        if len(peaks) < 3:
            return None
        
        # Buscar patrón H-C-H en los últimos picos
        for i in range(len(peaks) - 2):
            left_shoulder = peaks[i]
            head = peaks[i + 1]
            right_shoulder = peaks[i + 2]
            
            # Verificar que la cabeza sea más alta que los hombros
            if not (highs[head] > highs[left_shoulder] and highs[head] > highs[right_shoulder]):
                continue
            
            # Verificar simetría aproximada de los hombros
            shoulder_diff = abs(highs[left_shoulder] - highs[right_shoulder]) / highs[head]
            if shoulder_diff > 0.05:  # Más del 5% de diferencia
                continue
            
            # Encontrar línea de cuello
            neckline_level = self._find_neckline(data, left_shoulder, head, right_shoulder)
            
            if neckline_level is None:
                continue
            
            # Calcular confianza
            symmetry_score = 1 - shoulder_diff
            height_ratio = (highs[head] - neckline_level) / neckline_level
            confidence = min(0.9, symmetry_score * 0.5 + min(0.4, height_ratio * 10))
            
            # Calcular objetivos
            current_price = data['close'].iloc[-1]
            head_height = highs[head] - neckline_level
            target_price = neckline_level - head_height  # Proyección bajista
            
            pattern = DetectedPattern(
                pattern_type="head_shoulders",
                category=PatternCategory.REVERSAL,
                reliability=PatternReliability.HIGH,
                confidence=confidence,
                start_index=start_idx + left_shoulder,
                end_index=start_idx + right_shoulder,
                formation_bars=right_shoulder - left_shoulder,
                entry_price=neckline_level,
                stop_loss=highs[head] * 1.02,
                targets=[PatternTarget(target_price, 0.75, 15)],
                invalidation_level=highs[head],
                trend_context="against_trend",
                volume_confirmation=self._check_volume_confirmation(data['volume'].values),
                timeframe=timeframe,
                detection_time=datetime.now(ECUADOR_TZ),
                description="Hombro-Cabeza-Hombro detectado",
                notes=f"Línea de cuello: ${neckline_level:.2f}, Altura: {head_height:.2f}"
            )
            
            return pattern
        
        return None
    
    def _detect_double_patterns(self, data: pd.DataFrame, symbol: str, 
                               timeframe: str) -> List[DetectedPattern]:
        """Detecta patrones de doble techo y doble suelo"""
        patterns = []
        
        # Detectar dobles techos
        double_tops = self._detect_double_tops(data, symbol, timeframe)
        patterns.extend(double_tops)
        
        # Detectar dobles suelos
        double_bottoms = self._detect_double_bottoms(data, symbol, timeframe)
        patterns.extend(double_bottoms)
        
        return patterns
    
    def _detect_double_tops(self, data: pd.DataFrame, symbol: str, 
                           timeframe: str) -> List[DetectedPattern]:
        """Detecta patrones de doble techo"""
        patterns = []
        
        if len(data) < 20:
            return patterns
        
        highs = data['high'].values
        peaks = self._find_peaks(highs, min_distance=5)
        
        # Buscar pares de picos similares
        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                peak1, peak2 = peaks[i], peaks[j]
                
                # Verificar que estén suficientemente separados
                if peak2 - peak1 < 8:
                    continue
                
                # Verificar similitud de alturas
                height_diff = abs(highs[peak1] - highs[peak2]) / max(highs[peak1], highs[peak2])
                if height_diff > 0.03:  # Más del 3% de diferencia
                    continue
                
                # Verificar que haya un valle significativo entre ellos
                valley_data = data.iloc[peak1:peak2]
                valley_low = valley_data['low'].min()
                valley_depth = (min(highs[peak1], highs[peak2]) - valley_low) / valley_low
                
                if valley_depth < 0.02:  # Valle muy poco profundo
                    continue
                
                # Calcular confianza
                confidence = 0.7 + (0.2 if valley_depth > 0.05 else 0.1) + (0.1 if height_diff < 0.01 else 0)
                
                # Crear patrón
                avg_peak_height = (highs[peak1] + highs[peak2]) / 2
                target_price = valley_low - (avg_peak_height - valley_low)
                
                pattern = DetectedPattern(
                    pattern_type="double_top",
                    category=PatternCategory.REVERSAL,
                    reliability=PatternReliability.HIGH,
                    confidence=confidence,
                    start_index=peak1,
                    end_index=peak2,
                    formation_bars=peak2 - peak1,
                    entry_price=valley_low,
                    stop_loss=avg_peak_height * 1.02,
                    targets=[PatternTarget(target_price, 0.7, 12)],
                    invalidation_level=avg_peak_height,
                    trend_context="against_trend",
                    volume_confirmation=self._check_volume_confirmation(data['volume'].values),
                    timeframe=timeframe,
                    detection_time=datetime.now(ECUADOR_TZ),
                    description="Doble techo detectado",
                    notes=f"Picos en ${highs[peak1]:.2f} y ${highs[peak2]:.2f}"
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _detect_double_bottoms(self, data: pd.DataFrame, symbol: str, 
                              timeframe: str) -> List[DetectedPattern]:
        """Detecta patrones de doble suelo"""
        patterns = []
        
        if len(data) < 20:
            return patterns
        
        lows = data['low'].values
        valleys = self._find_valleys(lows, min_distance=5)
        
        # Buscar pares de valles similares
        for i in range(len(valleys) - 1):
            for j in range(i + 1, len(valleys)):
                valley1, valley2 = valleys[i], valleys[j]
                
                if valley2 - valley1 < 8:
                    continue
                
                # Verificar similitud de profundidades
                depth_diff = abs(lows[valley1] - lows[valley2]) / min(lows[valley1], lows[valley2])
                if depth_diff > 0.03:
                    continue
                
                # Verificar pico intermedio
                peak_data = data.iloc[valley1:valley2]
                peak_high = peak_data['high'].max()
                peak_height = (peak_high - max(lows[valley1], lows[valley2])) / max(lows[valley1], lows[valley2])
                
                if peak_height < 0.02:
                    continue
                
                # Calcular confianza
                confidence = 0.7 + (0.2 if peak_height > 0.05 else 0.1) + (0.1 if depth_diff < 0.01 else 0)
                
                # Crear patrón
                avg_valley_depth = (lows[valley1] + lows[valley2]) / 2
                target_price = peak_high + (peak_high - avg_valley_depth)
                
                pattern = DetectedPattern(
                    pattern_type="double_bottom",
                    category=PatternCategory.REVERSAL,
                    reliability=PatternReliability.HIGH,
                    confidence=confidence,
                    start_index=valley1,
                    end_index=valley2,
                    formation_bars=valley2 - valley1,
                    entry_price=peak_high,
                    stop_loss=avg_valley_depth * 0.98,
                    targets=[PatternTarget(target_price, 0.7, 12)],
                    invalidation_level=avg_valley_depth,
                    trend_context="against_trend",
                    volume_confirmation=self._check_volume_confirmation(data['volume'].values),
                    timeframe=timeframe,
                    detection_time=datetime.now(ECUADOR_TZ),
                    description="Doble suelo detectado",
                    notes=f"Valles en ${lows[valley1]:.2f} y ${lows[valley2]:.2f}"
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _detect_cup_handle(self, data: pd.DataFrame, symbol: str, 
                          timeframe: str) -> List[DetectedPattern]:
        """Detecta patrones de copa y asa"""
        patterns = []
        
        if len(data) < 30:
            return patterns
        
        # Buscar formaciones de copa (forma de U)
        for window_size in range(30, min(80, len(data))):
            for start_idx in range(len(data) - window_size):
                end_idx = start_idx + window_size
                window_data = data.iloc[start_idx:end_idx]
                
                cup_pattern = self._analyze_cup_formation(
                    window_data, start_idx, end_idx, symbol, timeframe
                )
                
                if cup_pattern:
                    patterns.append(cup_pattern)
        
        return self._remove_overlapping_patterns(patterns)
    
    def _analyze_cup_formation(self, data: pd.DataFrame, start_idx: int,
                              end_idx: int, symbol: str, 
                              timeframe: str) -> Optional[DetectedPattern]:
        """Analiza formación de copa y asa"""
        
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        
        # La copa debe empezar y terminar en niveles similares
        start_price = closes[0]
        end_price = closes[-1]
        
        price_diff = abs(end_price - start_price) / start_price
        if price_diff > 0.05:  # Más del 5% de diferencia
            return None
        
        # Debe haber una caída significativa en el medio (la copa)
        min_price = min(lows)
        cup_depth = (start_price - min_price) / start_price
        
        if cup_depth < 0.1:  # Menos del 10% de profundidad
            return None
        
        # Buscar el asa (consolidación al final)
        handle_start = int(len(data) * 0.7)  # Último 30%
        handle_data = data.iloc[handle_start:]
        
        if len(handle_data) < 5:
            return None
        
        # El asa debe ser una pequeña consolidación
        handle_range = (handle_data['high'].max() - handle_data['low'].min()) / handle_data['close'].iloc[0]
        
        if handle_range > 0.15:  # Más del 15% de rango
            return None
        
        # Calcular confianza
        cup_symmetry = self._calculate_cup_symmetry(lows)
        confidence = 0.6 + cup_symmetry * 0.2 + (0.1 if handle_range < 0.1 else 0)
        
        # Calcular objetivos
        current_price = closes[-1]
        cup_height = start_price - min_price
        target_price = current_price + cup_height
        
        pattern = DetectedPattern(
            pattern_type="cup_handle",
            category=PatternCategory.CONTINUATION,
            reliability=PatternReliability.MEDIUM,
            confidence=confidence,
            start_index=start_idx,
            end_index=end_idx,
            formation_bars=end_idx - start_idx,
            entry_price=current_price,
            stop_loss=handle_data['low'].min() * 0.98,
            targets=[PatternTarget(target_price, 0.65, 15)],
            invalidation_level=min_price,
            trend_context="with_trend",
            volume_confirmation=self._check_volume_confirmation(data['volume'].values),
            timeframe=timeframe,
            detection_time=datetime.now(ECUADOR_TZ),
            description="Copa y asa detectada",
            notes=f"Profundidad copa: {cup_depth:.2%}, Rango asa: {handle_range:.2%}"
        )
        
        return pattern
    
    # Métodos auxiliares
    
    def _find_peaks(self, data: np.ndarray, min_distance: int = 1, 
                   prominence: float = 0.0) -> List[int]:
        """Encuentra picos en los datos"""
        peaks = []
        
        for i in range(min_distance, len(data) - min_distance):
            is_peak = True
            
            # Verificar que sea mayor que los vecinos
            for j in range(i - min_distance, i + min_distance + 1):
                if j != i and data[j] >= data[i]:
                    is_peak = False
                    break
            
            # Verificar prominencia si se especifica
            if is_peak and prominence > 0:
                left_min = min(data[max(0, i - 10):i])
                right_min = min(data[i:min(len(data), i + 10)])
                peak_prominence = data[i] - max(left_min, right_min)
                
                if peak_prominence / data[i] < prominence:
                    is_peak = False
            
            if is_peak:
                peaks.append(i)
        
        return peaks
    
    def _find_valleys(self, data: np.ndarray, min_distance: int = 1) -> List[int]:
        """Encuentra valles en los datos"""
        valleys = []
        
        for i in range(min_distance, len(data) - min_distance):
            is_valley = True
            
            # Verificar que sea menor que los vecinos
            for j in range(i - min_distance, i + min_distance + 1):
                if j != i and data[j] <= data[i]:
                    is_valley = False
                    break
            
            if is_valley:
                valleys.append(i)
        
        return valleys
    
    def _calculate_trendline(self, points: List[int], values: np.ndarray) -> Tuple[float, float]:
        """Calcula línea de tendencia y R²"""
        if len(points) < 2:
            return 0.0, 0.0
        
        x = np.array(points)
        y = np.array([values[i] for i in points])
        
        # Regresión lineal
        slope, intercept = np.polyfit(x, y, 1)
        
        # Calcular R²
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return slope, max(0, r_squared)
    
    def _classify_triangle(self, high_slope: float, low_slope: float) -> Optional[str]:
        """Clasifica el tipo de triángulo"""
        slope_threshold = 0.001
        
        if abs(high_slope) < slope_threshold and low_slope > slope_threshold:
            return "ascending"
        elif high_slope < -slope_threshold and abs(low_slope) < slope_threshold:
            return "descending"
        elif abs(high_slope - low_slope) < slope_threshold:
            return "symmetric"
        
        return None
    
    def _calculate_triangle_confidence(self, high_r2: float, low_r2: float,
                                     high_peaks: int, low_valleys: int,
                                     volumes: np.ndarray) -> float:
        """Calcula confianza del triángulo"""
        base_confidence = (high_r2 + low_r2) / 2
        
        # Bonus por número de puntos de contacto
        contact_bonus = min(0.2, (high_peaks + low_valleys - 4) * 0.05)
        
        # Bonus por confirmación de volumen
        volume_bonus = 0.1 if self._check_volume_confirmation(volumes) else 0
        
        return min(0.95, base_confidence + contact_bonus + volume_bonus)
    
    def _check_volume_confirmation(self, volumes: np.ndarray) -> bool:
        """Verifica confirmación de volumen"""
        if len(volumes) < 10:
            return False
        
        recent_avg = np.mean(volumes[-5:])
        historical_avg = np.mean(volumes[:-5])
        
        return recent_avg > historical_avg * self.volume_confirmation_threshold
    
    def _find_neckline(self, data: pd.DataFrame, left_shoulder: int, 
                      head: int, right_shoulder: int) -> Optional[float]:
        """Encuentra nivel de línea de cuello"""
        # Buscar mínimos entre hombros y cabeza
        left_valley_data = data.iloc[left_shoulder:head]
        right_valley_data = data.iloc[head:right_shoulder]
        
        if len(left_valley_data) == 0 or len(right_valley_data) == 0:
            return None
        
        left_valley = left_valley_data['low'].min()
        right_valley = right_valley_data['low'].min()
        
        # Línea de cuello es el promedio de los valles
        return (left_valley + right_valley) / 2
    
    def _calculate_cup_symmetry(self, lows: np.ndarray) -> float:
        """Calcula simetría de la copa"""
        if len(lows) < 10:
            return 0.0
        
        # Dividir en dos mitades
        mid_point = len(lows) // 2
        left_half = lows[:mid_point]
        right_half = lows[mid_point:]
        
        # Calcular correlación entre mitades (invertida la derecha)
        if len(left_half) != len(right_half):
            min_len = min(len(left_half), len(right_half))
            left_half = left_half[-min_len:]
            right_half = right_half[:min_len]
        
        right_half_reversed = right_half[::-1]
        
        correlation = np.corrcoef(left_half, right_half_reversed)[0, 1]
        
        return max(0.0, correlation) if not np.isnan(correlation) else 0.0
    
    def _remove_overlapping_patterns(self, patterns: List[DetectedPattern]) -> List[DetectedPattern]:
        """Elimina patrones superpuestos, manteniendo los de mayor confianza"""
        if len(patterns) <= 1:
            return patterns
        
        # Ordenar por confianza (descendente)
        sorted_patterns = sorted(patterns, key=lambda p: p.confidence, reverse=True)
        
        filtered_patterns = []
        
        for pattern in sorted_patterns:
            overlaps = False
            
            for existing in filtered_patterns:
                # Verificar superposición
                if (pattern.start_index < existing.end_index and 
                    pattern.end_index > existing.start_index):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_patterns.append(pattern)
        
        return filtered_patterns


# Función de conveniencia
def detect_patterns(data: pd.DataFrame, symbol: str, timeframe: str) -> List[DetectedPattern]:
    """
    Función de conveniencia para detección de patrones
    
    Args:
        data: DataFrame con datos OHLCV
        symbol: Símbolo del activo
        timeframe: Timeframe de los datos
        
    Returns:
        Lista de patrones detectados
    """
    detector = PatternDetector()
    return detector.detect_patterns(data, symbol, timeframe)