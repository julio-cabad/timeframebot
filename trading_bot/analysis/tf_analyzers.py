"""
Analizadores Especializados por Timeframe
========================================

Este módulo implementa analizadores especializados para cada timeframe del sistema de trading.
Cada analizador tiene un propósito específico en la jerarquía de análisis multi-timeframe:

- 1D (Daily): Tendencia principal y contexto macro
- 4H (4-Hour): Momentum intermedio y cambios de tendencia  
- 1H (1-Hour): Setups de entrada y zonas de valor
- 15M (15-Minute): Timing preciso y micro-estructura

Cada timeframe responde preguntas específicas que se combinan para formar
una visión completa del mercado y identificar oportunidades de alta probabilidad.

Autor: Sistema de Trading Algorítmico
Zona Horaria: UTC-5 (Ecuador)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import pytz

from ..config.settings import config
from ..utils.logger import get_logger, LogContext
from ..utils.exceptions import (
    AnalysisException,
    IndicatorException,
    ErrorCodes,
    create_exception
)

# Zona horaria de Ecuador (UTC-5)
ECUADOR_TZ = pytz.timezone('America/Guayaquil')

class MarketStructure(Enum):
    """Estructura del mercado"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGING = "ranging"
    CONSOLIDATION = "consolidation"

class PatternType(Enum):
    """Tipos de patrones detectados"""
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    TRIANGLE = "triangle"
    CHANNEL = "channel"
    SUPPORT_RESISTANCE = "support_resistance"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"

@dataclass
class TimeframeAnalysis:
    """Resultado del análisis de un timeframe específico"""
    timeframe: str
    timestamp: datetime
    
    # Tendencia
    trend_direction: str  # "bullish", "bearish", "neutral"
    trend_strength: float  # 0.0 a 1.0
    trend_duration_candles: int
    
    # Momentum
    momentum_score: float  # -1.0 a 1.0
    momentum_divergence: bool
    
    # Estructura
    market_structure: MarketStructure
    structure_break: bool
    
    # Niveles clave
    support_levels: List[float]
    resistance_levels: List[float]
    current_level_type: str  # "support", "resistance", "neutral"
    
    # Patrones
    patterns: List[PatternType]
    pattern_confidence: float  # 0.0 a 1.0
    
    # Indicadores técnicos
    rsi: float
    macd_signal: str  # "bullish", "bearish", "neutral"
    ema_alignment: bool
    
    # Volumen
    volume_trend: str  # "increasing", "decreasing", "stable"
    volume_confirmation: bool
    
    # Scores
    overall_score: float  # 0.0 a 100.0
    confidence: float  # 0.0 a 1.0
    
    # Contexto adicional
    volatility: float
    liquidity_score: float
    market_hours: str  # "active", "low", "closed"

class TechnicalIndicators:
    """Calculadora de indicadores técnicos optimizada para trading"""
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    @staticmethod
    def support_resistance_levels(data: pd.DataFrame, lookback: int = 50) -> Tuple[List[float], List[float]]:
        """Detecta niveles de soporte y resistencia usando pivots"""
        if len(data) < lookback:
            return [], []
        
        recent_data = data.tail(lookback)
        high = recent_data['high']
        low = recent_data['low']
        
        supports = []
        resistances = []
        
        # Buscar pivots con ventana de 5 períodos
        for i in range(5, len(recent_data) - 5):
            # Pivot alto (resistencia)
            if (high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i+1] and
                high.iloc[i] > high.iloc[i-2] and high.iloc[i] > high.iloc[i+2] and
                high.iloc[i] > high.iloc[i-3] and high.iloc[i] > high.iloc[i+3]):
                resistances.append(high.iloc[i])
            
            # Pivot bajo (soporte)
            if (low.iloc[i] < low.iloc[i-1] and low.iloc[i] < low.iloc[i+1] and
                low.iloc[i] < low.iloc[i-2] and low.iloc[i] < low.iloc[i+2] and
                low.iloc[i] < low.iloc[i-3] and low.iloc[i] < low.iloc[i+3]):
                supports.append(low.iloc[i])
        
        # Filtrar niveles más relevantes (últimos 5 de cada tipo)
        return supports[-5:], resistances[-5:]

class BaseTimeframeAnalyzer(ABC):
    """
    Clase base para todos los analizadores de timeframe
    Define la interfaz común y funcionalidades compartidas
    """
    
    def __init__(self, timeframe: str):
        self.timeframe = timeframe
        self.logger = get_logger(f"{self.__class__.__name__}")
        self.indicators = TechnicalIndicators()
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame, symbol: str) -> TimeframeAnalysis:
        """Método principal de análisis - debe ser implementado por cada subclase"""
        pass
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Valida que los datos sean suficientes para el análisis"""
        if data.empty:
            raise AnalysisException(
                f"No hay datos para analizar en {self.timeframe}",
                ErrorCodes.ANALYSIS_INSUFFICIENT_DATA
            )
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise AnalysisException(
                f"Columnas faltantes en datos: {missing_columns}",
                ErrorCodes.ANALYSIS_INSUFFICIENT_DATA
            )
        
        if len(data) < 20:
            raise AnalysisException(
                f"Datos insuficientes para análisis: {len(data)} velas (mínimo 20)",
                ErrorCodes.ANALYSIS_INSUFFICIENT_DATA
            )
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> Tuple[str, float]:
        """Calcula dirección y fuerza de la tendencia"""
        close = data['close']
        
        # EMAs para determinar tendencia
        ema_20 = self.indicators.ema(close, 20)
        ema_50 = self.indicators.ema(close, 50)
        
        current_price = close.iloc[-1]
        current_ema_20 = ema_20.iloc[-1]
        current_ema_50 = ema_50.iloc[-1]
        
        # Determinar dirección
        if current_price > current_ema_20 > current_ema_50:
            direction = "bullish"
        elif current_price < current_ema_20 < current_ema_50:
            direction = "bearish"
        else:
            direction = "neutral"
        
        # Calcular fuerza basada en pendiente de EMA y distancia
        ema_20_slope = (ema_20.iloc[-1] - ema_20.iloc[-10]) / ema_20.iloc[-10]
        price_distance = abs(current_price - current_ema_20) / current_ema_20
        
        strength = min(1.0, abs(ema_20_slope) * 10 + price_distance * 5)
        
        return direction, strength
    
    def _calculate_momentum(self, data: pd.DataFrame) -> Tuple[float, bool]:
        """Calcula momentum y detecta divergencias"""
        close = data['close']
        
        # RSI para momentum
        rsi = self.indicators.rsi(close)
        current_rsi = rsi.iloc[-1]
        
        # Normalizar RSI a rango -1 a 1
        momentum_score = (current_rsi - 50) / 50
        
        # Detectar divergencia simple
        price_trend = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]
        rsi_trend = (rsi.iloc[-1] - rsi.iloc[-10]) / rsi.iloc[-10]
        
        # Divergencia si precio y RSI van en direcciones opuestas
        divergence = (price_trend > 0 and rsi_trend < 0) or (price_trend < 0 and rsi_trend > 0)
        
        return momentum_score, divergence
    
    def _detect_market_structure(self, data: pd.DataFrame) -> Tuple[MarketStructure, bool]:
        """Detecta estructura del mercado y rupturas"""
        if len(data) < 20:
            return MarketStructure.RANGING, False
        
        # Analizar últimos 15 períodos
        recent_data = data.tail(15)
        highs = recent_data['high']
        lows = recent_data['low']
        
        # Calcular tendencias de máximos y mínimos
        high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
        
        # Determinar estructura
        if high_slope > 0 and low_slope > 0:
            structure = MarketStructure.UPTREND
        elif high_slope < 0 and low_slope < 0:
            structure = MarketStructure.DOWNTREND
        else:
            structure = MarketStructure.RANGING
        
        # Detectar ruptura de estructura (cambio en últimas 5 velas)
        last_5_high = highs.tail(5).max()
        last_5_low = lows.tail(5).min()
        prev_10_high = highs.iloc[-15:-5].max()
        prev_10_low = lows.iloc[-15:-5].min()
        
        structure_break = (last_5_high > prev_10_high * 1.02 or 
                          last_5_low < prev_10_low * 0.98)
        
        return structure, structure_break
    
    def _determine_current_level(self, current_price: float, 
                                supports: List[float], 
                                resistances: List[float]) -> str:
        """Determina si el precio actual está en soporte, resistencia o neutral"""
        tolerance = 0.01  # 1% de tolerancia
        
        # Verificar proximidad a resistencias
        for resistance in resistances:
            if abs(current_price - resistance) / resistance < tolerance:
                return "resistance"
        
        # Verificar proximidad a soportes
        for support in supports:
            if abs(current_price - support) / support < tolerance:
                return "support"
        
        return "neutral"
    
    def _check_ema_alignment(self, data: pd.DataFrame, trend_direction: str) -> bool:
        """Verifica alineación de EMAs con la tendencia"""
        close = data['close']
        
        ema_9 = self.indicators.ema(close, 9)
        ema_21 = self.indicators.ema(close, 21)
        ema_50 = self.indicators.ema(close, 50)
        
        current_9 = ema_9.iloc[-1]
        current_21 = ema_21.iloc[-1]
        current_50 = ema_50.iloc[-1]
        
        if trend_direction == "bullish":
            return current_9 > current_21 > current_50
        elif trend_direction == "bearish":
            return current_9 < current_21 < current_50
        else:
            return False
    
    def _analyze_volume(self, data: pd.DataFrame) -> Tuple[str, bool]:
        """Analiza tendencia de volumen y confirmación"""
        volume = data['volume']
        close = data['close']
        
        # Tendencia de volumen (últimas 10 velas)
        recent_volume = volume.tail(10)
        volume_slope = np.polyfit(range(len(recent_volume)), recent_volume, 1)[0]
        avg_volume = recent_volume.mean()
        
        if volume_slope > avg_volume * 0.1:
            volume_trend = "increasing"
        elif volume_slope < -avg_volume * 0.1:
            volume_trend = "decreasing"
        else:
            volume_trend = "stable"
        
        # Confirmación de volumen (volumen actual vs promedio)
        current_volume = volume.iloc[-1]
        avg_volume_20 = volume.tail(20).mean()
        volume_confirmation = current_volume > avg_volume_20 * 1.2
        
        return volume_trend, volume_confirmation
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calcula volatilidad usando ATR normalizado"""
        atr = self.indicators.atr(data['high'], data['low'], data['close'])
        current_atr = atr.iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # Volatilidad como porcentaje del precio
        volatility = current_atr / current_price if current_price > 0 else 0.0
        
        return volatility
    
    def _calculate_trend_duration(self, data: pd.DataFrame) -> int:
        """Calcula duración de la tendencia actual en velas"""
        close = data['close']
        ema_20 = self.indicators.ema(close, 20)
        
        # Contar velas consecutivas por encima/debajo de EMA
        duration = 0
        current_above = close.iloc[-1] > ema_20.iloc[-1]
        
        for i in range(len(data) - 1, 0, -1):
            if i >= len(ema_20):
                continue
                
            price_above = close.iloc[i] > ema_20.iloc[i]
            if price_above == current_above:
                duration += 1
            else:
                break
        
        return duration
    
    def _get_market_hours_status(self) -> str:
        """Determina el estado de las horas de mercado"""
        # Para crypto, el mercado está siempre abierto
        # Pero podemos considerar horas de mayor/menor actividad
        now = datetime.now(ECUADOR_TZ)
        hour = now.hour
        
        # Horas de mayor actividad (coinciden con mercados tradicionales)
        if 8 <= hour <= 16:  # 8 AM - 4 PM Ecuador
            return "active"
        elif 17 <= hour <= 23 or 0 <= hour <= 2:  # Mercados asiáticos/europeos
            return "active"
        else:
            return "low"


class DailyAnalyzer(BaseTimeframeAnalyzer):
    """
    Analizador para timeframe diario (1D)
    
    Propósito: Contexto macro y tendencia principal
    - ¿Cuál es la tendencia principal del mercado?
    - ¿Estamos en un mercado alcista o bajista?
    - ¿Qué niveles macro son importantes?
    
    Este timeframe define el CONTEXTO GENERAL del mercado
    """
    
    def __init__(self):
        super().__init__("1d")
    
    def analyze(self, data: pd.DataFrame, symbol: str) -> TimeframeAnalysis:
        """
        Análisis específico para timeframe diario
        Enfoque en tendencia macro y contexto general
        """
        context = LogContext(
            component="daily_analyzer",
            symbol=symbol,
            timeframe="1d"
        )
        
        self.logger.debug(f"Iniciando análisis 1D para {symbol}", context=context)
        
        try:
            self._validate_data(data)
            
            # Análisis de tendencia macro
            trend_direction, trend_strength = self._calculate_trend_strength(data)
            
            # Momentum a largo plazo
            momentum_score, momentum_divergence = self._calculate_momentum(data)
            
            # Estructura macro del mercado
            market_structure, structure_break = self._detect_market_structure(data)
            
            # Niveles macro importantes
            supports, resistances = self.indicators.support_resistance_levels(data, lookback=100)
            current_price = data['close'].iloc[-1]
            current_level_type = self._determine_current_level(current_price, supports, resistances)
            
            # Patrones macro
            patterns, pattern_confidence = self._detect_macro_patterns(data)
            
            # Indicadores a largo plazo
            rsi = self.indicators.rsi(data['close'], period=21).iloc[-1]  # RSI más suave para 1D
            macd_line, signal_line, _ = self.indicators.macd(data['close'], fast=12, slow=26, signal=9)
            macd_signal = self._analyze_macro_macd(macd_line, signal_line)
            
            # Alineación de EMAs macro
            ema_alignment = self._check_ema_alignment(data, trend_direction)
            
            # Análisis de volumen macro
            volume_trend, volume_confirmation = self._analyze_volume(data)
            
            # Volatilidad macro
            volatility = self._calculate_volatility(data)
            
            # Score general para 1D (tendencia + estructura)
            overall_score = self._calculate_daily_score(
                trend_strength, momentum_score, market_structure,
                pattern_confidence, volume_confirmation
            )
            
            # Confianza basada en claridad de tendencia macro
            confidence = self._calculate_macro_confidence(
                trend_strength, pattern_confidence, ema_alignment, rsi
            )
            
            return TimeframeAnalysis(
                timeframe="1d",
                timestamp=datetime.now(ECUADOR_TZ),
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                trend_duration_candles=self._calculate_trend_duration(data),
                momentum_score=momentum_score,
                momentum_divergence=momentum_divergence,
                market_structure=market_structure,
                structure_break=structure_break,
                support_levels=supports,
                resistance_levels=resistances,
                current_level_type=current_level_type,
                patterns=patterns,
                pattern_confidence=pattern_confidence,
                rsi=rsi,
                macd_signal=macd_signal,
                ema_alignment=ema_alignment,
                volume_trend=volume_trend,
                volume_confirmation=volume_confirmation,
                overall_score=overall_score,
                confidence=confidence,
                volatility=volatility,
                liquidity_score=1.0,  # Máxima liquidez en 1D
                market_hours=self._get_market_hours_status()
            )
            
        except Exception as e:
            self.logger.error(f"Error en análisis 1D: {str(e)}", context=context)
            raise AnalysisException(
                f"Fallo en análisis 1D: {str(e)}",
                ErrorCodes.ANALYSIS_CALCULATION_FAILED
            )
    
    def _detect_macro_patterns(self, data: pd.DataFrame) -> Tuple[List[PatternType], float]:
        """Detecta patrones macro en timeframe diario"""
        patterns = []
        confidence = 0.0
        
        if len(data) < 50:
            return patterns, confidence
        
        # Detectar tendencias macro
        trend_pattern, trend_conf = self._detect_macro_trend_pattern(data)
        if trend_pattern:
            patterns.append(trend_pattern)
            confidence = max(confidence, trend_conf)
        
        # Detectar consolidaciones macro
        consolidation_pattern, consol_conf = self._detect_macro_consolidation(data)
        if consolidation_pattern:
            patterns.append(consolidation_pattern)
            confidence = max(confidence, consol_conf)
        
        return patterns, confidence
    
    def _detect_macro_trend_pattern(self, data: pd.DataFrame) -> Tuple[Optional[PatternType], float]:
        """Detecta patrones de tendencia macro"""
        if len(data) < 30:
            return None, 0.0
        
        close = data['close']
        
        # Analizar últimos 30 días
        recent_close = close.tail(30)
        
        # Calcular pendiente de la tendencia
        x = np.arange(len(recent_close))
        slope, _ = np.polyfit(x, recent_close, 1)
        
        # Calcular R² para medir fuerza de la tendencia
        y_pred = slope * x + recent_close.iloc[0]
        ss_res = np.sum((recent_close - y_pred) ** 2)
        ss_tot = np.sum((recent_close - recent_close.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Determinar patrón basado en pendiente y R²
        if slope > 0 and r_squared > 0.7:
            return PatternType.BULL_FLAG, r_squared
        elif slope < 0 and r_squared > 0.7:
            return PatternType.BEAR_FLAG, r_squared
        
        return None, 0.0
    
    def _detect_macro_consolidation(self, data: pd.DataFrame) -> Tuple[Optional[PatternType], float]:
        """Detecta consolidaciones macro"""
        if len(data) < 20:
            return None, 0.0
        
        # Analizar últimos 20 días
        recent_data = data.tail(20)
        high_max = recent_data['high'].max()
        low_min = recent_data['low'].min()
        
        # Calcular rango de consolidación
        range_pct = (high_max - low_min) / low_min
        
        # Si el rango es pequeño, es consolidación
        if range_pct < 0.15:  # Menos del 15% de rango
            return PatternType.TRIANGLE, 0.8
        
        return None, 0.0
    
    def _analyze_macro_macd(self, macd_line: pd.Series, signal_line: pd.Series) -> str:
        """Análisis de MACD para contexto macro"""
        if len(macd_line) < 10:
            return "neutral"
        
        # Analizar últimos 5 días
        recent_macd = macd_line.tail(5)
        recent_signal = signal_line.tail(5)
        
        # Tendencia del MACD
        macd_trend = recent_macd.iloc[-1] - recent_macd.iloc[0]
        
        # Posición relativa a línea de señal
        above_signal = recent_macd.iloc[-1] > recent_signal.iloc[-1]
        
        if macd_trend > 0 and above_signal:
            return "bullish"
        elif macd_trend < 0 and not above_signal:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_daily_score(self, trend_strength: float, momentum_score: float,
                              market_structure: MarketStructure, pattern_confidence: float,
                              volume_confirmation: bool) -> float:
        """Calcula score específico para timeframe diario"""
        score = 0.0
        
        # Fuerza de tendencia (40% del score en 1D)
        score += trend_strength * 40
        
        # Estructura del mercado (25% del score)
        if market_structure in [MarketStructure.UPTREND, MarketStructure.DOWNTREND]:
            score += 25
        elif market_structure == MarketStructure.RANGING:
            score += 10
        
        # Momentum (20% del score)
        score += abs(momentum_score) * 20
        
        # Patrones macro (10% del score)
        score += pattern_confidence * 10
        
        # Confirmación de volumen (5% del score)
        if volume_confirmation:
            score += 5
        
        return min(100.0, score)
    
    def _calculate_macro_confidence(self, trend_strength: float, pattern_confidence: float,
                                   ema_alignment: bool, rsi: float) -> float:
        """Calcula confianza para análisis macro"""
        confidence = 0.2  # Base
        
        # Tendencia fuerte aumenta confianza
        confidence += trend_strength * 0.4
        
        # Patrones claros aumentan confianza
        confidence += pattern_confidence * 0.2
        
        # Alineación de EMAs
        if ema_alignment:
            confidence += 0.15
        
        # RSI no extremo (más confiable)
        if 25 < rsi < 75:
            confidence += 0.05
        
        return max(0.0, min(1.0, confidence))


class FourHourAnalyzer(BaseTimeframeAnalyzer):
    """
    Analizador para timeframe de 4 horas (4H)
    
    Propósito: Momentum intermedio y cambios de tendencia
    - ¿Está cambiando el momentum?
    - ¿Hay señales de reversión o continuación?
    - ¿El momentum confirma la tendencia diaria?
    
    Este timeframe define los CAMBIOS DE MOMENTUM
    """
    
    def __init__(self):
        super().__init__("4h")
    
    def analyze(self, data: pd.DataFrame, symbol: str) -> TimeframeAnalysis:
        """
        Análisis específico para timeframe de 4 horas
        Enfoque en momentum y cambios de tendencia
        """
        context = LogContext(
            component="fourhour_analyzer",
            symbol=symbol,
            timeframe="4h"
        )
        
        self.logger.debug(f"Iniciando análisis 4H para {symbol}", context=context)
        
        try:
            self._validate_data(data)
            
            # Análisis de momentum intermedio
            trend_direction, trend_strength = self._calculate_trend_strength(data)
            
            # Momentum más sensible
            momentum_score, momentum_divergence = self._calculate_momentum(data)
            
            # Estructura intermedia
            market_structure, structure_break = self._detect_market_structure(data)
            
            # Niveles intermedios
            supports, resistances = self.indicators.support_resistance_levels(data, lookback=75)
            current_price = data['close'].iloc[-1]
            current_level_type = self._determine_current_level(current_price, supports, resistances)
            
            # Patrones de momentum
            patterns, pattern_confidence = self._detect_momentum_patterns(data)
            
            # Indicadores de momentum
            rsi = self.indicators.rsi(data['close'], period=14).iloc[-1]
            macd_line, signal_line, _ = self.indicators.macd(data['close'])
            macd_signal = self._analyze_momentum_macd(macd_line, signal_line)
            
            # Alineación para momentum
            ema_alignment = self._check_ema_alignment(data, trend_direction)
            
            # Análisis de volumen intermedio
            volume_trend, volume_confirmation = self._analyze_volume(data)
            
            # Detección de cambios de momentum
            momentum_change = self._detect_momentum_change(data)
            
            # Volatilidad intermedia
            volatility = self._calculate_volatility(data)
            
            # Score para 4H (momentum + cambios)
            overall_score = self._calculate_fourhour_score(
                momentum_score, momentum_change, pattern_confidence,
                trend_strength, volume_confirmation
            )
            
            # Confianza basada en claridad de momentum
            confidence = self._calculate_momentum_confidence(
                momentum_score, momentum_divergence, pattern_confidence,
                momentum_change, rsi
            )
            
            return TimeframeAnalysis(
                timeframe="4h",
                timestamp=datetime.now(ECUADOR_TZ),
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                trend_duration_candles=self._calculate_trend_duration(data),
                momentum_score=momentum_score,
                momentum_divergence=momentum_divergence,
                market_structure=market_structure,
                structure_break=structure_break,
                support_levels=supports,
                resistance_levels=resistances,
                current_level_type=current_level_type,
                patterns=patterns,
                pattern_confidence=pattern_confidence,
                rsi=rsi,
                macd_signal=macd_signal,
                ema_alignment=ema_alignment,
                volume_trend=volume_trend,
                volume_confirmation=volume_confirmation,
                overall_score=overall_score,
                confidence=confidence,
                volatility=volatility,
                liquidity_score=0.9,  # Buena liquidez en 4H
                market_hours=self._get_market_hours_status()
            )
            
        except Exception as e:
            self.logger.error(f"Error en análisis 4H: {str(e)}", context=context)
            raise AnalysisException(
                f"Fallo en análisis 4H: {str(e)}",
                ErrorCodes.ANALYSIS_CALCULATION_FAILED
            )
    
    def _detect_momentum_patterns(self, data: pd.DataFrame) -> Tuple[List[PatternType], float]:
        """Detecta patrones específicos de momentum en 4H"""
        patterns = []
        confidence = 0.0
        
        if len(data) < 40:
            return patterns, confidence
        
        # Detectar reversiones de momentum
        reversal_pattern, reversal_conf = self._detect_momentum_reversal(data)
        if reversal_pattern:
            patterns.append(reversal_pattern)
            confidence = max(confidence, reversal_conf)
        
        # Detectar continuaciones de momentum
        continuation_pattern, cont_conf = self._detect_momentum_continuation(data)
        if continuation_pattern:
            patterns.append(continuation_pattern)
            confidence = max(confidence, cont_conf)
        
        return patterns, confidence
    
    def _detect_momentum_reversal(self, data: pd.DataFrame) -> Tuple[Optional[PatternType], float]:
        """Detecta reversiones de momentum"""
        if len(data) < 20:
            return None, 0.0
        
        close = data['close']
        rsi = self.indicators.rsi(close)
        
        # Buscar divergencias RSI-Precio
        recent_close = close.tail(10)
        recent_rsi = rsi.tail(10)
        
        # Tendencia de precio vs RSI
        price_trend = (recent_close.iloc[-1] - recent_close.iloc[0]) / recent_close.iloc[0]
        rsi_trend = recent_rsi.iloc[-1] - recent_rsi.iloc[0]
        
        # Divergencia bajista: precio sube, RSI baja
        if price_trend > 0.02 and rsi_trend < -5:
            return PatternType.REVERSAL, 0.7
        
        # Divergencia alcista: precio baja, RSI sube
        if price_trend < -0.02 and rsi_trend > 5:
            return PatternType.REVERSAL, 0.7
        
        return None, 0.0
    
    def _detect_momentum_continuation(self, data: pd.DataFrame) -> Tuple[Optional[PatternType], float]:
        """Detecta continuaciones de momentum"""
        if len(data) < 15:
            return None, 0.0
        
        close = data['close']
        volume = data['volume']
        
        # Analizar últimas 10 velas
        recent_close = close.tail(10)
        recent_volume = volume.tail(10)
        
        # Momentum consistente
        price_changes = recent_close.diff().dropna()
        positive_moves = (price_changes > 0).sum()
        negative_moves = (price_changes < 0).sum()
        
        # Volumen creciente
        volume_trend = np.polyfit(range(len(recent_volume)), recent_volume, 1)[0]
        
        # Continuación alcista
        if positive_moves >= 7 and volume_trend > 0:
            return PatternType.BULL_FLAG, 0.8
        
        # Continuación bajista
        if negative_moves >= 7 and volume_trend > 0:
            return PatternType.BEAR_FLAG, 0.8
        
        return None, 0.0
    
    def _analyze_momentum_macd(self, macd_line: pd.Series, signal_line: pd.Series) -> str:
        """Análisis de MACD específico para momentum"""
        if len(macd_line) < 5:
            return "neutral"
        
        # Analizar últimas 3 velas para momentum
        recent_macd = macd_line.tail(3)
        recent_signal = signal_line.tail(3)
        
        # Cruce reciente
        if (recent_macd.iloc[-1] > recent_signal.iloc[-1] and 
            recent_macd.iloc[-2] <= recent_signal.iloc[-2]):
            return "bullish"
        
        if (recent_macd.iloc[-1] < recent_signal.iloc[-1] and 
            recent_macd.iloc[-2] >= recent_signal.iloc[-2]):
            return "bearish"
        
        # Aceleración del momentum
        macd_acceleration = recent_macd.iloc[-1] - recent_macd.iloc[-2]
        if macd_acceleration > 0 and recent_macd.iloc[-1] > recent_signal.iloc[-1]:
            return "bullish"
        elif macd_acceleration < 0 and recent_macd.iloc[-1] < recent_signal.iloc[-1]:
            return "bearish"
        
        return "neutral"
    
    def _detect_momentum_change(self, data: pd.DataFrame) -> float:
        """Detecta cambios en el momentum (0.0 a 1.0)"""
        if len(data) < 15:
            return 0.0
        
        close = data['close']
        
        # Comparar momentum reciente vs anterior
        recent_momentum = (close.iloc[-5:].iloc[-1] - close.iloc[-5:].iloc[0]) / close.iloc[-5:].iloc[0]
        prev_momentum = (close.iloc[-10:-5].iloc[-1] - close.iloc[-10:-5].iloc[0]) / close.iloc[-10:-5].iloc[0]
        
        # Cambio en momentum
        momentum_change = abs(recent_momentum - prev_momentum)
        
        # Normalizar a 0-1
        return min(1.0, momentum_change * 10)
    
    def _calculate_fourhour_score(self, momentum_score: float, momentum_change: float,
                                 pattern_confidence: float, trend_strength: float,
                                 volume_confirmation: bool) -> float:
        """Calcula score específico para timeframe 4H"""
        score = 0.0
        
        # Momentum (35% del score en 4H)
        score += abs(momentum_score) * 35
        
        # Cambio de momentum (25% del score)
        score += momentum_change * 25
        
        # Patrones de momentum (20% del score)
        score += pattern_confidence * 20
        
        # Tendencia (15% del score)
        score += trend_strength * 15
        
        # Confirmación de volumen (5% del score)
        if volume_confirmation:
            score += 5
        
        return min(100.0, score)
    
    def _calculate_momentum_confidence(self, momentum_score: float, momentum_divergence: bool,
                                     pattern_confidence: float, momentum_change: float,
                                     rsi: float) -> float:
        """Calcula confianza para análisis de momentum"""
        confidence = 0.3  # Base
        
        # Momentum fuerte aumenta confianza
        confidence += abs(momentum_score) * 0.25
        
        # Cambio de momentum claro
        confidence += momentum_change * 0.2
        
        # Patrones claros
        confidence += pattern_confidence * 0.15
        
        # Divergencia reduce confianza (señal de cambio)
        if momentum_divergence:
            confidence -= 0.1
        
        # RSI en zona favorable
        if 30 < rsi < 70:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))


class OneHourAnalyzer(BaseTimeframeAnalyzer):
    """
    Analizador para timeframe de 1 hora (1H)
    
    Propósito: Identificar setups de entrada y patrones de continuación
    - ¿Hay un setup válido para entrar?
    - ¿El precio está en una zona de valor?
    - ¿Los patrones confirman la dirección?
    
    Este timeframe define DÓNDE entrar en la dirección de la tendencia
    """
    
    def __init__(self):
        super().__init__("1h")
    
    def analyze(self, data: pd.DataFrame, symbol: str) -> TimeframeAnalysis:
        """
        Análisis específico para timeframe de 1 hora
        Enfoque en setups de entrada y zonas de valor
        """
        context = LogContext(
            component="onehour_analyzer",
            symbol=symbol,
            timeframe="1h"
        )
        
        self.logger.debug(f"Iniciando análisis 1H para {symbol}", context=context)
        
        try:
            self._validate_data(data)
            
            # Análisis de tendencia
            trend_direction, trend_strength = self._calculate_trend_strength(data)
            
            # Momentum
            momentum_score, momentum_divergence = self._calculate_momentum(data)
            
            # Estructura del mercado
            market_structure, structure_break = self._detect_market_structure(data)
            
            # Niveles de soporte y resistencia (muy importantes para entradas)
            supports, resistances = self.indicators.support_resistance_levels(data, lookback=50)
            current_price = data['close'].iloc[-1]
            current_level_type = self._determine_current_level(current_price, supports, resistances)
            
            # Detección de patrones de entrada
            patterns, pattern_confidence = self._detect_entry_patterns(data)
            
            # Indicadores técnicos para timing
            rsi = self.indicators.rsi(data['close']).iloc[-1]
            macd_line, signal_line, _ = self.indicators.macd(data['close'])
            macd_signal = self._analyze_macd_timing(macd_line, signal_line)
            
            # Alineación de EMAs para confirmación
            ema_alignment = self._check_ema_alignment(data, trend_direction)
            
            # Análisis de volumen para confirmación
            volume_trend, volume_confirmation = self._analyze_volume(data)
            
            # Análisis de pullbacks y retrocesos
            pullback_quality = self._analyze_pullback_quality(data)
            
            # Volatilidad
            volatility = self._calculate_volatility(data)
            
            # Score general para 1H (setup quality + timing)
            overall_score = self._calculate_onehour_score(
                trend_strength, momentum_score, current_level_type,
                pattern_confidence, pullback_quality, volume_confirmation
            )
            
            # Confianza basada en confluencias de entrada
            confidence = self._calculate_entry_confidence(
                trend_strength, current_level_type, pattern_confidence,
                pullback_quality, ema_alignment, rsi
            )
            
            return TimeframeAnalysis(
                timeframe="1h",
                timestamp=datetime.now(ECUADOR_TZ),
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                trend_duration_candles=self._calculate_trend_duration(data),
                momentum_score=momentum_score,
                momentum_divergence=momentum_divergence,
                market_structure=market_structure,
                structure_break=structure_break,
                support_levels=supports,
                resistance_levels=resistances,
                current_level_type=current_level_type,
                patterns=patterns,
                pattern_confidence=pattern_confidence,
                rsi=rsi,
                macd_signal=macd_signal,
                ema_alignment=ema_alignment,
                volume_trend=volume_trend,
                volume_confirmation=volume_confirmation,
                overall_score=overall_score,
                confidence=confidence,
                volatility=volatility,
                liquidity_score=0.8,  # Buena liquidez en 1H
                market_hours=self._get_market_hours_status()
            )
            
        except Exception as e:
            self.logger.error(f"Error en análisis 1H: {str(e)}", context=context)
            raise AnalysisException(
                f"Fallo en análisis 1H: {str(e)}",
                ErrorCodes.ANALYSIS_CALCULATION_FAILED
            )
    
    def _detect_entry_patterns(self, data: pd.DataFrame) -> Tuple[List[PatternType], float]:
        """
        Detecta patrones específicos de entrada en 1H
        Enfoque en pullbacks, banderas y continuaciones
        """
        patterns = []
        confidence = 0.0
        
        if len(data) < 30:
            return patterns, confidence
        
        # Detectar pullback a EMA
        pullback_pattern, pullback_conf = self._detect_pullback_to_ema(data)
        if pullback_pattern:
            patterns.append(pullback_pattern)
            confidence = max(confidence, pullback_conf)
        
        # Detectar ruptura de canal
        channel_pattern, channel_conf = self._detect_channel_breakout(data)
        if channel_pattern:
            patterns.append(channel_pattern)
            confidence = max(confidence, channel_conf)
        
        # Detectar rebote en soporte/resistencia
        sr_pattern, sr_conf = self._detect_support_resistance_bounce(data)
        if sr_pattern:
            patterns.append(sr_pattern)
            confidence = max(confidence, sr_conf)
        
        return patterns, confidence
    
    def _detect_pullback_to_ema(self, data: pd.DataFrame) -> Tuple[Optional[PatternType], float]:
        """Detecta pullbacks a EMAs importantes"""
        if len(data) < 25:
            return None, 0.0
        
        close = data['close']
        ema_21 = self.indicators.ema(close, 21)
        
        current_price = close.iloc[-1]
        current_ema = ema_21.iloc[-1]
        
        # Verificar si estamos cerca de EMA21
        distance_to_ema = abs(current_price - current_ema) / current_ema
        
        if distance_to_ema < 0.01:  # Dentro del 1% de la EMA
            # Verificar dirección de la tendencia
            ema_slope = (ema_21.iloc[-1] - ema_21.iloc[-5]) / ema_21.iloc[-5]
            
            if ema_slope > 0.005:  # Tendencia alcista
                return PatternType.BULL_FLAG, 0.8
            elif ema_slope < -0.005:  # Tendencia bajista
                return PatternType.BEAR_FLAG, 0.8
        
        return None, 0.0
    
    def _detect_channel_breakout(self, data: pd.DataFrame) -> Tuple[Optional[PatternType], float]:
        """Detecta rupturas de canales"""
        if len(data) < 20:
            return None, 0.0
        
        recent_data = data.tail(15)
        high = recent_data['high']
        low = recent_data['low']
        close = recent_data['close']
        
        # Canal simple: máximo y mínimo de los últimos 10 períodos
        channel_high = high.iloc[:-2].max()  # Excluir las últimas 2 velas
        channel_low = low.iloc[:-2].min()
        
        current_price = close.iloc[-1]
        
        # Ruptura alcista
        if current_price > channel_high * 1.005:  # 0.5% por encima
            return PatternType.CHANNEL, 0.7
        
        # Ruptura bajista
        if current_price < channel_low * 0.995:  # 0.5% por debajo
            return PatternType.CHANNEL, 0.7
        
        return None, 0.0
    
    def _detect_support_resistance_bounce(self, data: pd.DataFrame) -> Tuple[Optional[PatternType], float]:
        """Detecta rebotes en niveles de soporte/resistencia"""
        if len(data) < 15:
            return None, 0.0
        
        supports, resistances = self.indicators.support_resistance_levels(data, lookback=30)
        current_price = data['close'].iloc[-1]
        previous_price = data['close'].iloc[-2]
        
        # Verificar rebote en soporte
        for support in supports:
            if (previous_price <= support * 1.01 and  # Tocó soporte
                current_price > support * 1.005):     # Rebotó
                return PatternType.SUPPORT_RESISTANCE, 0.75
        
        # Verificar rebote en resistencia
        for resistance in resistances:
            if (previous_price >= resistance * 0.99 and  # Tocó resistencia
                current_price < resistance * 0.995):     # Rebotó
                return PatternType.SUPPORT_RESISTANCE, 0.75
        
        return None, 0.0
    
    def _analyze_macd_timing(self, macd_line: pd.Series, signal_line: pd.Series) -> str:
        """Análisis de MACD específico para timing de entrada"""
        if len(macd_line) < 5:
            return "neutral"
        
        # Buscar cruces recientes y momentum
        recent_macd = macd_line.tail(3)
        recent_signal = signal_line.tail(3)
        
        # Cruce muy reciente (última vela)
        if (recent_macd.iloc[-1] > recent_signal.iloc[-1] and 
            recent_macd.iloc[-2] <= recent_signal.iloc[-2]):
            return "bullish"
        
        if (recent_macd.iloc[-1] < recent_signal.iloc[-1] and 
            recent_macd.iloc[-2] >= recent_signal.iloc[-2]):
            return "bearish"
        
        # Momentum creciente
        if recent_macd.iloc[-1] > recent_macd.iloc[-2] > recent_macd.iloc[-3]:
            return "bullish"
        
        if recent_macd.iloc[-1] < recent_macd.iloc[-2] < recent_macd.iloc[-3]:
            return "bearish"
        
        return "neutral"
    
    def _analyze_pullback_quality(self, data: pd.DataFrame) -> float:
        """
        Analiza la calidad de pullbacks para entradas
        Pullbacks de calidad son poco profundos y con bajo volumen
        """
        if len(data) < 20:
            return 0.0
        
        close = data['close']
        volume = data['volume']
        ema_21 = self.indicators.ema(close, 21)
        
        # Analizar últimos 10 períodos
        recent_data = data.tail(10)
        recent_close = recent_data['close']
        recent_volume = recent_data['volume']
        recent_ema = ema_21.tail(10)
        
        # Calcular profundidad del pullback
        if len(recent_close) < 5:
            return 0.0
        
        max_price = recent_close.max()
        min_price = recent_close.min()
        pullback_depth = (max_price - min_price) / max_price
        
        # Calcular tendencia de volumen durante pullback
        volume_trend = np.polyfit(range(len(recent_volume)), recent_volume, 1)[0]
        avg_volume = recent_volume.mean()
        volume_trend_normalized = volume_trend / avg_volume if avg_volume > 0 else 0
        
        # Pullback de calidad: poco profundo y volumen decreciente
        quality_score = 0.0
        
        # Profundidad (mejor si es menor)
        if pullback_depth < 0.02:  # Menos del 2%
            quality_score += 0.4
        elif pullback_depth < 0.05:  # Menos del 5%
            quality_score += 0.2
        
        # Volumen decreciente durante pullback
        if volume_trend_normalized < -0.1:
            quality_score += 0.3
        elif volume_trend_normalized < 0:
            quality_score += 0.1
        
        # Proximidad a EMA
        current_price = close.iloc[-1]
        current_ema = ema_21.iloc[-1]
        distance_to_ema = abs(current_price - current_ema) / current_ema
        
        if distance_to_ema < 0.01:  # Muy cerca de EMA
            quality_score += 0.3
        elif distance_to_ema < 0.02:  # Cerca de EMA
            quality_score += 0.2
        
        return min(1.0, quality_score)
    
    def _calculate_onehour_score(self, trend_strength: float, momentum_score: float,
                                current_level_type: str, pattern_confidence: float,
                                pullback_quality: float, volume_confirmation: bool) -> float:
        """
        Calcula score específico para timeframe 1H
        En 1H, la calidad del setup es lo más importante
        """
        score = 0.0
        
        # Calidad del pullback (30% del score en 1H)
        score += pullback_quality * 30
        
        # Patrones de entrada (25% del score)
        score += pattern_confidence * 25
        
        # Posición en niveles clave (20% del score)
        if current_level_type in ["support", "resistance"]:
            score += 20
        
        # Tendencia (15% del score)
        score += trend_strength * 15
        
        # Momentum (5% del score)
        score += abs(momentum_score) * 5
        
        # Confirmación de volumen (5% del score)
        if volume_confirmation:
            score += 5
        
        return min(100.0, score)
    
    def _calculate_entry_confidence(self, trend_strength: float, current_level_type: str,
                                   pattern_confidence: float, pullback_quality: float,
                                   ema_alignment: bool, rsi: float) -> float:
        """Calcula la confianza específica para entradas en 1H"""
        confidence = 0.3  # Base
        
        # Pullback de calidad aumenta mucho la confianza
        confidence += pullback_quality * 0.3
        
        # Patrones claros aumentan confianza
        confidence += pattern_confidence * 0.2
        
        # Estar en nivel clave aumenta confianza
        if current_level_type in ["support", "resistance"]:
            confidence += 0.2
        
        # Alineación de EMAs
        if ema_alignment:
            confidence += 0.15
        
        # RSI en zona favorable (no extremos)
        if 30 < rsi < 70:
            confidence += 0.1
        elif rsi < 25 or rsi > 75:
            confidence -= 0.1
        
        # Tendencia fuerte
        confidence += trend_strength * 0.1
        
        return max(0.0, min(1.0, confidence))


class FifteenMinuteAnalyzer(BaseTimeframeAnalyzer):
    """
    Analizador para timeframe de 15 minutos (15M)
    
    Propósito: Timing preciso y micro-estructura del mercado
    - ¿Cuál es el momento exacto para entrar?
    - ¿Hay confirmación en la micro-estructura?
    - ¿El precio está mostrando fuerza/debilidad?
    
    Este timeframe define el TIMING EXACTO de entrada
    """
    
    def __init__(self):
        super().__init__("15m")
    
    def analyze(self, data: pd.DataFrame, symbol: str) -> TimeframeAnalysis:
        """
        Análisis específico para timeframe de 15 minutos
        Enfoque en timing preciso y micro-estructura
        """
        context = LogContext(
            component="fifteenmin_analyzer",
            symbol=symbol,
            timeframe="15m"
        )
        
        self.logger.debug(f"Iniciando análisis 15M para {symbol}", context=context)
        
        try:
            self._validate_data(data)
            
            # Análisis de tendencia (corto plazo)
            trend_direction, trend_strength = self._calculate_trend_strength(data)
            
            # Momentum (muy sensible en 15M)
            momentum_score, momentum_divergence = self._calculate_momentum(data)
            
            # Micro-estructura del mercado
            market_structure, structure_break = self._detect_microstructure(data)
            
            # Niveles micro (más granulares)
            supports, resistances = self._detect_micro_levels(data)
            current_price = data['close'].iloc[-1]
            current_level_type = self._determine_current_level(current_price, supports, resistances)
            
            # Patrones de timing
            patterns, pattern_confidence = self._detect_timing_patterns(data)
            
            # Indicadores para timing preciso
            rsi = self.indicators.rsi(data['close'], period=10).iloc[-1]  # RSI más sensible
            macd_line, signal_line, _ = self.indicators.macd(data['close'], fast=8, slow=17, signal=6)
            macd_signal = self._analyze_micro_macd(macd_line, signal_line)
            
            # Micro-momentum
            micro_momentum = self._calculate_micro_momentum(data)
            
            # Análisis de volumen tick-by-tick
            volume_trend, volume_confirmation = self._analyze_micro_volume(data)
            
            # Price action reciente
            price_action_quality = self._analyze_price_action(data)
            
            # Volatilidad intraday
            volatility = self._calculate_intraday_volatility(data)
            
            # Score para 15M (timing + price action)
            overall_score = self._calculate_fifteenmin_score(
                momentum_score, micro_momentum, price_action_quality,
                pattern_confidence, volume_confirmation, current_level_type
            )
            
            # Confianza para timing
            confidence = self._calculate_timing_confidence(
                micro_momentum, price_action_quality, pattern_confidence,
                volume_confirmation, volatility
            )
            
            return TimeframeAnalysis(
                timeframe="15m",
                timestamp=datetime.now(ECUADOR_TZ),
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                trend_duration_candles=self._calculate_trend_duration(data),
                momentum_score=momentum_score,
                momentum_divergence=momentum_divergence,
                market_structure=market_structure,
                structure_break=structure_break,
                support_levels=supports,
                resistance_levels=resistances,
                current_level_type=current_level_type,
                patterns=patterns,
                pattern_confidence=pattern_confidence,
                rsi=rsi,
                macd_signal=macd_signal,
                ema_alignment=True,  # Menos relevante en 15M
                volume_trend=volume_trend,
                volume_confirmation=volume_confirmation,
                overall_score=overall_score,
                confidence=confidence,
                volatility=volatility,
                liquidity_score=self._calculate_liquidity_score(),
                market_hours=self._get_market_hours_status()
            )
            
        except Exception as e:
            self.logger.error(f"Error en análisis 15M: {str(e)}", context=context)
            raise AnalysisException(
                f"Fallo en análisis 15M: {str(e)}",
                ErrorCodes.ANALYSIS_CALCULATION_FAILED
            )
    
    def _detect_microstructure(self, data: pd.DataFrame) -> Tuple[MarketStructure, bool]:
        """
        Detecta micro-estructura del mercado
        Más sensible a cambios pequeños
        """
        if len(data) < 10:
            return MarketStructure.RANGING, False
        
        # Analizar últimos 8 períodos para micro-estructura
        recent_data = data.tail(8)
        highs = recent_data['high']
        lows = recent_data['low']
        close = recent_data['close']
        
        # Micro-tendencias
        high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
        close_slope = np.polyfit(range(len(close)), close, 1)[0]
        
        # Determinar micro-estructura
        if high_slope > 0 and low_slope > 0 and close_slope > 0:
            structure = MarketStructure.UPTREND
        elif high_slope < 0 and low_slope < 0 and close_slope < 0:
            structure = MarketStructure.DOWNTREND
        else:
            structure = MarketStructure.RANGING
        
        # Micro-ruptura (cambio en las últimas 3 velas)
        last_3_close = close.tail(3)
        price_change = (last_3_close.iloc[-1] - last_3_close.iloc[0]) / last_3_close.iloc[0]
        structure_break = abs(price_change) > 0.005  # 0.5% en 3 velas
        
        return structure, structure_break
    
    def _detect_micro_levels(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """
        Detecta niveles micro de soporte y resistencia
        Más granular que timeframes mayores
        """
        if len(data) < 20:
            return [], []
        
        # Usar ventana más pequeña para niveles micro
        recent_data = data.tail(30)
        high = recent_data['high']
        low = recent_data['low']
        
        # Pivots con ventana más pequeña
        supports = []
        resistances = []
        
        for i in range(2, len(recent_data) - 2):
            # Pivot alto (resistencia)
            if (high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i+1] and
                high.iloc[i] > high.iloc[i-2] and high.iloc[i] > high.iloc[i+2]):
                resistances.append(high.iloc[i])
            
            # Pivot bajo (soporte)
            if (low.iloc[i] < low.iloc[i-1] and low.iloc[i] < low.iloc[i+1] and
                low.iloc[i] < low.iloc[i-2] and low.iloc[i] < low.iloc[i+2]):
                supports.append(low.iloc[i])
        
        # Filtrar niveles más relevantes
        current_price = data['close'].iloc[-1]
        
        # Solo niveles cercanos (dentro del 1%)
        supports = [s for s in supports if abs(s - current_price) / current_price < 0.01]
        resistances = [r for r in resistances if abs(r - current_price) / current_price < 0.01]
        
        return supports[-3:], resistances[-3:]  # Máximo 3 de cada tipo
    
    def _detect_timing_patterns(self, data: pd.DataFrame) -> Tuple[List[PatternType], float]:
        """
        Detecta patrones específicos de timing en 15M
        Enfoque en micro-patrones y price action
        """
        patterns = []
        confidence = 0.0
        
        if len(data) < 15:
            return patterns, confidence
        
        # Detectar engulfing patterns
        engulfing_pattern, engulfing_conf = self._detect_engulfing_pattern(data)
        if engulfing_pattern:
            patterns.append(engulfing_pattern)
            confidence = max(confidence, engulfing_conf)
        
        # Detectar pin bars
        pinbar_pattern, pinbar_conf = self._detect_pinbar_pattern(data)
        if pinbar_pattern:
            patterns.append(pinbar_pattern)
            confidence = max(confidence, pinbar_conf)
        
        # Detectar inside bars
        inside_pattern, inside_conf = self._detect_inside_bar_pattern(data)
        if inside_pattern:
            patterns.append(inside_pattern)
            confidence = max(confidence, inside_conf)
        
        return patterns, confidence
    
    def _detect_engulfing_pattern(self, data: pd.DataFrame) -> Tuple[Optional[PatternType], float]:
        """Detecta patrones envolventes"""
        if len(data) < 2:
            return None, 0.0
        
        # Últimas dos velas
        prev_candle = data.iloc[-2]
        curr_candle = data.iloc[-1]
        
        prev_body = abs(prev_candle['close'] - prev_candle['open'])
        curr_body = abs(curr_candle['close'] - curr_candle['open'])
        
        # Engulfing alcista
        if (prev_candle['close'] < prev_candle['open'] and  # Vela anterior bajista
            curr_candle['close'] > curr_candle['open'] and  # Vela actual alcista
            curr_candle['open'] < prev_candle['close'] and  # Abre por debajo del cierre anterior
            curr_candle['close'] > prev_candle['open'] and  # Cierra por encima de la apertura anterior
            curr_body > prev_body * 1.5):  # Cuerpo más grande
            return PatternType.BULL_FLAG, 0.8
        
        # Engulfing bajista
        if (prev_candle['close'] > prev_candle['open'] and  # Vela anterior alcista
            curr_candle['close'] < curr_candle['open'] and  # Vela actual bajista
            curr_candle['open'] > prev_candle['close'] and  # Abre por encima del cierre anterior
            curr_candle['close'] < prev_candle['open'] and  # Cierra por debajo de la apertura anterior
            curr_body > prev_body * 1.5):  # Cuerpo más grande
            return PatternType.BEAR_FLAG, 0.8
        
        return None, 0.0
    
    def _detect_pinbar_pattern(self, data: pd.DataFrame) -> Tuple[Optional[PatternType], float]:
        """Detecta pin bars (velas con mechas largas)"""
        if len(data) < 1:
            return None, 0.0
        
        candle = data.iloc[-1]
        
        body_size = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        
        if total_range == 0:
            return None, 0.0
        
        # Pin bar alcista (mecha inferior larga)
        if (lower_wick > body_size * 2 and  # Mecha inferior > 2x el cuerpo
            lower_wick > total_range * 0.6 and  # Mecha inferior > 60% del rango
            upper_wick < body_size * 0.5):  # Mecha superior pequeña
            return PatternType.SUPPORT_RESISTANCE, 0.7
        
        # Pin bar bajista (mecha superior larga)
        if (upper_wick > body_size * 2 and  # Mecha superior > 2x el cuerpo
            upper_wick > total_range * 0.6 and  # Mecha superior > 60% del rango
            lower_wick < body_size * 0.5):  # Mecha inferior pequeña
            return PatternType.SUPPORT_RESISTANCE, 0.7
        
        return None, 0.0
    
    def _detect_inside_bar_pattern(self, data: pd.DataFrame) -> Tuple[Optional[PatternType], float]:
        """Detecta inside bars (consolidación)"""
        if len(data) < 2:
            return None, 0.0
        
        prev_candle = data.iloc[-2]
        curr_candle = data.iloc[-1]
        
        # Inside bar: vela actual completamente dentro de la anterior
        if (curr_candle['high'] < prev_candle['high'] and
            curr_candle['low'] > prev_candle['low']):
            return PatternType.TRIANGLE, 0.6
        
        return None, 0.0
    
    def _analyze_micro_macd(self, macd_line: pd.Series, signal_line: pd.Series) -> str:
        """Análisis de MACD para micro-timing"""
        if len(macd_line) < 3:
            return "neutral"
        
        # Valores muy recientes
        macd_current = macd_line.iloc[-1]
        signal_current = signal_line.iloc[-1]
        macd_prev = macd_line.iloc[-2]
        
        # Momentum creciente por encima de señal
        if macd_current > signal_current and macd_current > macd_prev:
            return "bullish"
        
        # Momentum decreciente por debajo de señal
        if macd_current < signal_current and macd_current < macd_prev:
            return "bearish"
        
        return "neutral"
    
    def _calculate_micro_momentum(self, data: pd.DataFrame) -> float:
        """
        Calcula micro-momentum basado en price action reciente
        """
        if len(data) < 5:
            return 0.0
        
        # Analizar últimas 5 velas
        recent_data = data.tail(5)
        close_prices = recent_data['close']
        
        # Momentum basado en cambios de precio
        price_changes = close_prices.diff().dropna()
        
        # Score basado en consistencia direccional
        positive_moves = (price_changes > 0).sum()
        negative_moves = (price_changes < 0).sum()
        
        if positive_moves > negative_moves:
            momentum = (positive_moves - negative_moves) / len(price_changes)
        else:
            momentum = -(negative_moves - positive_moves) / len(price_changes)
        
        return momentum
    
    def _analyze_micro_volume(self, data: pd.DataFrame) -> Tuple[str, bool]:
        """
        Análisis de volumen para micro-timing
        Más sensible a cambios recientes
        """
        if len(data) < 5:
            return "stable", False
        
        volume = data['volume']
        close = data['close']
        
        # Últimas 3 velas para tendencia de volumen
        recent_volume = volume.tail(3)
        recent_close = close.tail(3)
        
        # Tendencia de volumen
        if len(recent_volume) >= 2:
            volume_change = recent_volume.iloc[-1] - recent_volume.iloc[-2]
            avg_volume = recent_volume.mean()
            
            if volume_change > avg_volume * 0.2:
                volume_trend = "increasing"
            elif volume_change < -avg_volume * 0.2:
                volume_trend = "decreasing"
            else:
                volume_trend = "stable"
        else:
            volume_trend = "stable"
        
        # Confirmación: volumen aumenta con movimiento direccional
        if len(recent_close) >= 2 and len(recent_volume) >= 2:
            price_move = recent_close.iloc[-1] - recent_close.iloc[-2]
            volume_increase = recent_volume.iloc[-1] > recent_volume.iloc[-2]
            
            volume_confirmation = abs(price_move) > 0 and volume_increase
        else:
            volume_confirmation = False
        
        return volume_trend, volume_confirmation
    
    def _analyze_price_action(self, data: pd.DataFrame) -> float:
        """
        Analiza la calidad del price action reciente
        Busca señales de fuerza o debilidad
        """
        if len(data) < 3:
            return 0.0
        
        # Últimas 3 velas
        recent_data = data.tail(3)
        quality_score = 0.0
        
        for _, candle in recent_data.iterrows():
            body_size = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            
            if total_range == 0:
                continue
            
            # Cuerpos grandes indican convicción
            body_ratio = body_size / total_range
            if body_ratio > 0.7:  # Cuerpo > 70% del rango
                quality_score += 0.3
            elif body_ratio > 0.5:  # Cuerpo > 50% del rango
                quality_score += 0.2
        
        return min(1.0, quality_score)
    
    def _calculate_intraday_volatility(self, data: pd.DataFrame) -> float:
        """Calcula volatilidad intraday específica para 15M"""
        if len(data) < 10:
            return 0.0
        
        # Usar últimas 10 velas para volatilidad reciente
        recent_data = data.tail(10)
        
        # Calcular rangos de cada vela
        ranges = (recent_data['high'] - recent_data['low']) / recent_data['close']
        
        # Volatilidad promedio
        avg_volatility = ranges.mean()
        
        return avg_volatility
    
    def _calculate_liquidity_score(self) -> float:
        """Calcula score de liquidez para 15M"""
        # En 15M la liquidez es menor que en timeframes mayores
        return 0.6
    
    def _calculate_fifteenmin_score(self, momentum_score: float, micro_momentum: float,
                                   price_action_quality: float, pattern_confidence: float,
                                   volume_confirmation: bool, current_level_type: str) -> float:
        """
        Calcula score específico para timeframe 15M
        En 15M, el timing y price action son críticos
        """
        score = 0.0
        
        # Price action (30% del score en 15M)
        score += price_action_quality * 30
        
        # Micro-momentum (25% del score)
        score += abs(micro_momentum) * 25
        
        # Patrones de timing (20% del score)
        score += pattern_confidence * 20
        
        # Momentum general (10% del score)
        score += abs(momentum_score) * 10
        
        # Confirmación de volumen (10% del score)
        if volume_confirmation:
            score += 10
        
        # Posición en niveles micro (5% del score)
        if current_level_type in ["support", "resistance"]:
            score += 5
        
        return min(100.0, score)
    
    def _calculate_timing_confidence(self, micro_momentum: float, price_action_quality: float,
                                    pattern_confidence: float, volume_confirmation: bool,
                                    volatility: float) -> float:
        """Calcula confianza específica para timing en 15M"""
        confidence = 0.2  # Base más baja para 15M (más ruido)
        
        # Price action de calidad aumenta mucho la confianza
        confidence += price_action_quality * 0.35
        
        # Micro-momentum claro
        confidence += abs(micro_momentum) * 0.25
        
        # Patrones de timing
        confidence += pattern_confidence * 0.2
        
        # Confirmación de volumen
        if volume_confirmation:
            confidence += 0.15
        
        # Volatilidad moderada es mejor para timing
        if 0.01 < volatility < 0.05:
            confidence += 0.05
        elif volatility > 0.1:  # Demasiada volatilidad reduce confianza
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))


# Factory function para crear analizadores
def create_analyzer(timeframe: str) -> BaseTimeframeAnalyzer:
    """
    Factory function para crear el analizador apropiado según el timeframe
    
    Args:
        timeframe: Timeframe a analizar ('1d', '4h', '1h', '15m')
        
    Returns:
        Instancia del analizador apropiado
        
    Raises:
        ValueError: Si el timeframe no es soportado
    """
    analyzers = {
        '1d': DailyAnalyzer,
        '4h': FourHourAnalyzer,
        '1h': OneHourAnalyzer,
        '15m': FifteenMinuteAnalyzer
    }
    
    if timeframe not in analyzers:
        raise ValueError(f"Timeframe no soportado: {timeframe}. Soportados: {list(analyzers.keys())}")
    
    return analyzers[timeframe]()


# Función de conveniencia para análisis completo
async def analyze_all_timeframes(data_dict: Dict[str, pd.DataFrame], 
                               symbol: str) -> Dict[str, TimeframeAnalysis]:
    """
    Analiza todos los timeframes disponibles para un símbolo
    
    Args:
        data_dict: Diccionario con datos por timeframe {timeframe: DataFrame}
        symbol: Símbolo a analizar
        
    Returns:
        Diccionario con análisis por timeframe {timeframe: TimeframeAnalysis}
    """
    results = {}
    
    for timeframe, data in data_dict.items():
        try:
            analyzer = create_analyzer(timeframe)
            analysis = analyzer.analyze(data, symbol)
            results[timeframe] = analysis
        except Exception as e:
            logger = get_logger("analyze_all_timeframes")
            logger.error(f"Error analizando {symbol} {timeframe}: {str(e)}")
            # Continuar con otros timeframes
            continue
    
    return results