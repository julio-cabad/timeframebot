"""
Filtro de Volumen para Trading Institucional
============================================

Como trader senior, el volumen es mi confirmación #1.
Sin volumen, no hay movimiento real - solo ruido.

Este módulo implementa filtros de volumen profesionales.

Autor: Trader Algorítmico Senior (10+ años)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

class VolumeFilter:
    """
    Filtro de volumen institucional
    
    Los traders profesionales sabemos:
    - Breakouts sin volumen = falsos
    - Volumen precede al precio
    - El volumen confirma la dirección
    """
    
    @staticmethod
    def is_volume_sufficient(df: pd.DataFrame, 
                            current_idx: int,
                            lookback: int = 20) -> Tuple[bool, float]:
        """
        Verifica si el volumen actual es suficiente para operar
        
        Args:
            df: DataFrame con datos OHLCV
            current_idx: Índice actual
            lookback: Períodos para calcular promedio
            
        Returns:
            (es_suficiente, ratio_volumen)
        """
        if current_idx < lookback:
            return False, 0.0
        
        # Volumen actual
        current_volume = df.iloc[current_idx]['volume']
        
        # Volumen promedio de los últimos N períodos
        avg_volume = df.iloc[current_idx-lookback:current_idx]['volume'].mean()
        
        if avg_volume == 0:
            return False, 0.0
        
        # Ratio de volumen
        volume_ratio = current_volume / avg_volume
        
        # Criterio: Volumen debe ser al menos 1.2x el promedio
        # Para breakouts, idealmente > 1.5x
        is_sufficient = volume_ratio >= 1.2
        
        return is_sufficient, volume_ratio
    
    @staticmethod
    def detect_volume_breakout(df: pd.DataFrame,
                              current_idx: int,
                              threshold: float = 2.0) -> bool:
        """
        Detecta un breakout de volumen
        
        Un breakout de volumen indica movimiento institucional
        """
        if current_idx < 20:
            return False
        
        current_volume = df.iloc[current_idx]['volume']
        avg_volume = df.iloc[current_idx-20:current_idx]['volume'].mean()
        
        if avg_volume == 0:
            return False
        
        # Breakout de volumen = 2x el promedio
        return (current_volume / avg_volume) >= threshold
    
    @staticmethod
    def calculate_volume_trend(df: pd.DataFrame,
                              current_idx: int,
                              periods: int = 10) -> str:
        """
        Calcula la tendencia del volumen
        
        Returns:
            'increasing', 'decreasing', 'stable'
        """
        if current_idx < periods:
            return 'stable'
        
        volumes = df.iloc[current_idx-periods:current_idx+1]['volume'].values
        
        # Calcular tendencia con regresión lineal simple
        x = np.arange(len(volumes))
        slope = np.polyfit(x, volumes, 1)[0]
        
        avg_volume = volumes.mean()
        if avg_volume == 0:
            return 'stable'
        
        # Normalizar slope
        normalized_slope = slope / avg_volume
        
        if normalized_slope > 0.05:
            return 'increasing'
        elif normalized_slope < -0.05:
            return 'decreasing'
        else:
            return 'stable'
    
    @staticmethod
    def get_volume_score(df: pd.DataFrame,
                        current_idx: int) -> float:
        """
        Calcula un score de volumen (0-100)
        
        Score alto = buenas condiciones de volumen para operar
        """
        score = 50.0  # Base
        
        # Check volumen suficiente
        is_sufficient, volume_ratio = VolumeFilter.is_volume_sufficient(df, current_idx)
        if is_sufficient:
            score += min(25, (volume_ratio - 1.0) * 50)  # Hasta +25 puntos
        else:
            score -= 20  # Penalización por volumen bajo
        
        # Check breakout de volumen
        if VolumeFilter.detect_volume_breakout(df, current_idx):
            score += 15  # Bonus por breakout
        
        # Check tendencia de volumen
        trend = VolumeFilter.calculate_volume_trend(df, current_idx)
        if trend == 'increasing':
            score += 10
        elif trend == 'decreasing':
            score -= 10
        
        return min(100, max(0, score))