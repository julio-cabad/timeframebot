"""
Sistema de Scoring Dinámico
Dynamic scoring system for trading opportunities with regime adaptation
"""

from .scorer import (
    DynamicScorer,
    ScoringResult,
    ScoringBreakdown,
    ComponentScore,
    ScoreComponent,
    ScoringRegime,
    calculate_dynamic_score
)

from .weight_manager import (
    WeightManager,
    WeightConfiguration,
    PerformanceMetrics,
    WeightOptimizationMethod,
    RegimeDetectionMethod,
    RegimeTransition,
    create_weight_manager
)

from .auto_calibrator import (
    AutoCalibrator,
    CalibrationMethod,
    CalibrationTarget,
    ParameterType,
    ParameterBounds,
    CalibrationResult,
    CalibrationConfig
)

__all__ = [
    'DynamicScorer',
    'ScoringResult',
    'ScoringBreakdown',
    'ComponentScore',
    'ScoreComponent',
    'ScoringRegime',
    'calculate_dynamic_score',
    'WeightManager',
    'WeightConfiguration',
    'PerformanceMetrics',
    'WeightOptimizationMethod',
    'RegimeDetectionMethod',
    'RegimeTransition',
    'create_weight_manager',
    'AutoCalibrator',
    'CalibrationMethod',
    'CalibrationTarget',
    'ParameterType',
    'ParameterBounds',
    'CalibrationResult',
    'CalibrationConfig'
]