"""
Analysis module for the advanced trading bot
Multi-timeframe technical analysis system
"""

from .tf_analyzers import (
    BaseTimeframeAnalyzer,
    DailyAnalyzer,
    FourHourAnalyzer,
    OneHourAnalyzer,
    FifteenMinuteAnalyzer,
    TimeframeAnalysis,
    MarketStructure,
    PatternType,
    TechnicalIndicators,
    create_analyzer,
    analyze_all_timeframes
)

from .confluence import (
    ConfluenceEngine,
    ConfluenceResult,
    ConfluenceStrength,
    AlignmentType,
    DivergenceType,
    ConfluenceLevel,
    Divergence,
    AlignmentScore,
    analyze_confluence
)

from .patterns import (
    PatternDetector,
    DetectedPattern,
    PatternCategory,
    PatternReliability,
    PatternTarget,
    detect_patterns
)

__all__ = [
    'BaseTimeframeAnalyzer',
    'DailyAnalyzer',
    'FourHourAnalyzer',
    'OneHourAnalyzer',
    'FifteenMinuteAnalyzer',
    'TimeframeAnalysis',
    'MarketStructure',
    'PatternType',
    'TechnicalIndicators',
    'create_analyzer',
    'analyze_all_timeframes',
    'ConfluenceEngine',
    'ConfluenceResult',
    'ConfluenceStrength',
    'AlignmentType',
    'DivergenceType',
    'ConfluenceLevel',
    'Divergence',
    'AlignmentScore',
    'analyze_confluence',
    'PatternDetector',
    'DetectedPattern',
    'PatternCategory',
    'PatternReliability',
    'PatternTarget',
    'detect_patterns'
]