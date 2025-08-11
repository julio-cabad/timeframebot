"""
Sistema de Backtesting Profesional para Trading Algorítmico
==========================================================

Como trader senior con más de 10 años de experiencia, he aprendido que
el backtesting es la diferencia entre un sistema rentable y uno que
explota en producción.

Este módulo implementa:
- Backtesting histórico con datos reales
- Simulación de slippage y comisiones realistas
- Análisis de drawdown y risk metrics
- Walk-forward analysis para validación robusta
- Monte Carlo simulation para stress testing
- Comparación con benchmarks (Buy & Hold)
- Reportes detallados con métricas institucionales

Filosofía: "En backtesting confiamos, pero verificamos en papel primero"

Métricas clave que SIEMPRE monitoreamos:
- Profit Factor > 2.0
- Win Rate > 65%
- Sharpe Ratio > 1.5
- Max Drawdown < 15%
- Calmar Ratio > 2.0

Autor: Trader Algorítmico Senior (10+ años)
Zona Horaria: UTC-5 (Ecuador)
"""

from .engine import (
    BacktestEngine,
    BacktestConfig,
    TradeExecution,
    PortfolioState
)

from .data_handler import (
    HistoricalDataHandler,
    DataSource,
    MarketDataPoint,
    DataQuality
)

from .metrics import (
    PerformanceAnalyzer,
    RiskMetrics,
    TradeMetrics,
    DrawdownAnalysis,
    MonteCarloAnalysis
)

from .reports import (
    BacktestReporter,
    ReportType,
    create_backtest_report
)

from .validation import (
    WalkForwardValidator,
    ValidationResult,
    OutOfSampleTest,
    RobustnessTest
)

__all__ = [
    'BacktestEngine',
    'BacktestConfig', 
    'TradeExecution',
    'PortfolioState',
    'HistoricalDataHandler',
    'DataSource',
    'MarketDataPoint',
    'DataQuality',
    'PerformanceAnalyzer',
    'RiskMetrics',
    'TradeMetrics',
    'DrawdownAnalysis',
    'MonteCarloAnalysis',
    'BacktestReporter',
    'ReportType',
    'create_backtest_report',
    'WalkForwardValidator',
    'ValidationResult',
    'OutOfSampleTest',
    'RobustnessTest'
]