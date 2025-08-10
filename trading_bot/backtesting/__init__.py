"""
Sistema de Backtesting Profesional
==================================

Como trader algorítmico con 10+ años de experiencia, he diseñado este sistema
para validar estrategias con datos 100% REALES de Binance.

Filosofía:
- Usar datos reales siempre (nada simulado excepto el capital)
- Métricas que importan: Win Rate, Profit Factor, Drawdown
- Código limpio, escalable y profesional
- Capital inicial: $200 USD (ajustable)

Autor: Trader Algorítmico Senior
Zona Horaria: UTC-5 (Ecuador)
"""

from .engine import BacktestEngine
from .portfolio import VirtualPortfolio, Trade
from .metrics import PerformanceMetrics, MetricsCalculator
from .reporter import BacktestReporter

__all__ = [
    'BacktestEngine',
    'VirtualPortfolio',
    'Trade',
    'PerformanceMetrics',
    'MetricsCalculator',
    'BacktestReporter'
]

# Configuración por defecto
DEFAULT_INITIAL_CAPITAL = 200.0  # USD
DEFAULT_POSITION_SIZE_PCT = 0.10  # 10% del capital por trade
DEFAULT_STOP_LOSS_PCT = 0.02  # 2% stop loss
DEFAULT_TAKE_PROFIT_PCT = 0.06  # 6% take profit (3:1 RR)