"""
Configuration module for the trading bot
Centralized configuration management with validation
"""

from .settings import config, TradingConfig, BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TESTNET
from .symbols import SymbolManager, get_active_symbols
from .risk_params import RiskParameters, get_risk_params
from .market_regimes import MarketRegime, RegimeDetector, get_regime_config

__all__ = [
    "config",
    "TradingConfig",
    "BINANCE_API_KEY",
    "BINANCE_API_SECRET", 
    "BINANCE_TESTNET",
    "SymbolManager",
    "get_active_symbols",
    "RiskParameters",
    "get_risk_params",
    "MarketRegime",
    "RegimeDetector",
    "get_regime_config"
]