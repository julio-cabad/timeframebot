"""
Advanced Multi-Timeframe Algorithmic Trading Bot
A sophisticated trading system with AI-enhanced decision making
"""

__version__ = "1.0.0"
__author__ = "Trading Bot Team"
__description__ = "Advanced Multi-Timeframe Algorithmic Trading Bot"

# Import core components for easy access
from .config.settings import config, TradingConfig
from .utils.logger import get_logger, LogContext
from .utils.exceptions import (
    TradingBotException,
    DataException,
    AnalysisException,
    RiskException,
    ExecutionException,
    LLMException,
    SystemException
)

__all__ = [
    "config",
    "TradingConfig", 
    "get_logger",
    "LogContext",
    "TradingBotException",
    "DataException",
    "AnalysisException", 
    "RiskException",
    "ExecutionException",
    "LLMException",
    "SystemException"
]