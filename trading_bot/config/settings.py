"""
Core configuration settings for the trading bot
Handles environment variables, API keys, and global parameters
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class TradingConfig:
    """Main configuration class for the trading bot"""
    
    # API Configuration
    binance_api_key: str = field(default_factory=lambda: os.getenv('BINANCE_API_KEY', ''))
    binance_api_secret: str = field(default_factory=lambda: os.getenv('BINANCE_API_SECRET', ''))
    binance_testnet: bool = field(default_factory=lambda: os.getenv('BINANCE_TESTNET', 'False').lower() == 'true')
    
    # Trading Parameters
    symbols: List[str] = field(default_factory=lambda: ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT'])
    timeframes: List[str] = field(default_factory=lambda: ['1d', '4h', '1h', '15m'])
    candles_limit: int = 500
    
    # Scoring Weights (Dynamic - will be auto-calibrated)
    mtf_structure_weight: float = 0.35
    technical_confluence_weight: float = 0.25
    market_context_weight: float = 0.20
    risk_metrics_weight: float = 0.20
    
    # Risk Management Parameters
    max_daily_drawdown: float = 0.02  # 2%
    max_weekly_drawdown: float = 0.05  # 5%
    max_monthly_drawdown: float = 0.10  # 10%
    max_absolute_drawdown: float = 0.15  # 15%
    max_portfolio_heat: float = 0.06  # 6%
    max_position_per_symbol: float = 0.30  # 30%
    max_sector_exposure: float = 0.50  # 50%
    max_correlated_positions: int = 3
    max_total_positions: int = 5
    
    # Circuit Breaker Parameters
    consecutive_loss_limit: int = 3
    rapid_drawdown_percent: float = 0.015  # 1.5%
    rapid_drawdown_timeframe: str = '1h'
    correlation_breach_threshold: float = 0.8
    
    # LLM Integration Parameters
    llm_score_threshold_min: float = 60.0
    llm_score_threshold_max: float = 75.0
    max_daily_llm_cost: float = 50.0
    llm_cache_duration_minutes: int = 15
    
    # Execution Parameters
    max_slippage_tolerance: float = 0.002  # 0.2%
    position_scaling_levels: int = 3
    execution_timeout_seconds: int = 30
    
    # Data Management
    cache_ttl_seconds: int = 300  # 5 minutes
    data_quality_threshold: float = 0.95
    max_data_age_hours: int = 24
    
    # System Parameters
    cycle_interval_seconds: int = 60
    health_check_interval_seconds: int = 30
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    timezone: str = "UTC"
    
    # Performance Targets
    target_profit_factor: float = 2.0
    target_win_rate: float = 0.65
    target_sharpe_ratio: float = 1.5
    target_calmar_ratio: float = 2.0
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if not self.binance_api_key:
            raise ValueError("BINANCE_API_KEY is required")
        if not self.binance_api_secret:
            raise ValueError("BINANCE_API_SECRET is required")
        
        # Validate weights sum to 1.0
        total_weight = (self.mtf_structure_weight + 
                       self.technical_confluence_weight + 
                       self.market_context_weight + 
                       self.risk_metrics_weight)
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total_weight}")
        
        # Validate risk parameters
        if self.max_daily_drawdown >= self.max_weekly_drawdown:
            raise ValueError("Daily drawdown limit must be less than weekly limit")
        if self.max_weekly_drawdown >= self.max_monthly_drawdown:
            raise ValueError("Weekly drawdown limit must be less than monthly limit")
        if self.max_monthly_drawdown >= self.max_absolute_drawdown:
            raise ValueError("Monthly drawdown limit must be less than absolute limit")

# Global configuration instance
config = TradingConfig()

# Legacy compatibility - maintain existing config structure
BINANCE_API_KEY = config.binance_api_key
BINANCE_API_SECRET = config.binance_api_secret
BINANCE_TESTNET = config.binance_testnet
SYMBOLS = config.symbols
TIMEZONE = config.timezone