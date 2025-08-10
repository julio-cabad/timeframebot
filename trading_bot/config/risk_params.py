"""
Risk management parameters and configurations
Institutional-grade risk controls and circuit breakers
"""
from dataclasses import dataclass
from typing import Dict, Any
from enum import Enum

class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MarketRegime(Enum):
    """Market regime classifications"""
    STRONG_TREND = "strong_trend"
    WEAK_TREND = "weak_trend"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

@dataclass
class RiskParameters:
    """Comprehensive risk management parameters"""
    
    # Drawdown Limits
    drawdown_limits: Dict[str, float] = None
    
    # Circuit Breaker Rules
    circuit_breakers: Dict[str, Dict[str, Any]] = None
    
    # Position Limits
    position_limits: Dict[str, float] = None
    
    # Emergency Protocols
    emergency_protocols: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.drawdown_limits is None:
            self.drawdown_limits = {
                "daily": 0.02,      # 2%
                "weekly": 0.05,     # 5%
                "monthly": 0.10,    # 10%
                "absolute_max": 0.15 # 15%
            }
        
        if self.circuit_breakers is None:
            self.circuit_breakers = {
                "consecutive_losses": {
                    "count": 3,
                    "action": "pause_symbol_24h"
                },
                "rapid_drawdown": {
                    "percent": 0.015,  # 1.5%
                    "timeframe": "1h",
                    "action": "pause_all_1h"
                },
                "correlation_breach": {
                    "threshold": 0.8,
                    "action": "block_correlated_trades"
                },
                "portfolio_heat": {
                    "threshold": 0.08,  # 8%
                    "action": "reduce_position_sizes"
                }
            }
        
        if self.position_limits is None:
            self.position_limits = {
                "max_positions": 5,
                "max_per_symbol": 0.30,      # 30%
                "max_sector_exposure": 0.50,  # 50%
                "max_correlated_positions": 3,
                "min_position_size": 0.01,    # 1%
                "max_leverage": 3.0
            }
        
        if self.emergency_protocols is None:
            self.emergency_protocols = {
                "kill_switch_file": "emergency_stop.flag",
                "auto_deleverage_threshold": 0.12,  # 12%
                "liquidation_mode": "market_orders",
                "emergency_contact": "admin@tradingbot.com",
                "max_recovery_attempts": 3
            }

# Regime-specific risk adjustments
REGIME_RISK_ADJUSTMENTS = {
    MarketRegime.STRONG_TREND: {
        "position_size_multiplier": 1.2,
        "stop_loss_multiplier": 0.9,
        "max_portfolio_heat": 0.07
    },
    MarketRegime.RANGING: {
        "position_size_multiplier": 0.8,
        "stop_loss_multiplier": 1.1,
        "max_portfolio_heat": 0.05
    },
    MarketRegime.HIGH_VOLATILITY: {
        "position_size_multiplier": 0.6,
        "stop_loss_multiplier": 1.3,
        "max_portfolio_heat": 0.04
    },
    MarketRegime.LOW_VOLATILITY: {
        "position_size_multiplier": 1.1,
        "stop_loss_multiplier": 0.95,
        "max_portfolio_heat": 0.06
    }
}

# Default risk parameters instance
risk_params = RiskParameters()

def get_risk_params() -> RiskParameters:
    """Get risk parameters - convenience function"""
    return risk_params

def get_regime_risk_adjustments(regime: MarketRegime) -> Dict[str, float]:
    """Get risk adjustments for a specific market regime"""
    return REGIME_RISK_ADJUSTMENTS.get(regime, {})