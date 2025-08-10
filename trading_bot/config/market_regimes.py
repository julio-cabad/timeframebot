"""
Market regime detection and configuration
Adaptive parameters based on market conditions
"""
from dataclasses import dataclass
from typing import Dict, Any
from .risk_params import MarketRegime

@dataclass
class RegimeConfig:
    """Configuration for a specific market regime"""
    name: str
    description: str
    scoring_adjustments: Dict[str, float]
    risk_adjustments: Dict[str, float]
    min_score_threshold: float
    llm_usage_multiplier: float

# Market regime configurations
REGIME_CONFIGS = {
    MarketRegime.STRONG_TREND: RegimeConfig(
        name="Strong Trend",
        description="Clear directional movement with strong momentum",
        scoring_adjustments={
            "mtf_structure_weight": 1.2,
            "technical_confluence_weight": 1.0,
            "market_context_weight": 0.9,
            "risk_metrics_weight": 0.9
        },
        risk_adjustments={
            "position_size_multiplier": 1.2,
            "stop_loss_multiplier": 0.9,
            "max_portfolio_heat": 0.07
        },
        min_score_threshold=70.0,
        llm_usage_multiplier=0.8
    ),
    
    MarketRegime.WEAK_TREND: RegimeConfig(
        name="Weak Trend",
        description="Mild directional bias with moderate momentum",
        scoring_adjustments={
            "mtf_structure_weight": 1.1,
            "technical_confluence_weight": 1.1,
            "market_context_weight": 1.0,
            "risk_metrics_weight": 1.0
        },
        risk_adjustments={
            "position_size_multiplier": 1.0,
            "stop_loss_multiplier": 1.0,
            "max_portfolio_heat": 0.06
        },
        min_score_threshold=75.0,
        llm_usage_multiplier=1.0
    ),
    
    MarketRegime.RANGING: RegimeConfig(
        name="Ranging Market",
        description="Sideways movement with defined support/resistance",
        scoring_adjustments={
            "mtf_structure_weight": 0.8,
            "technical_confluence_weight": 1.3,
            "market_context_weight": 1.1,
            "risk_metrics_weight": 1.1
        },
        risk_adjustments={
            "position_size_multiplier": 0.8,
            "stop_loss_multiplier": 1.1,
            "max_portfolio_heat": 0.05
        },
        min_score_threshold=80.0,
        llm_usage_multiplier=1.2
    ),
    
    MarketRegime.HIGH_VOLATILITY: RegimeConfig(
        name="High Volatility",
        description="Elevated price swings and uncertainty",
        scoring_adjustments={
            "mtf_structure_weight": 0.9,
            "technical_confluence_weight": 1.0,
            "market_context_weight": 1.2,
            "risk_metrics_weight": 1.4
        },
        risk_adjustments={
            "position_size_multiplier": 0.6,
            "stop_loss_multiplier": 1.3,
            "max_portfolio_heat": 0.04
        },
        min_score_threshold=85.0,
        llm_usage_multiplier=1.5
    ),
    
    MarketRegime.LOW_VOLATILITY: RegimeConfig(
        name="Low Volatility",
        description="Calm market conditions with minimal price swings",
        scoring_adjustments={
            "mtf_structure_weight": 1.0,
            "technical_confluence_weight": 1.0,
            "market_context_weight": 0.9,
            "risk_metrics_weight": 0.9
        },
        risk_adjustments={
            "position_size_multiplier": 1.1,
            "stop_loss_multiplier": 0.95,
            "max_portfolio_heat": 0.06
        },
        min_score_threshold=65.0,
        llm_usage_multiplier=0.9
    )
}

class RegimeDetector:
    """Detects current market regime based on various indicators"""
    
    def __init__(self):
        self.current_regime = MarketRegime.WEAK_TREND
        self.regime_confidence = 0.5
        self.regime_history = []
    
    def detect_regime(self, market_data: Dict[str, Any]) -> MarketRegime:
        """
        Detect current market regime based on market data
        
        Args:
            market_data: Dictionary containing market indicators
            
        Returns:
            Detected market regime
        """
        # Placeholder implementation - will be enhanced in later phases
        volatility = market_data.get('volatility', 0.02)
        trend_strength = market_data.get('trend_strength', 0.5)
        volume_trend = market_data.get('volume_trend', 0.0)
        
        # Simple regime detection logic
        if volatility > 0.05:
            regime = MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.015:
            regime = MarketRegime.LOW_VOLATILITY
        elif trend_strength > 0.7:
            regime = MarketRegime.STRONG_TREND
        elif trend_strength > 0.3:
            regime = MarketRegime.WEAK_TREND
        else:
            regime = MarketRegime.RANGING
        
        self.current_regime = regime
        self.regime_history.append(regime)
        
        # Keep only last 100 regime detections
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
        
        return regime
    
    def get_regime_config(self, regime: MarketRegime = None) -> RegimeConfig:
        """Get configuration for specified regime or current regime"""
        if regime is None:
            regime = self.current_regime
        return REGIME_CONFIGS[regime]
    
    def get_regime_stability(self) -> float:
        """Calculate regime stability based on recent history"""
        if len(self.regime_history) < 10:
            return 0.5
        
        recent_regimes = self.regime_history[-10:]
        current_regime_count = recent_regimes.count(self.current_regime)
        return current_regime_count / len(recent_regimes)

# Global regime detector instance
regime_detector = RegimeDetector()

def get_regime_config(regime: MarketRegime = None) -> RegimeConfig:
    """Get regime configuration - convenience function"""
    return regime_detector.get_regime_config(regime)