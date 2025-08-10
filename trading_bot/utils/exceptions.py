"""
Custom exception classes for the trading bot
Hierarchical exception structure for better error handling
"""
from typing import Optional, Dict, Any

class TradingBotException(Exception):
    """Base exception for all trading bot errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def __str__(self) -> str:
        base_msg = self.message
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.context:
            base_msg += f" | Context: {self.context}"
        return base_msg

class ConfigurationException(TradingBotException):
    """Configuration-related errors"""
    pass

class DataException(TradingBotException):
    """Data-related errors"""
    pass

class DataFetchException(DataException):
    """Data fetching errors"""
    pass

class DataValidationException(DataException):
    """Data validation errors"""
    pass

class DataStorageException(DataException):
    """Data storage errors"""
    pass

class AnalysisException(TradingBotException):
    """Analysis-related errors"""
    pass

class IndicatorException(AnalysisException):
    """Technical indicator calculation errors"""
    pass

class PatternDetectionException(AnalysisException):
    """Pattern detection errors"""
    pass

class ScoringException(TradingBotException):
    """Scoring system errors"""
    pass

class WeightCalculationException(ScoringException):
    """Weight calculation errors"""
    pass

class RiskException(TradingBotException):
    """Risk management errors"""
    pass

class DrawdownException(RiskException):
    """Drawdown limit exceeded errors"""
    pass

class PositionLimitException(RiskException):
    """Position limit exceeded errors"""
    pass

class CorrelationException(RiskException):
    """Correlation limit exceeded errors"""
    pass

class CircuitBreakerException(RiskException):
    """Circuit breaker triggered errors"""
    pass

class ExecutionException(TradingBotException):
    """Trade execution errors"""
    pass

class OrderException(ExecutionException):
    """Order placement/management errors"""
    pass

class SlippageException(ExecutionException):
    """Excessive slippage errors"""
    pass

class LiquidityException(ExecutionException):
    """Insufficient liquidity errors"""
    pass

class LLMException(TradingBotException):
    """LLM integration errors"""
    pass

class LLMConnectionException(LLMException):
    """LLM connection errors"""
    pass

class LLMResponseException(LLMException):
    """LLM response parsing errors"""
    pass

class LLMCostException(LLMException):
    """LLM cost limit exceeded errors"""
    pass

class MonitoringException(TradingBotException):
    """Monitoring and health check errors"""
    pass

class HealthCheckException(MonitoringException):
    """Health check failure errors"""
    pass

class AlertException(MonitoringException):
    """Alert system errors"""
    pass

class PerformanceException(TradingBotException):
    """Performance tracking errors"""
    pass

class BacktestException(TradingBotException):
    """Backtesting errors"""
    pass

class SystemException(TradingBotException):
    """System-level errors"""
    pass

class EmergencyStopException(SystemException):
    """Emergency stop triggered errors"""
    pass

class ResourceException(SystemException):
    """Resource exhaustion errors"""
    pass

# Error code constants
class ErrorCodes:
    """Standard error codes for the trading bot"""
    
    # Configuration errors (1000-1099)
    CONFIG_MISSING_API_KEY = "CFG_1001"
    CONFIG_INVALID_PARAMETER = "CFG_1002"
    CONFIG_VALIDATION_FAILED = "CFG_1003"
    
    # Data errors (2000-2099)
    DATA_FETCH_FAILED = "DATA_2001"
    DATA_VALIDATION_FAILED = "DATA_2002"
    DATA_STORAGE_FAILED = "DATA_2003"
    DATA_QUALITY_POOR = "DATA_2004"
    DATA_MISSING = "DATA_2005"
    
    # Analysis errors (3000-3099)
    ANALYSIS_CALCULATION_FAILED = "ANA_3001"
    ANALYSIS_INSUFFICIENT_DATA = "ANA_3002"
    ANALYSIS_PATTERN_DETECTION_FAILED = "ANA_3003"
    
    # Scoring errors (4000-4099)
    SCORING_CALCULATION_FAILED = "SCR_4001"
    SCORING_WEIGHT_INVALID = "SCR_4002"
    
    # Risk errors (5000-5099)
    RISK_DRAWDOWN_EXCEEDED = "RSK_5001"
    RISK_POSITION_LIMIT_EXCEEDED = "RSK_5002"
    RISK_CORRELATION_EXCEEDED = "RSK_5003"
    RISK_CIRCUIT_BREAKER_TRIGGERED = "RSK_5004"
    
    # Execution errors (6000-6099)
    EXEC_ORDER_FAILED = "EXE_6001"
    EXEC_SLIPPAGE_EXCEEDED = "EXE_6002"
    EXEC_INSUFFICIENT_LIQUIDITY = "EXE_6003"
    EXEC_TIMEOUT = "EXE_6004"
    
    # LLM errors (7000-7099)
    LLM_CONNECTION_FAILED = "LLM_7001"
    LLM_RESPONSE_INVALID = "LLM_7002"
    LLM_COST_EXCEEDED = "LLM_7003"
    LLM_TIMEOUT = "LLM_7004"
    
    # System errors (8000-8099)
    SYS_EMERGENCY_STOP = "SYS_8001"
    SYS_RESOURCE_EXHAUSTED = "SYS_8002"
    SYS_HEALTH_CHECK_FAILED = "SYS_8003"
    
    # Performance errors (9000-9099)
    PERF_TRACKING_FAILED = "PRF_9001"
    PERF_BACKTEST_FAILED = "PRF_9002"

def create_exception(exception_class: type, message: str, 
                    error_code: Optional[str] = None,
                    context: Optional[Dict[str, Any]] = None) -> TradingBotException:
    """Factory function to create exceptions with consistent structure"""
    return exception_class(message, error_code, context)