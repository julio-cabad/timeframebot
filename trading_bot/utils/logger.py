"""
Comprehensive logging system for the trading bot
Structured logging with multiple levels and output formats
"""
import logging
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, asdict

@dataclass
class LogContext:
    """Structured context for log entries"""
    component: str
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    trade_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Base log structure
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add context if available
        if hasattr(record, 'context') and record.context:
            log_entry["context"] = asdict(record.context)
        
        # Add extra fields
        if hasattr(record, 'extra_fields') and record.extra_fields:
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str)

class TradingBotLogger:
    """Enhanced logger for the trading bot with structured logging"""
    
    def __init__(self, name: str, log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup console and file handlers"""
        # Console handler with simple format
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        
        # File handler with structured JSON format
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_dir / "trading_bot.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter())
        
        # Error file handler
        error_handler = RotatingFileHandler(
            log_dir / "trading_bot_errors.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
    
    def _log_with_context(self, level: int, message: str, 
                         context: Optional[LogContext] = None,
                         extra_fields: Optional[Dict[str, Any]] = None,
                         exc_info: bool = False):
        """Internal method to log with context"""
        extra = {}
        if context:
            extra['context'] = context
        if extra_fields:
            extra['extra_fields'] = extra_fields
        
        self.logger.log(level, message, extra=extra, exc_info=exc_info)
    
    def debug(self, message: str, context: Optional[LogContext] = None,
              extra_fields: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        self._log_with_context(logging.DEBUG, message, context, extra_fields)
    
    def info(self, message: str, context: Optional[LogContext] = None,
             extra_fields: Optional[Dict[str, Any]] = None):
        """Log info message"""
        self._log_with_context(logging.INFO, message, context, extra_fields)
    
    def warning(self, message: str, context: Optional[LogContext] = None,
                extra_fields: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        self._log_with_context(logging.WARNING, message, context, extra_fields)
    
    def error(self, message: str, context: Optional[LogContext] = None,
              extra_fields: Optional[Dict[str, Any]] = None,
              exc_info: bool = True):
        """Log error message"""
        self._log_with_context(logging.ERROR, message, context, extra_fields, exc_info)
    
    def critical(self, message: str, context: Optional[LogContext] = None,
                 extra_fields: Optional[Dict[str, Any]] = None,
                 exc_info: bool = True):
        """Log critical message"""
        self._log_with_context(logging.CRITICAL, message, context, extra_fields, exc_info)
    
    # Trading-specific logging methods
    def log_trade_signal(self, symbol: str, signal_type: str, score: float,
                        timeframe: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log trading signal generation"""
        context = LogContext(component="signal_generator", symbol=symbol, timeframe=timeframe)
        extra_fields = {
            "signal_type": signal_type,
            "score": score,
            **(extra_data or {})
        }
        self.info(f"Trading signal generated: {signal_type} for {symbol}", 
                 context=context, extra_fields=extra_fields)
    
    def log_trade_execution(self, trade_id: str, symbol: str, action: str,
                           price: float, quantity: float, success: bool,
                           extra_data: Optional[Dict[str, Any]] = None):
        """Log trade execution"""
        context = LogContext(component="trade_executor", symbol=symbol, trade_id=trade_id)
        extra_fields = {
            "action": action,
            "price": price,
            "quantity": quantity,
            "success": success,
            **(extra_data or {})
        }
        level_method = self.info if success else self.error
        level_method(f"Trade execution: {action} {quantity} {symbol} at {price}",
                    context=context, extra_fields=extra_fields)
    
    def log_risk_event(self, event_type: str, severity: str, description: str,
                      symbol: Optional[str] = None, extra_data: Optional[Dict[str, Any]] = None):
        """Log risk management events"""
        context = LogContext(component="risk_manager", symbol=symbol)
        extra_fields = {
            "event_type": event_type,
            "severity": severity,
            **(extra_data or {})
        }
        
        if severity.lower() in ["critical", "high"]:
            self.critical(f"Risk event: {description}", context=context, extra_fields=extra_fields)
        elif severity.lower() == "medium":
            self.warning(f"Risk event: {description}", context=context, extra_fields=extra_fields)
        else:
            self.info(f"Risk event: {description}", context=context, extra_fields=extra_fields)
    
    def log_performance_metric(self, metric_name: str, value: float,
                              period: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log performance metrics"""
        context = LogContext(component="performance_tracker")
        extra_fields = {
            "metric_name": metric_name,
            "value": value,
            "period": period,
            **(extra_data or {})
        }
        self.info(f"Performance metric: {metric_name} = {value} ({period})",
                 context=context, extra_fields=extra_fields)
    
    def log_system_health(self, component: str, status: str, metrics: Dict[str, Any]):
        """Log system health checks"""
        context = LogContext(component=component)
        extra_fields = {
            "health_status": status,
            "health_metrics": metrics
        }
        
        if status.lower() == "healthy":
            self.debug(f"System health check: {component} is {status}",
                      context=context, extra_fields=extra_fields)
        else:
            self.warning(f"System health check: {component} is {status}",
                        context=context, extra_fields=extra_fields)

def get_logger(name: str, log_level: str = "INFO") -> TradingBotLogger:
    """Get or create a logger instance"""
    return TradingBotLogger(name, log_level)

# Compatibility with existing code
def get_logger_legacy(name: str) -> logging.Logger:
    """Legacy compatibility function"""
    return logging.getLogger(name)