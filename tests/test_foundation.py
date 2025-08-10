#!/usr/bin/env python3
"""
Test script for the trading bot foundation
Validates core infrastructure components
"""
import sys
import unittest
from pathlib import Path

# Add trading_bot to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_bot.config.settings import config, TradingConfig
from trading_bot.config.risk_params import risk_params, MarketRegime
from trading_bot.config.market_regimes import regime_detector, REGIME_CONFIGS
from trading_bot.config.symbols import symbol_manager, SymbolCategory
from trading_bot.utils.logger import get_logger, LogContext
from trading_bot.utils.exceptions import (
    TradingBotException, ConfigurationException, DataException,
    RiskException, ExecutionException, LLMException
)

class TestFoundation(unittest.TestCase):
    """Test cases for the trading bot foundation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = get_logger("TestFoundation", "DEBUG")
        self.context = LogContext(component="test_foundation")
    
    def test_configuration_loading(self):
        """Test configuration loading and validation"""
        self.logger.info("Testing configuration loading...", context=self.context)
        
        # Test that config is loaded
        self.assertIsInstance(config, TradingConfig)
        
        # Test required fields
        self.assertTrue(config.binance_api_key)
        self.assertTrue(config.binance_api_secret)
        self.assertTrue(config.symbols)
        self.assertTrue(config.timeframes)
        
        # Test weight validation
        total_weight = (config.mtf_structure_weight + 
                       config.technical_confluence_weight + 
                       config.market_context_weight + 
                       config.risk_metrics_weight)
        self.assertAlmostEqual(total_weight, 1.0, places=2)
        
        self.logger.info("✅ Configuration loading test passed", context=self.context)
    
    def test_risk_parameters(self):
        """Test risk parameters configuration"""
        self.logger.info("Testing risk parameters...", context=self.context)
        
        # Test risk parameters are loaded
        self.assertIsNotNone(risk_params.drawdown_limits)
        self.assertIsNotNone(risk_params.circuit_breakers)
        self.assertIsNotNone(risk_params.position_limits)
        self.assertIsNotNone(risk_params.emergency_protocols)
        
        # Test drawdown limits are properly ordered
        self.assertLess(risk_params.drawdown_limits["daily"], 
                       risk_params.drawdown_limits["weekly"])
        self.assertLess(risk_params.drawdown_limits["weekly"], 
                       risk_params.drawdown_limits["monthly"])
        self.assertLess(risk_params.drawdown_limits["monthly"], 
                       risk_params.drawdown_limits["absolute_max"])
        
        self.logger.info("✅ Risk parameters test passed", context=self.context)
    
    def test_market_regimes(self):
        """Test market regime detection and configuration"""
        self.logger.info("Testing market regimes...", context=self.context)
        
        # Test regime detector initialization
        self.assertIsNotNone(regime_detector.current_regime)
        self.assertIsInstance(regime_detector.current_regime, MarketRegime)
        
        # Test regime configurations exist for all regimes
        for regime in MarketRegime:
            self.assertIn(regime, REGIME_CONFIGS)
            regime_config = REGIME_CONFIGS[regime]
            self.assertIsNotNone(regime_config.scoring_adjustments)
            self.assertIsNotNone(regime_config.risk_adjustments)
        
        # Test regime detection with sample data
        sample_data = {
            'volatility': 0.03,
            'trend_strength': 0.6,
            'volume_trend': 0.1
        }
        detected_regime = regime_detector.detect_regime(sample_data)
        self.assertIsInstance(detected_regime, MarketRegime)
        
        self.logger.info("✅ Market regimes test passed", context=self.context)
    
    def test_symbol_management(self):
        """Test symbol management functionality"""
        self.logger.info("Testing symbol management...", context=self.context)
        
        # Test active symbols
        active_symbols = symbol_manager.get_active_symbols()
        self.assertGreater(len(active_symbols), 0)
        
        # Test symbol categories
        major_symbols = symbol_manager.get_symbols_by_category(SymbolCategory.MAJOR)
        self.assertIn("BTCUSDT", major_symbols)
        self.assertIn("ETHUSDT", major_symbols)
        
        # Test symbol configuration
        btc_config = symbol_manager.get_symbol_config("BTCUSDT")
        self.assertIsNotNone(btc_config)
        self.assertEqual(btc_config.category, SymbolCategory.MAJOR)
        
        # Test correlation groups
        btc_correlated = symbol_manager.get_correlated_symbols("BTCUSDT")
        self.assertIn("ETHUSDT", btc_correlated)
        
        self.logger.info("✅ Symbol management test passed", context=self.context)
    
    def test_logging_system(self):
        """Test logging system functionality"""
        self.logger.info("Testing logging system...", context=self.context)
        
        # Test different log levels
        test_context = LogContext(component="test_logger", symbol="BTCUSDT")
        
        self.logger.debug("Debug message test", context=test_context)
        self.logger.info("Info message test", context=test_context)
        self.logger.warning("Warning message test", context=test_context)
        
        # Test structured logging methods
        self.logger.log_trade_signal("BTCUSDT", "BUY", 85.5, "1h", {"test": True})
        self.logger.log_performance_metric("win_rate", 0.67, "daily", {"test": True})
        self.logger.log_system_health("test_component", "healthy", {"cpu": 45.2})
        
        self.logger.info("✅ Logging system test passed", context=self.context)
    
    def test_exception_hierarchy(self):
        """Test exception hierarchy and error handling"""
        self.logger.info("Testing exception hierarchy...", context=self.context)
        
        # Test base exception
        try:
            raise TradingBotException("Test exception", "TEST_001", {"test": True})
        except TradingBotException as e:
            self.assertEqual(e.message, "Test exception")
            self.assertEqual(e.error_code, "TEST_001")
            self.assertIn("test", e.context)
        
        # Test specific exceptions
        exceptions_to_test = [
            ConfigurationException,
            DataException,
            RiskException,
            ExecutionException,
            LLMException
        ]
        
        for exc_class in exceptions_to_test:
            try:
                raise exc_class(f"Test {exc_class.__name__}")
            except TradingBotException as e:
                self.assertIsInstance(e, exc_class)
        
        self.logger.info("✅ Exception hierarchy test passed", context=self.context)
    
    def test_integration(self):
        """Test integration between components"""
        self.logger.info("Testing component integration...", context=self.context)
        
        # Test that all components can work together
        active_symbols = symbol_manager.get_active_symbols()
        self.assertGreater(len(active_symbols), 0)
        
        # Test regime detection with symbol data
        for symbol in active_symbols[:2]:  # Test first 2 symbols
            symbol_config = symbol_manager.get_symbol_config(symbol)
            self.assertIsNotNone(symbol_config)
            
            # Simulate market data for regime detection
            market_data = {
                'volatility': symbol_config.volatility_factor * 0.02,
                'trend_strength': 0.5,
                'volume_trend': 0.0
            }
            
            regime = regime_detector.detect_regime(market_data)
            regime_config = regime_detector.get_regime_config(regime)
            
            self.assertIsNotNone(regime_config)
            self.logger.info(f"Symbol {symbol}: regime={regime.value}, config={regime_config.name}",
                           context=self.context)
        
        self.logger.info("✅ Integration test passed", context=self.context)

def run_foundation_tests():
    """Run all foundation tests"""
    logger = get_logger("FoundationTests", "INFO")
    context = LogContext(component="foundation_tests")
    
    logger.info("🧪 Starting Foundation Tests...", context=context)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFoundation)
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run tests
    result = runner.run(suite)
    
    # Log results
    if result.wasSuccessful():
        logger.info("🎉 All foundation tests passed!", context=context)
        logger.info(f"Tests run: {result.testsRun}", context=context)
        return True
    else:
        logger.error(f"❌ Foundation tests failed!", context=context)
        logger.error(f"Tests run: {result.testsRun}, Failures: {len(result.failures)}, Errors: {len(result.errors)}", context=context)
        
        for test, error in result.failures + result.errors:
            logger.error(f"Failed test: {test} - {error}", context=context)
        
        return False

if __name__ == "__main__":
    success = run_foundation_tests()
    sys.exit(0 if success else 1)