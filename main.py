#!/usr/bin/env python3
"""
Advanced Multi-Timeframe Algorithmic Trading Bot
Main orchestrator for the trading system
"""
import sys
import asyncio
from pathlib import Path

# Add trading_bot to Python path
sys.path.insert(0, str(Path(__file__).parent))

from trading_bot import config, get_logger, LogContext
from trading_bot.config import get_active_symbols
from trading_bot.utils.exceptions import TradingBotException, ConfigurationException, SystemException

def display_system_info():
    """Display system information and configuration"""
    print("🚀 Advanced Multi-Timeframe Algorithmic Trading Bot")
    print("=" * 60)
    print("📊 Multi-Timeframe Analysis System (1D, 4H, 1H, 15M)")
    print("🧠 AI-Enhanced Decision Making with LLM Integration")
    print("🛡️ Institutional Risk Management & Circuit Breakers")
    print("📈 Dynamic Scoring with Regime Adaptation")
    print("🔄 Continuous Learning & Performance Optimization")
    print("=" * 60)
    
    # Display configuration summary
    active_symbols = get_active_symbols()
    print(f"\n📋 Configuration Summary:")
    print(f"   • Active Symbols: {len(active_symbols)} ({', '.join(active_symbols)})")
    print(f"   • Timeframes: {', '.join(config.timeframes)}")
    print(f"   • Risk Limits: {config.max_daily_drawdown*100:.1f}% daily, {config.max_absolute_drawdown*100:.1f}% max")
    print(f"   • Performance Targets: {config.target_profit_factor:.1f}x PF, {config.target_win_rate*100:.1f}% WR")
    print(f"   • Environment: {'Testnet' if config.binance_testnet else 'Mainnet'}")

async def main():
    """Main entry point for the trading bot"""
    logger = get_logger("TradingBot", config.log_level)
    context = LogContext(component="main")
    
    try:
        # Display system information
        display_system_info()
        
        logger.info("🚀 Trading Bot initialization starting...", context=context)
        
        # Validate configuration
        logger.info("Validating configuration...", context=context)
        validate_configuration()
        logger.info("✅ Configuration validated successfully", context=context)
        
        # Check system readiness
        logger.info("Checking system readiness...", context=context)
        
        # Run foundation tests
        logger.info("Running foundation tests...", context=context)
        from tests.test_foundation import run_foundation_tests
        if not run_foundation_tests():
            raise SystemException("Foundation tests failed", "SYS_8003")
        
        print("\n✅ System Status: READY")
        print("🔧 Implementation Status: FOUNDATION COMPLETE")
        print("\n📝 Next Implementation Steps:")
        print("   1. Data Layer (Tasks 2.1-2.3) - Multi-timeframe data fetching & validation")
        print("   2. Analysis Engine (Tasks 3.1-3.3) - Technical analysis & pattern detection")
        print("   3. Scoring System (Tasks 4.1-4.3) - Dynamic scoring with regime adaptation")
        print("   4. AI Integration (Tasks 5.1-5.3) - LLM-enhanced decision making")
        print("   5. Risk Management (Tasks 6.1-6.4) - Institutional risk controls")
        print("   6. Execution Engine (Tasks 7.1-7.3) - Trade execution & position management")
        print("   7. Monitoring System (Tasks 8.1-8.4) - Health checks & performance tracking")
        print("   8. Feedback Loop (Tasks 9.1-9.3) - Learning & optimization")
        print("   9. Reporting (Tasks 10.1-10.3) - Analytics & visualization")
        print("   10. Final Integration (Tasks 11.1-11.3) - Complete system orchestration")
        
        print(f"\n🎯 Ready to implement next phase!")
        print(f"📖 Open .kiro/specs/advanced-trading-bot/tasks.md to start implementing tasks")
        
        logger.info("✅ Foundation setup complete! Ready for next implementation phase.", context=context)
        
    except TradingBotException as e:
        logger.error(f"Trading bot error: {e}", context=context)
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", context=context, exc_info=True)
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

def validate_configuration():
    """Validate the trading bot configuration"""
    try:
        # This will trigger validation in the config's __post_init__
        _ = config.binance_api_key
        _ = config.binance_api_secret
        
        if not config.symbols:
            raise ConfigurationException("No trading symbols configured")
        
        if not config.timeframes:
            raise ConfigurationException("No timeframes configured")
            
    except Exception as e:
        raise ConfigurationException(f"Configuration validation failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())