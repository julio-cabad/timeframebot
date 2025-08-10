# Advanced Multi-Timeframe Algorithmic Trading Bot

🚀 A sophisticated trading system that combines quantitative technical analysis with AI-enhanced decision making for cryptocurrency markets.

## 🎯 Key Features

- **Multi-Timeframe Analysis**: 1D, 4H, 1H, 15M comprehensive analysis
- **AI-Enhanced Decisions**: LLM integration for ambiguous market situations
- **Institutional Risk Management**: Circuit breakers, drawdown limits, position controls
- **Adaptive Learning**: Feedback loop system for continuous optimization
- **Scalable Architecture**: Modular design supporting $100 to $100,000+ capital

## 🏗️ Project Structure

```
trading_bot/
├── config/                 # Configuration management
│   ├── settings.py         # Core settings and parameters
│   ├── risk_params.py      # Risk management parameters
│   ├── market_regimes.py   # Market regime detection
│   └── symbols.py          # Symbol management
├── utils/                  # Utility modules
│   ├── logger.py           # Structured logging system
│   └── exceptions.py       # Custom exception hierarchy
└── __init__.py

tests/                      # Test suite
├── test_foundation.py      # Foundation tests
└── __init__.py

logs/                       # Log files (auto-created)
├── trading_bot.log         # Main log file
└── trading_bot_errors.log  # Error log file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Ensure your `.env` file contains:
```env
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_TESTNET=false
LOG_LEVEL=INFO
```

### 3. Run Foundation Tests

```bash
python tests/test_foundation.py
```

### 4. Initialize the Bot

```bash
python main.py
```

## 📊 Current Implementation Status

### ✅ Phase 1: Foundation (COMPLETED)
- [x] Modular project structure
- [x] Configuration system with environment variables
- [x] Comprehensive logging with structured output
- [x] Exception hierarchy and error handling
- [x] Risk parameters and market regime detection
- [x] Symbol management system
- [x] Foundation tests (7/7 passing)

### 🚧 Next Phases
- **Phase 2**: Data Layer (Multi-timeframe fetcher, validation, storage)
- **Phase 3**: Analysis Engine (Technical analysis, pattern detection)
- **Phase 4**: Scoring System (Dynamic scoring, regime adaptation)
- **Phase 5**: AI Integration (LLM connector, intelligent usage)
- **Phase 6**: Risk Management (Global risk manager, circuit breakers)

## 🛡️ Risk Management Features

- **Drawdown Limits**: Daily (2%), Weekly (5%), Monthly (10%), Absolute (15%)
- **Circuit Breakers**: Consecutive losses, rapid drawdown, correlation limits
- **Position Limits**: Max 30% per symbol, 6% portfolio heat
- **Emergency Controls**: Kill switch, auto-liquidation protocols

## 📈 Performance Targets

- **Profit Factor**: > 2.0
- **Win Rate**: > 65%
- **Sharpe Ratio**: > 1.5
- **Max Drawdown**: < 15%
- **System Uptime**: > 99.5%

## 🧪 Testing

The project includes comprehensive testing:

```bash
# Run foundation tests
python tests/test_foundation.py

# Run specific test categories (coming in future phases)
python -m pytest tests/test_data_layer.py
python -m pytest tests/test_analysis.py
python -m pytest tests/test_risk_management.py
```

## 📝 Logging

The system uses structured logging with multiple levels:

- **Console Output**: Human-readable format for development
- **File Logging**: JSON structured logs for analysis
- **Error Logging**: Separate error log for critical issues
- **Trading Logs**: Specialized logs for trades, signals, and performance

## 🔧 Configuration

Key configuration parameters:

```python
# Risk Management
max_daily_drawdown = 0.02      # 2%
max_portfolio_heat = 0.06      # 6%
max_position_per_symbol = 0.30 # 30%

# Scoring Weights (auto-calibrated)
mtf_structure_weight = 0.35    # 35%
technical_confluence_weight = 0.25  # 25%
market_context_weight = 0.20   # 20%
risk_metrics_weight = 0.20     # 20%

# Performance Targets
target_profit_factor = 2.0
target_win_rate = 0.65
```

## 🤝 Contributing

This is a sophisticated trading system under active development. Each phase builds upon the previous one with comprehensive testing and validation.

## ⚠️ Disclaimer

This trading bot is for educational and research purposes. Cryptocurrency trading involves substantial risk of loss. Never trade with money you cannot afford to lose.

## 📄 License

This project is proprietary software for educational purposes.