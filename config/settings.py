"""
Configuration settings for the multicrypto trading system
"""
import os
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables from .env file
load_dotenv()

# Binance API Configuration
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
BINANCE_TESTNET = os.getenv('BINANCE_TESTNET', 'False').lower() == 'true'

# Data Collection Configuration
SYMBOLS: List[str] = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT']

 # Comisiones de Binance (incluir en cálculos de rentabilidad)
maker_fee: float = 0.0004  # 0.04% - cuando agregas liquidez
taker_fee: float = 0.0005  # 0.05% - cuando tomas liquidez

time_frame: str = '4h'

# Número de velas a obtener por defecto
CANDLES_LIMIT: int = 500

# System Configuration
TIMEZONE: str = "America/Guayaquil"  # Ecuador timezone (UTC-5)