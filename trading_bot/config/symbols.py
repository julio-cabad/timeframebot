"""
Symbol management and configuration
Dynamic symbol lists and market-specific parameters
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class SymbolCategory(Enum):
    """Symbol categories for risk management"""
    MAJOR = "major"          # BTC, ETH
    ALTCOIN = "altcoin"      # ADA, SOL, DOT, etc.
    DEFI = "defi"           # UNI, AAVE, COMP, etc.
    MEME = "meme"           # DOGE, SHIB, etc.
    STABLE = "stable"        # Stablecoin pairs

@dataclass
class SymbolConfig:
    """Configuration for individual trading symbols"""
    symbol: str
    category: SymbolCategory
    min_notional: float
    tick_size: float
    step_size: float
    max_position_size: float
    correlation_group: str
    volatility_factor: float = 1.0
    liquidity_score: float = 1.0
    active: bool = True

class SymbolManager:
    """Manages trading symbols and their configurations"""
    
    def __init__(self):
        self.symbols = self._initialize_symbols()
        self.correlation_groups = self._define_correlation_groups()
    
    def _initialize_symbols(self) -> Dict[str, SymbolConfig]:
        """Initialize symbol configurations"""
        return {
            "BTCUSDT": SymbolConfig(
                symbol="BTCUSDT",
                category=SymbolCategory.MAJOR,
                min_notional=5.0,
                tick_size=0.01,
                step_size=0.00001,
                max_position_size=0.40,  # 40% max for BTC
                correlation_group="crypto_major",
                volatility_factor=1.0,
                liquidity_score=1.0,
                active=True  # ✅ ACTIVO - Cambiar a False para desactivar
            ),
            "ETHUSDT": SymbolConfig(
                symbol="ETHUSDT",
                category=SymbolCategory.MAJOR,
                min_notional=5.0,
                tick_size=0.01,
                step_size=0.0001,
                max_position_size=0.35,  # 35% max for ETH
                correlation_group="crypto_major",
                volatility_factor=1.1,
                liquidity_score=0.95,
                active=True  # ✅ ACTIVO - Cambiar a False para desactivar
            ),
            "ADAUSDT": SymbolConfig(
                symbol="ADAUSDT",
                category=SymbolCategory.ALTCOIN,
                min_notional=5.0,
                tick_size=0.0001,
                step_size=1.0,
                max_position_size=0.25,  # 25% max for altcoins
                correlation_group="crypto_altcoin",
                volatility_factor=1.3,
                liquidity_score=0.8,
                active=False  # ❌ DESACTIVADO - Cambiar a True para activar
            ),
            "SOLUSDT": SymbolConfig(
                symbol="SOLUSDT",
                category=SymbolCategory.ALTCOIN,
                min_notional=5.0,
                tick_size=0.001,
                step_size=0.01,
                max_position_size=0.25,
                correlation_group="crypto_altcoin",
                volatility_factor=1.4,
                liquidity_score=0.85,
                active=False  # ❌ DESACTIVADO - Cambiar a True para activar
            ),
            "DOTUSDT": SymbolConfig(
                symbol="DOTUSDT",
                category=SymbolCategory.ALTCOIN,
                min_notional=5.0,
                tick_size=0.001,
                step_size=0.01,
                max_position_size=0.25,
                correlation_group="crypto_altcoin",
                volatility_factor=1.2,
                liquidity_score=0.75,
                active=False  # ❌ DESACTIVADO - Cambiar a True para activar
            ),
            # 🆕 AGREGAR NUEVOS SÍMBOLOS AQUÍ
            "BNBUSDT": SymbolConfig(
                symbol="BNBUSDT",
                category=SymbolCategory.ALTCOIN,
                min_notional=5.0,
                tick_size=0.01,
                step_size=0.001,
                max_position_size=0.25,
                correlation_group="crypto_altcoin",
                volatility_factor=1.2,
                liquidity_score=0.9,
                active=False  # ❌ DESACTIVADO - Cambiar a True para activar
            ),
            "MATICUSDT": SymbolConfig(
                symbol="MATICUSDT",
                category=SymbolCategory.ALTCOIN,
                min_notional=5.0,
                tick_size=0.0001,
                step_size=1.0,
                max_position_size=0.25,
                correlation_group="crypto_altcoin",
                volatility_factor=1.5,
                liquidity_score=0.8,
                active=False  # ❌ DESACTIVADO - Cambiar a True para activar
            ),
            "AVAXUSDT": SymbolConfig(
                symbol="AVAXUSDT",
                category=SymbolCategory.ALTCOIN,
                min_notional=5.0,
                tick_size=0.001,
                step_size=0.01,
                max_position_size=0.25,
                correlation_group="crypto_altcoin",
                volatility_factor=1.3,
                liquidity_score=0.8,
                active=False  # ❌ DESACTIVADO - Cambiar a True para activar
            )
        }
    
    def _define_correlation_groups(self) -> Dict[str, List[str]]:
        """Define correlation groups for risk management"""
        return {
            "crypto_major": ["BTCUSDT", "ETHUSDT"],
            "crypto_altcoin": ["ADAUSDT", "SOLUSDT", "DOTUSDT"],
            "defi_tokens": [],  # To be populated later
            "layer1_tokens": ["ETHUSDT", "SOLUSDT", "DOTUSDT"]
        }
    
    def get_active_symbols(self) -> List[str]:
        """Get list of active trading symbols"""
        return [symbol for symbol, config in self.symbols.items() if config.active]
    
    def get_symbol_config(self, symbol: str) -> Optional[SymbolConfig]:
        """Get configuration for a specific symbol"""
        return self.symbols.get(symbol)
    
    def get_symbols_by_category(self, category: SymbolCategory) -> List[str]:
        """Get symbols by category"""
        return [symbol for symbol, config in self.symbols.items() 
                if config.category == category and config.active]
    
    def get_correlation_group(self, symbol: str) -> Optional[str]:
        """Get correlation group for a symbol"""
        config = self.get_symbol_config(symbol)
        return config.correlation_group if config else None
    
    def get_correlated_symbols(self, symbol: str) -> List[str]:
        """Get symbols correlated with the given symbol"""
        group = self.get_correlation_group(symbol)
        if not group:
            return []
        
        correlated = self.correlation_groups.get(group, [])
        return [s for s in correlated if s != symbol]
    
    def add_symbol(self, config: SymbolConfig) -> None:
        """Add a new symbol configuration"""
        self.symbols[config.symbol] = config
    
    def deactivate_symbol(self, symbol: str) -> None:
        """Deactivate a symbol from trading"""
        if symbol in self.symbols:
            self.symbols[symbol].active = False
    
    def activate_symbol(self, symbol: str) -> None:
        """Activate a symbol for trading"""
        if symbol in self.symbols:
            self.symbols[symbol].active = True
    
    def update_symbol_config(self, symbol: str, **kwargs) -> None:
        """Update symbol configuration parameters"""
        if symbol in self.symbols:
            config = self.symbols[symbol]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

# Global symbol manager instance
symbol_manager = SymbolManager()

# Convenience functions
def get_active_symbols() -> List[str]:
    """Get active symbols - convenience function"""
    return symbol_manager.get_active_symbols()

def get_symbol_config(symbol: str) -> Optional[SymbolConfig]:
    """Get symbol config - convenience function"""
    return symbol_manager.get_symbol_config(symbol)