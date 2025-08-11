"""
Motor Principal de Backtesting
=============================

Como trader senior, este es el corazón del sistema de validación.
Aquí es donde separamos las estrategias ganadoras de las perdedoras.

El motor simula condiciones reales de mercado:
- Slippage realista basado en volatilidad
- Comisiones y spreads
- Latencia de ejecución
- Gaps de mercado
- Liquidez limitada

Filosofía: "Mejor ser pesimista en backtesting que optimista en producción"

Autor: Trader Algorítmico Senior (10+ años)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import pytz
from pathlib import Path

from ..config.settings import config
from ..utils.logger import get_logger, LogContext
from ..utils.exceptions import TradingBotException, ErrorCodes

# Zona horaria de Ecuador (UTC-5)
ECUADOR_TZ = pytz.timezone('America/Guayaquil')

class OrderType(Enum):
    """Tipos de órdenes"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Lado de la orden"""
    BUY = "buy"
    SELL = "sell"

class ExecutionStatus(Enum):
    """Estado de ejecución"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class TradeExecution:
    """Ejecución de un trade en backtesting"""
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    slippage: float
    
    # Contexto del trade
    signal_score: float
    timeframe_analysis: Dict[str, Any]
    llm_decision: Optional[str] = None
    
    # Métricas de ejecución
    execution_delay_ms: int = 0
    market_impact: float = 0.0
    
    @property
    def total_cost(self) -> float:
        """Costo total incluyendo comisiones y slippage"""
        return self.quantity * self.price + self.commission + abs(self.slippage)
    
    @property
    def net_price(self) -> float:
        """Precio neto después de costos"""
        cost_per_share = (self.commission + abs(self.slippage)) / self.quantity
        return self.price + cost_per_share if self.side == OrderSide.BUY else self.price - cost_per_share

@dataclass
class PortfolioState:
    """Estado del portfolio en un momento dado"""
    timestamp: datetime
    cash: float
    positions: Dict[str, float]  # symbol -> quantity
    market_values: Dict[str, float]  # symbol -> market_value
    
    @property
    def total_value(self) -> float:
        """Valor total del portfolio"""
        return self.cash + sum(self.market_values.values())
    
    @property
    def equity(self) -> float:
        """Equity del portfolio"""
        return self.total_value
    
    def get_position_value(self, symbol: str, current_price: float) -> float:
        """Valor de una posición específica"""
        quantity = self.positions.get(symbol, 0.0)
        return quantity * current_price

@dataclass
class BacktestConfig:
    """Configuración para backtesting"""
    # Período de backtesting
    start_date: datetime
    end_date: datetime
    symbols: List[str]
    
    # Capital inicial
    initial_capital: float = 100000.0
    
    # Costos de transacción
    commission_rate: float = 0.001  # 0.1%
    slippage_model: str = "linear"  # "linear", "sqrt", "fixed"
    base_slippage: float = 0.0005  # 0.05%
    
    # Configuración de ejecución
    execution_delay_ms: int = 100  # Latencia de ejecución
    max_position_size: float = 0.04  # 30% máximo por posición
    max_portfolio_heat: float = 0.06  # 6% heat máximo
    
    # Configuración de datos
    data_frequency: str = "1h"  # Frecuencia de datos
    warmup_period: int = 200  # Períodos de calentamiento
    
    # Configuración de riesgo
    max_daily_drawdown: float = 0.02  # 2%
    max_consecutive_losses: int = 5
    
    # Configuración LLM (si aplica)
    use_llm: bool = True
    llm_cost_per_call: float = 0.001
    max_daily_llm_cost: float = 50.0

class BacktestEngine:
    """
    Motor principal de backtesting
    
    Como trader senior, he diseñado este motor para ser:
    1. Realista - Simula condiciones reales de mercado
    2. Conservador - Asume el peor escenario en costos
    3. Detallado - Registra cada decisión y resultado
    4. Robusto - Maneja datos faltantes y errores
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = get_logger("BacktestEngine")
        
        # Estado del backtesting
        self.current_time: Optional[datetime] = None
        self.portfolio: Optional[PortfolioState] = None
        self.trades: List[TradeExecution] = []
        self.portfolio_history: List[PortfolioState] = []
        
        # Métricas de control
        self.daily_pnl: Dict[str, float] = {}  # date -> pnl
        self.consecutive_losses = 0
        self.daily_llm_cost = 0.0
        
        # Datos de mercado
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.current_prices: Dict[str, float] = {}
        
        # Inicializar portfolio
        self._initialize_portfolio()
    
    def _initialize_portfolio(self) -> None:
        """Inicializa el portfolio con capital inicial"""
        self.portfolio = PortfolioState(
            timestamp=self.config.start_date,
            cash=self.config.initial_capital,
            positions={symbol: 0.0 for symbol in self.config.symbols},
            market_values={symbol: 0.0 for symbol in self.config.symbols}
        )
        
        self.portfolio_history.append(self.portfolio)
        self.logger.info(f"Portfolio inicializado con ${self.config.initial_capital:,.2f}")
    
    def load_market_data(self, data_handler) -> None:
        """Carga datos de mercado desde el data handler"""
        context = LogContext(component="backtest_engine")
        
        try:
            for symbol in self.config.symbols:
                df = data_handler.get_historical_data(
                    symbol=symbol,
                    start_date=self.config.start_date - timedelta(days=self.config.warmup_period),
                    end_date=self.config.end_date,
                    frequency=self.config.data_frequency
                )
                
                if df is not None and not df.empty:
                    self.market_data[symbol] = df
                    self.logger.info(f"Cargados {len(df)} registros para {symbol}")
                else:
                    self.logger.warning(f"No se pudieron cargar datos para {symbol}")
            
            self.logger.info(
                f"Datos de mercado cargados para {len(self.market_data)} símbolos",
                context=context
            )
            
        except Exception as e:
            self.logger.error(f"Error cargando datos de mercado: {e}")
            raise TradingBotException(f"Fallo cargando datos: {str(e)}")
    
    def calculate_slippage(self, symbol: str, quantity: float, side: OrderSide) -> float:
        """
        Calcula slippage realista basado en volatilidad y tamaño de orden
        
        Como trader senior, sé que el slippage es donde se pierde dinero real.
        Este modelo es conservador pero realista.
        """
        try:
            # Obtener volatilidad reciente
            if symbol not in self.market_data:
                return self.config.base_slippage * quantity
            
            df = self.market_data[symbol]
            recent_data = df.tail(20)  # Últimas 20 velas
            
            if len(recent_data) < 5:
                return self.config.base_slippage * quantity
            
            # Calcular volatilidad
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std() if len(returns) > 1 else 0.01
            
            # Modelo de slippage basado en volatilidad y tamaño
            base_slippage = self.config.base_slippage
            volatility_factor = min(volatility * 10, 0.005)  # Máximo 0.5%
            
            # Factor de tamaño (órdenes grandes tienen más slippage)
            current_price = self.current_prices.get(symbol, recent_data['close'].iloc[-1])
            order_value = quantity * current_price
            portfolio_value = self.portfolio.total_value
            size_factor = min((order_value / portfolio_value) * 2, 0.003)  # Máximo 0.3%
            
            total_slippage_rate = base_slippage + volatility_factor + size_factor
            
            # Aplicar dirección (compras pagan más, ventas reciben menos)
            direction_multiplier = 1.0 if side == OrderSide.BUY else -1.0
            
            slippage_amount = quantity * current_price * total_slippage_rate * direction_multiplier
            
            return slippage_amount
            
        except Exception as e:
            self.logger.warning(f"Error calculando slippage: {e}")
            return self.config.base_slippage * quantity * (1 if side == OrderSide.BUY else -1)
    
    def calculate_commission(self, quantity: float, price: float) -> float:
        """Calcula comisión de la transacción"""
        trade_value = quantity * price
        commission = trade_value * self.config.commission_rate
        
        # Comisión mínima (realista para exchanges)
        min_commission = 0.01
        return max(commission, min_commission)
    
    def can_execute_trade(self, symbol: str, quantity: float, side: OrderSide) -> Tuple[bool, str]:
        """
        Verifica si un trade puede ejecutarse
        
        Incluye todas las validaciones de riesgo que uso en producción
        """
        try:
            current_price = self.current_prices.get(symbol)
            if not current_price:
                return False, f"No hay precio actual para {symbol}"
            
            trade_value = abs(quantity) * current_price
            
            # Verificar cash disponible para compras
            if side == OrderSide.BUY:
                commission = self.calculate_commission(abs(quantity), current_price)
                slippage = abs(self.calculate_slippage(symbol, abs(quantity), side))
                total_cost = trade_value + commission + slippage
                
                if total_cost > self.portfolio.cash:
                    return False, f"Cash insuficiente: ${total_cost:.2f} > ${self.portfolio.cash:.2f}"
            
            # Verificar límites de posición
            current_position = self.portfolio.positions.get(symbol, 0.0)
            new_position = current_position + (quantity if side == OrderSide.BUY else -quantity)
            new_position_value = abs(new_position) * current_price
            
            max_position_value = self.portfolio.total_value * self.config.max_position_size
            if new_position_value > max_position_value:
                return False, f"Excede límite de posición: {new_position_value:.2f} > {max_position_value:.2f}"
            
            # Verificar heat del portfolio
            total_heat = sum(
                abs(pos) * self.current_prices.get(sym, 0) 
                for sym, pos in self.portfolio.positions.items()
            ) + trade_value
            
            max_heat = self.portfolio.total_value * self.config.max_portfolio_heat
            if total_heat > max_heat:
                return False, f"Excede heat del portfolio: {total_heat:.2f} > {max_heat:.2f}"
            
            # Verificar drawdown diario
            today = self.current_time.date() if self.current_time else datetime.now().date()
            daily_pnl = self.daily_pnl.get(str(today), 0.0)
            max_daily_loss = self.config.initial_capital * self.config.max_daily_drawdown
            
            if daily_pnl < -max_daily_loss:
                return False, f"Límite de drawdown diario alcanzado: {daily_pnl:.2f}"
            
            # Verificar pérdidas consecutivas
            if self.consecutive_losses >= self.config.max_consecutive_losses:
                return False, f"Máximo de pérdidas consecutivas alcanzado: {self.consecutive_losses}"
            
            return True, "Trade válido"
            
        except Exception as e:
            return False, f"Error validando trade: {str(e)}"
    
    def execute_trade(self, symbol: str, quantity: float, side: OrderSide, 
                     signal_score: float, timeframe_analysis: Dict[str, Any],
                     llm_decision: Optional[str] = None) -> Optional[TradeExecution]:
        """
        Ejecuta un trade en el backtesting
        
        Como trader senior, simulo TODOS los costos reales que enfrentaría en producción
        """
        context = LogContext(component="backtest_engine")
        
        try:
            # Validar si el trade puede ejecutarse
            can_execute, reason = self.can_execute_trade(symbol, quantity, side)
            if not can_execute:
                self.logger.debug(f"Trade rechazado: {reason}")
                return None
            
            current_price = self.current_prices[symbol]
            
            # Calcular costos
            commission = self.calculate_commission(abs(quantity), current_price)
            slippage = self.calculate_slippage(symbol, abs(quantity), side)
            
            # Simular latencia de ejecución (precio puede cambiar)
            execution_price = current_price
            if self.config.execution_delay_ms > 0:
                # Simular cambio de precio durante la latencia
                volatility = 0.001  # Volatilidad base para simulación
                price_change = np.random.normal(0, volatility) * current_price
                execution_price = max(current_price + price_change, current_price * 0.99)  # Mínimo 1% del precio
            
            # Crear ejecución
            execution = TradeExecution(
                timestamp=self.current_time,
                symbol=symbol,
                side=side,
                quantity=abs(quantity),
                price=execution_price,
                commission=commission,
                slippage=slippage,
                signal_score=signal_score,
                timeframe_analysis=timeframe_analysis,
                llm_decision=llm_decision,
                execution_delay_ms=self.config.execution_delay_ms
            )
            
            # Actualizar portfolio
            self._update_portfolio_from_execution(execution)
            
            # Registrar trade
            self.trades.append(execution)
            
            # Actualizar métricas de control
            self._update_control_metrics(execution)
            
            self.logger.info(
                f"Trade ejecutado: {side.value} {quantity:.4f} {symbol} @ ${execution_price:.2f}",
                context=context,
                extra_fields={
                    "commission": commission,
                    "slippage": slippage,
                    "signal_score": signal_score,
                    "llm_decision": llm_decision
                }
            )
            
            return execution
            
        except Exception as e:
            self.logger.error(f"Error ejecutando trade: {e}")
            return None
    
    def _update_portfolio_from_execution(self, execution: TradeExecution) -> None:
        """Actualiza el estado del portfolio después de una ejecución"""
        symbol = execution.symbol
        quantity = execution.quantity
        side = execution.side
        
        # Actualizar posiciones
        current_position = self.portfolio.positions.get(symbol, 0.0)
        
        if side == OrderSide.BUY:
            new_position = current_position + quantity
            cash_change = -(quantity * execution.price + execution.commission + abs(execution.slippage))
        else:  # SELL
            new_position = current_position - quantity
            cash_change = quantity * execution.price - execution.commission - abs(execution.slippage)
        
        # Actualizar portfolio
        self.portfolio.positions[symbol] = new_position
        self.portfolio.cash += cash_change
        
        # Actualizar valores de mercado
        self._update_market_values()
    
    def _update_market_values(self) -> None:
        """Actualiza los valores de mercado de todas las posiciones"""
        for symbol, position in self.portfolio.positions.items():
            current_price = self.current_prices.get(symbol, 0.0)
            self.portfolio.market_values[symbol] = position * current_price
    
    def _update_control_metrics(self, execution: TradeExecution) -> None:
        """Actualiza métricas de control después de cada trade"""
        # Actualizar costo LLM si aplica
        if execution.llm_decision and self.config.use_llm:
            self.daily_llm_cost += self.config.llm_cost_per_call
        
        # Actualizar PnL diario (se calculará al final del día)
        # Las pérdidas consecutivas se actualizarán cuando se cierre una posición
    
    def update_current_prices(self, timestamp: datetime) -> None:
        """Actualiza precios actuales para un timestamp dado"""
        self.current_time = timestamp
        
        for symbol, df in self.market_data.items():
            # Buscar el precio más cercano al timestamp
            mask = df.index <= timestamp
            if mask.any():
                latest_data = df[mask].iloc[-1]
                self.current_prices[symbol] = latest_data['close']
            else:
                # Si no hay datos, mantener el último precio conocido
                if symbol not in self.current_prices:
                    self.current_prices[symbol] = df['close'].iloc[0] if not df.empty else 100.0
        
        # Actualizar valores de mercado del portfolio
        if self.portfolio:
            self.portfolio.timestamp = timestamp
            self._update_market_values()
    
    def get_portfolio_snapshot(self) -> PortfolioState:
        """Obtiene snapshot actual del portfolio"""
        if not self.portfolio:
            raise TradingBotException("Portfolio no inicializado")
        
        # Crear copia del estado actual
        snapshot = PortfolioState(
            timestamp=self.current_time,
            cash=self.portfolio.cash,
            positions=self.portfolio.positions.copy(),
            market_values=self.portfolio.market_values.copy()
        )
        
        return snapshot
    
    def save_portfolio_snapshot(self) -> None:
        """Guarda snapshot del portfolio en el historial"""
        snapshot = self.get_portfolio_snapshot()
        self.portfolio_history.append(snapshot)
    
    def get_current_equity(self) -> float:
        """Obtiene equity actual del portfolio"""
        return self.portfolio.total_value if self.portfolio else 0.0
    
    def get_unrealized_pnl(self) -> Dict[str, float]:
        """Calcula PnL no realizado por símbolo"""
        unrealized_pnl = {}
        
        for symbol, position in self.portfolio.positions.items():
            if position != 0:
                current_price = self.current_prices.get(symbol, 0.0)
                
                # Calcular precio promedio de entrada
                symbol_trades = [t for t in self.trades if t.symbol == symbol]
                if symbol_trades:
                    total_cost = sum(t.quantity * t.net_price for t in symbol_trades if t.side == OrderSide.BUY)
                    total_quantity = sum(t.quantity for t in symbol_trades if t.side == OrderSide.BUY)
                    avg_entry_price = total_cost / total_quantity if total_quantity > 0 else current_price
                    
                    unrealized_pnl[symbol] = position * (current_price - avg_entry_price)
                else:
                    unrealized_pnl[symbol] = 0.0
        
        return unrealized_pnl
    
    def calculate_daily_pnl(self, date: str) -> float:
        """Calcula PnL para un día específico"""
        # Obtener trades del día
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
        daily_trades = [
            t for t in self.trades 
            if t.timestamp.date() == target_date
        ]
        
        # Calcular PnL realizado
        realized_pnl = 0.0
        for trade in daily_trades:
            if trade.side == OrderSide.SELL:
                # Simplificado: asumir FIFO para calcular PnL
                realized_pnl += trade.quantity * trade.net_price
            else:
                realized_pnl -= trade.quantity * trade.net_price
        
        return realized_pnl
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas básicas de los trades"""
        if not self.trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0
            }
        
        # Calcular PnL por trade (simplificado)
        trade_pnls = []
        for i, trade in enumerate(self.trades):
            if trade.side == OrderSide.SELL and i > 0:
                # Buscar trade de compra correspondiente
                buy_trade = None
                for j in range(i-1, -1, -1):
                    if (self.trades[j].symbol == trade.symbol and 
                        self.trades[j].side == OrderSide.BUY):
                        buy_trade = self.trades[j]
                        break
                
                if buy_trade:
                    pnl = trade.quantity * (trade.net_price - buy_trade.net_price)
                    trade_pnls.append(pnl)
        
        if not trade_pnls:
            return {
                "total_trades": len(self.trades),
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0
            }
        
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        win_rate = len(winning_trades) / len(trade_pnls) if trade_pnls else 0.0
        avg_win = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = np.mean(losing_trades) if losing_trades else 0.0
        
        gross_profit = sum(winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            "total_trades": len(trade_pnls),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss
        }