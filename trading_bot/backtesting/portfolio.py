"""
Virtual Portfolio Manager
========================

Gestiona el portfolio virtual con capital inicial de $200 USD.
Todo es real excepto el dinero - usamos precios reales de Binance.

Como trader rentable, sé que la gestión del capital es CRÍTICA.
Este módulo simula exactamente cómo se comportaría nuestro dinero.

Autor: Trader Algorítmico Senior (10+ años)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pytz

from ..utils.logger import get_logger, LogContext
from ..utils.exceptions import TradingBotException

# Zona horaria de Ecuador
ECUADOR_TZ = pytz.timezone('America/Guayaquil')

class TradeStatus(Enum):
    """Estado de un trade"""
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"

class TradeSide(Enum):
    """Lado del trade"""
    LONG = "long"
    SHORT = "short"

@dataclass
class Trade:
    """
    Representa un trade individual
    Como trader senior, registro TODO para análisis posterior
    """
    # Identificación
    trade_id: str
    symbol: str
    side: TradeSide
    
    # Entrada
    entry_price: float
    entry_time: datetime
    quantity: float  # En crypto (ej: 0.01 BTC)
    position_size_usd: float  # En USD
    
    # Salida planificada
    stop_loss: float
    take_profit: float
    
    # Salida real (se llena al cerrar)
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None  # "stop_loss", "take_profit", "manual", "signal"
    
    # Performance
    pnl_usd: float = 0.0
    pnl_percent: float = 0.0
    fees_usd: float = 0.0
    
    # Estado
    status: TradeStatus = TradeStatus.OPEN
    
    # Análisis
    entry_score: float = 0.0  # Score del sistema al entrar
    entry_confidence: float = 0.0
    max_profit: float = 0.0  # Máximo profit alcanzado
    max_loss: float = 0.0  # Máxima pérdida alcanzada
    
    def calculate_pnl(self, current_price: float) -> Tuple[float, float]:
        """Calcula PnL actual sin cerrar el trade"""
        if self.side == TradeSide.LONG:
            price_change = current_price - self.entry_price
        else:  # SHORT
            price_change = self.entry_price - current_price
        
        pnl_usd = price_change * self.quantity
        pnl_percent = (price_change / self.entry_price) * 100
        
        return pnl_usd, pnl_percent
    
    def close(self, exit_price: float, exit_time: datetime, reason: str, fees: float = 0.0):
        """Cierra el trade y calcula PnL final"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        self.status = TradeStatus.CLOSED
        
        # Calcular PnL final
        if self.side == TradeSide.LONG:
            price_change = exit_price - self.entry_price
        else:  # SHORT
            price_change = self.entry_price - exit_price
        
        self.pnl_usd = (price_change * self.quantity) - fees
        self.pnl_percent = (price_change / self.entry_price) * 100
        self.fees_usd = fees

class VirtualPortfolio:
    """
    Portfolio virtual con gestión profesional de riesgo
    
    Como trader con 10+ años, implemento:
    - Position sizing dinámico
    - Risk management estricto
    - Tracking detallado de performance
    """
    
    def __init__(self, initial_capital: float = 200.0, 
                 max_position_pct: float = 0.10,
                 max_trades: int = 3):
        """
        Args:
            initial_capital: Capital inicial en USD ($200 por defecto)
            max_position_pct: Máximo % del capital por posición (10%)
            max_trades: Máximo de trades simultáneos (3)
        """
        self.logger = get_logger("VirtualPortfolio")
        
        # Capital
        self.initial_capital = initial_capital
        self.current_balance = initial_capital
        self.available_balance = initial_capital
        
        # Risk management
        self.max_position_pct = max_position_pct
        self.max_trades = max_trades
        
        # Trades
        self.open_trades: Dict[str, Trade] = {}
        self.closed_trades: List[Trade] = []
        self.trade_counter = 0
        
        # Performance tracking
        self.balance_history: List[Dict] = [{
            'timestamp': datetime.now(ECUADOR_TZ),
            'balance': initial_capital,
            'equity': initial_capital,
            'open_pnl': 0.0
        }]
        
        # Estadísticas
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'total_fees': 0.0,
            'max_drawdown': 0.0,
            'peak_balance': initial_capital,
            'lowest_balance': initial_capital
        }
        
        self.logger.info(f"Portfolio iniciado con ${initial_capital} USD")
    
    def can_open_trade(self, position_size_usd: float) -> bool:
        """
        Verifica si podemos abrir un nuevo trade
        
        Como trader profesional, SIEMPRE verifico:
        1. Suficiente balance disponible
        2. No exceder máximo de trades
        3. No exceder tamaño máximo de posición
        """
        # Verificar número de trades abiertos
        if len(self.open_trades) >= self.max_trades:
            self.logger.warning(f"Máximo de trades alcanzado: {self.max_trades}")
            return False
        
        # Verificar balance disponible
        if position_size_usd > self.available_balance:
            self.logger.warning(f"Balance insuficiente: ${self.available_balance:.2f}")
            return False
        
        # Verificar tamaño máximo de posición
        max_position = self.current_balance * self.max_position_pct
        if position_size_usd > max_position:
            self.logger.warning(f"Posición muy grande: ${position_size_usd:.2f} > ${max_position:.2f}")
            return False
        
        return True
    
    def open_trade(self, symbol: str, side: TradeSide, entry_price: float,
                   stop_loss: float, take_profit: float,
                   position_size_usd: Optional[float] = None,
                   entry_score: float = 0.0,
                   entry_confidence: float = 0.0) -> Optional[Trade]:
        """
        Abre un nuevo trade con gestión de riesgo
        
        Args:
            symbol: Par de trading (ej: BTCUSDT)
            side: LONG o SHORT
            entry_price: Precio de entrada (real de Binance)
            stop_loss: Precio de stop loss
            take_profit: Precio de take profit
            position_size_usd: Tamaño en USD (auto si None)
            entry_score: Score del sistema al entrar
            entry_confidence: Confianza del sistema
            
        Returns:
            Trade creado o None si no se pudo abrir
        """
        context = LogContext(component="portfolio", symbol=symbol)
        
        # Calcular tamaño de posición si no se especifica
        if position_size_usd is None:
            position_size_usd = self.current_balance * self.max_position_pct
        
        # Verificar que podemos abrir el trade
        if not self.can_open_trade(position_size_usd):
            return None
        
        # Calcular cantidad de crypto
        quantity = position_size_usd / entry_price
        
        # Calcular fees (0.1% Binance spot)
        fees = position_size_usd * 0.001
        
        # Crear trade
        self.trade_counter += 1
        trade_id = f"T{self.trade_counter:04d}"
        
        trade = Trade(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            entry_time=datetime.now(ECUADOR_TZ),
            quantity=quantity,
            position_size_usd=position_size_usd,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_score=entry_score,
            entry_confidence=entry_confidence,
            fees_usd=fees
        )
        
        # Actualizar portfolio
        self.open_trades[trade_id] = trade
        self.available_balance -= (position_size_usd + fees)
        self.stats['total_trades'] += 1
        self.stats['total_fees'] += fees
        
        self.logger.info(
            f"Trade abierto: {trade_id} {symbol} {side.value} "
            f"${position_size_usd:.2f} @ {entry_price:.2f}",
            context=context,
            extra_fields={
                'trade_id': trade_id,
                'quantity': quantity,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'available_balance': self.available_balance
            }
        )
        
        return trade
    
    def update_trades(self, current_prices: Dict[str, float], current_time: datetime):
        """
        Actualiza todos los trades abiertos con precios actuales
        Verifica stop loss y take profit
        
        Args:
            current_prices: Diccionario {symbol: price} con precios actuales
            current_time: Timestamp actual
        """
        trades_to_close = []
        total_open_pnl = 0.0
        
        for trade_id, trade in self.open_trades.items():
            if trade.symbol not in current_prices:
                continue
            
            current_price = current_prices[trade.symbol]
            
            # Calcular PnL actual
            pnl_usd, pnl_percent = trade.calculate_pnl(current_price)
            total_open_pnl += pnl_usd
            
            # Actualizar max profit/loss
            trade.max_profit = max(trade.max_profit, pnl_usd)
            trade.max_loss = min(trade.max_loss, pnl_usd)
            
            # Verificar stop loss y take profit
            should_close = False
            close_reason = None
            
            if trade.side == TradeSide.LONG:
                if current_price <= trade.stop_loss:
                    should_close = True
                    close_reason = "stop_loss"
                elif current_price >= trade.take_profit:
                    should_close = True
                    close_reason = "take_profit"
            else:  # SHORT
                if current_price >= trade.stop_loss:
                    should_close = True
                    close_reason = "stop_loss"
                elif current_price <= trade.take_profit:
                    should_close = True
                    close_reason = "take_profit"
            
            if should_close:
                trades_to_close.append((trade_id, current_price, close_reason))
        
        # Cerrar trades que tocaron SL o TP
        for trade_id, exit_price, reason in trades_to_close:
            self.close_trade(trade_id, exit_price, current_time, reason)
        
        # Actualizar equity
        current_equity = self.current_balance + total_open_pnl
        
        # Actualizar balance history
        self.balance_history.append({
            'timestamp': current_time,
            'balance': self.current_balance,
            'equity': current_equity,
            'open_pnl': total_open_pnl
        })
        
        # Actualizar estadísticas
        if current_equity > self.stats['peak_balance']:
            self.stats['peak_balance'] = current_equity
        if current_equity < self.stats['lowest_balance']:
            self.stats['lowest_balance'] = current_equity
        
        # Calcular drawdown
        drawdown = (self.stats['peak_balance'] - current_equity) / self.stats['peak_balance']
        if drawdown > self.stats['max_drawdown']:
            self.stats['max_drawdown'] = drawdown
    
    def close_trade(self, trade_id: str, exit_price: float, 
                   exit_time: datetime, reason: str) -> Optional[Trade]:
        """
        Cierra un trade y actualiza el portfolio
        
        Args:
            trade_id: ID del trade a cerrar
            exit_price: Precio de salida
            exit_time: Momento de salida
            reason: Razón del cierre
            
        Returns:
            Trade cerrado o None si no existe
        """
        if trade_id not in self.open_trades:
            return None
        
        trade = self.open_trades[trade_id]
        
        # Calcular fees de salida
        exit_fees = (trade.quantity * exit_price) * 0.001
        
        # Cerrar trade
        trade.close(exit_price, exit_time, reason, exit_fees)
        
        # Actualizar balance
        exit_value = trade.quantity * exit_price
        self.available_balance += exit_value - exit_fees
        self.current_balance += trade.pnl_usd
        
        # Actualizar estadísticas
        self.stats['total_pnl'] += trade.pnl_usd
        self.stats['total_fees'] += exit_fees
        
        if trade.pnl_usd > 0:
            self.stats['winning_trades'] += 1
        else:
            self.stats['losing_trades'] += 1
        
        # Mover a trades cerrados
        self.closed_trades.append(trade)
        del self.open_trades[trade_id]
        
        self.logger.info(
            f"Trade cerrado: {trade_id} {trade.symbol} "
            f"PnL: ${trade.pnl_usd:.2f} ({trade.pnl_percent:.2f}%) "
            f"Razón: {reason}",
            extra_fields={
                'trade_id': trade_id,
                'exit_price': exit_price,
                'pnl_usd': trade.pnl_usd,
                'current_balance': self.current_balance
            }
        )
        
        return trade
    
    def close_all_trades(self, current_prices: Dict[str, float], 
                        current_time: datetime, reason: str = "end_backtest"):
        """Cierra todos los trades abiertos (al final del backtest)"""
        trades_to_close = list(self.open_trades.keys())
        
        for trade_id in trades_to_close:
            trade = self.open_trades[trade_id]
            if trade.symbol in current_prices:
                exit_price = current_prices[trade.symbol]
                self.close_trade(trade_id, exit_price, current_time, reason)
    
    def get_statistics(self) -> Dict:
        """
        Obtiene estadísticas completas del portfolio
        
        Como trader profesional, estas son las métricas que REALMENTE importan
        """
        total_trades = self.stats['total_trades']
        winning_trades = self.stats['winning_trades']
        losing_trades = self.stats['losing_trades']
        
        # Win rate
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Profit factor
        total_wins = sum(t.pnl_usd for t in self.closed_trades if t.pnl_usd > 0)
        total_losses = abs(sum(t.pnl_usd for t in self.closed_trades if t.pnl_usd < 0))
        profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')
        
        # Average win/loss
        avg_win = (total_wins / winning_trades) if winning_trades > 0 else 0
        avg_loss = (total_losses / losing_trades) if losing_trades > 0 else 0
        
        # ROI
        roi = ((self.current_balance - self.initial_capital) / self.initial_capital * 100)
        
        return {
            'initial_capital': self.initial_capital,
            'final_balance': self.current_balance,
            'total_pnl': self.stats['total_pnl'],
            'roi_percent': roi,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown_pct': self.stats['max_drawdown'] * 100,
            'total_fees': self.stats['total_fees'],
            'peak_balance': self.stats['peak_balance'],
            'lowest_balance': self.stats['lowest_balance']
        }
    
    def get_trade_history(self) -> pd.DataFrame:
        """Retorna historial de trades como DataFrame para análisis"""
        if not self.closed_trades:
            return pd.DataFrame()
        
        trades_data = []
        for trade in self.closed_trades:
            trades_data.append({
                'trade_id': trade.trade_id,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'entry_time': trade.entry_time,
                'entry_price': trade.entry_price,
                'exit_time': trade.exit_time,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'position_size_usd': trade.position_size_usd,
                'pnl_usd': trade.pnl_usd,
                'pnl_percent': trade.pnl_percent,
                'exit_reason': trade.exit_reason,
                'entry_score': trade.entry_score,
                'entry_confidence': trade.entry_confidence,
                'fees_usd': trade.fees_usd,
                'max_profit': trade.max_profit,
                'max_loss': trade.max_loss
            })
        
        return pd.DataFrame(trades_data)
    
    def get_balance_curve(self) -> pd.DataFrame:
        """Retorna curva de balance/equity como DataFrame"""
        return pd.DataFrame(self.balance_history)