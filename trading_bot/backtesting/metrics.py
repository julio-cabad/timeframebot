"""
Calculador de Métricas de Performance
=====================================

Como trader con 10+ años, estas son las métricas que REALMENTE importan.
No te dejes engañar por métricas vanidosas - enfócate en lo que genera dinero.

Métricas clave:
- Win Rate: % de trades ganadores
- Profit Factor: Ganancias totales / Pérdidas totales
- Max Drawdown: Máxima pérdida desde un pico
- Sharpe Ratio: Retorno ajustado por riesgo
- Recovery Factor: Ganancia total / Max drawdown

Autor: Trader Algorítmico Senior
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PerformanceMetrics:
    """Métricas completas de performance"""
    # Básicas
    total_return: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    
    # Win/Loss
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Risk metrics
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    recovery_factor: float
    
    # Ratios avanzados
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Estadísticas de trades
    avg_trade_duration: float  # En horas
    avg_bars_in_trade: int
    consecutive_wins: int
    consecutive_losses: int
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    # Distribución
    win_loss_ratio: float
    expectancy: float
    payoff_ratio: float

class MetricsCalculator:
    """
    Calculador profesional de métricas de trading
    
    Como trader experimentado, sé que las métricas correctas
    son la diferencia entre el éxito y el fracaso.
    """
    
    @staticmethod
    def calculate_all_metrics(trades_df: pd.DataFrame, 
                            balance_curve: pd.DataFrame,
                            initial_capital: float = 200.0) -> PerformanceMetrics:
        """
        Calcula todas las métricas de performance
        
        Args:
            trades_df: DataFrame con historial de trades
            balance_curve: DataFrame con evolución del balance
            initial_capital: Capital inicial
            
        Returns:
            PerformanceMetrics con todas las métricas calculadas
        """
        if trades_df.empty:
            return MetricsCalculator._empty_metrics()
        
        # Métricas básicas
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_usd'] > 0])
        losing_trades = len(trades_df[trades_df['pnl_usd'] < 0])
        
        # PnL
        total_pnl = trades_df['pnl_usd'].sum()
        total_return_pct = (total_pnl / initial_capital) * 100
        
        # Win rate
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Average win/loss
        wins = trades_df[trades_df['pnl_usd'] > 0]['pnl_usd']
        losses = trades_df[trades_df['pnl_usd'] < 0]['pnl_usd']
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        largest_win = wins.max() if len(wins) > 0 else 0
        largest_loss = abs(losses.min()) if len(losses) > 0 else 0
        
        # Profit factor
        total_wins = wins.sum() if len(wins) > 0 else 0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')
        
        # Drawdown
        max_dd, max_dd_pct = MetricsCalculator._calculate_max_drawdown(balance_curve)
        
        # Recovery factor
        recovery_factor = (total_pnl / max_dd) if max_dd > 0 else float('inf')
        
        # Ratios avanzados
        sharpe = MetricsCalculator._calculate_sharpe_ratio(balance_curve)
        sortino = MetricsCalculator._calculate_sortino_ratio(balance_curve)
        calmar = (total_return_pct / max_dd_pct) if max_dd_pct > 0 else 0
        
        # Duración de trades
        if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
            durations = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600
            avg_duration = durations.mean()
        else:
            avg_duration = 0
        
        # Rachas
        consecutive_wins, consecutive_losses = MetricsCalculator._calculate_streaks(trades_df)
        max_consecutive_wins = MetricsCalculator._max_consecutive_wins(trades_df)
        max_consecutive_losses = MetricsCalculator._max_consecutive_losses(trades_df)
        
        # Win/Loss ratio
        win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else float('inf')
        
        # Expectancy (ganancia esperada por trade)
        expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
        
        # Payoff ratio
        payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0
        
        return PerformanceMetrics(
            total_return=total_pnl,
            total_return_pct=total_return_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            recovery_factor=recovery_factor,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            avg_trade_duration=avg_duration,
            avg_bars_in_trade=0,  # TODO: Calcular si es necesario
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            win_loss_ratio=win_loss_ratio,
            expectancy=expectancy,
            payoff_ratio=payoff_ratio
        )
    
    @staticmethod
    def _calculate_max_drawdown(balance_curve: pd.DataFrame) -> Tuple[float, float]:
        """
        Calcula el máximo drawdown
        
        Como trader, esta es LA métrica de riesgo más importante
        """
        if balance_curve.empty or 'equity' not in balance_curve.columns:
            return 0, 0
        
        equity = balance_curve['equity'].values
        
        # Calcular running maximum
        running_max = np.maximum.accumulate(equity)
        
        # Calcular drawdown
        drawdown = running_max - equity
        
        # Max drawdown
        max_dd = drawdown.max()
        
        # Max drawdown percentage
        max_dd_pct = (max_dd / running_max[drawdown.argmax()]) * 100 if running_max[drawdown.argmax()] > 0 else 0
        
        return max_dd, max_dd_pct
    
    @staticmethod
    def _calculate_sharpe_ratio(balance_curve: pd.DataFrame, 
                               risk_free_rate: float = 0.02) -> float:
        """
        Calcula Sharpe Ratio (retorno ajustado por riesgo)
        
        Un Sharpe > 1.5 es excelente para trading algorítmico
        """
        if balance_curve.empty or len(balance_curve) < 2:
            return 0
        
        # Calcular retornos
        returns = balance_curve['equity'].pct_change().dropna()
        
        if returns.empty:
            return 0
        
        # Retorno promedio
        avg_return = returns.mean()
        
        # Desviación estándar
        std_return = returns.std()
        
        if std_return == 0:
            return 0
        
        # Sharpe ratio anualizado (asumiendo trading diario)
        sharpe = (avg_return - risk_free_rate/252) / std_return * np.sqrt(252)
        
        return sharpe
    
    @staticmethod
    def _calculate_sortino_ratio(balance_curve: pd.DataFrame,
                                risk_free_rate: float = 0.02) -> float:
        """
        Calcula Sortino Ratio (solo penaliza volatilidad negativa)
        
        Mejor que Sharpe para evaluar estrategias de trading
        """
        if balance_curve.empty or len(balance_curve) < 2:
            return 0
        
        # Calcular retornos
        returns = balance_curve['equity'].pct_change().dropna()
        
        if returns.empty:
            return 0
        
        # Retorno promedio
        avg_return = returns.mean()
        
        # Downside deviation (solo retornos negativos)
        negative_returns = returns[returns < 0]
        
        if negative_returns.empty:
            return float('inf')  # No hay retornos negativos
        
        downside_std = negative_returns.std()
        
        if downside_std == 0:
            return 0
        
        # Sortino ratio anualizado
        sortino = (avg_return - risk_free_rate/252) / downside_std * np.sqrt(252)
        
        return sortino
    
    @staticmethod
    def _calculate_streaks(trades_df: pd.DataFrame) -> Tuple[int, int]:
        """Calcula rachas actuales de wins/losses"""
        if trades_df.empty:
            return 0, 0
        
        last_trades = trades_df['pnl_usd'].tail(10).values
        
        consecutive_wins = 0
        consecutive_losses = 0
        
        # Contar desde el último trade
        for pnl in reversed(last_trades):
            if pnl > 0:
                if consecutive_losses > 0:
                    break
                consecutive_wins += 1
            else:
                if consecutive_wins > 0:
                    break
                consecutive_losses += 1
        
        return consecutive_wins, consecutive_losses
    
    @staticmethod
    def _max_consecutive_wins(trades_df: pd.DataFrame) -> int:
        """Calcula máxima racha de trades ganadores"""
        if trades_df.empty:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for pnl in trades_df['pnl_usd']:
            if pnl > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    @staticmethod
    def _max_consecutive_losses(trades_df: pd.DataFrame) -> int:
        """Calcula máxima racha de trades perdedores"""
        if trades_df.empty:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for pnl in trades_df['pnl_usd']:
            if pnl < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    @staticmethod
    def _empty_metrics() -> PerformanceMetrics:
        """Retorna métricas vacías cuando no hay trades"""
        return PerformanceMetrics(
            total_return=0,
            total_return_pct=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            avg_win=0,
            avg_loss=0,
            largest_win=0,
            largest_loss=0,
            profit_factor=0,
            max_drawdown=0,
            max_drawdown_pct=0,
            recovery_factor=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            avg_trade_duration=0,
            avg_bars_in_trade=0,
            consecutive_wins=0,
            consecutive_losses=0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            win_loss_ratio=0,
            expectancy=0,
            payoff_ratio=0
        )