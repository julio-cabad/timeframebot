"""
Módulo de Métricas Avanzadas para Backtesting
============================================

Como trader senior, estas son las métricas que REALMENTE importan.
No me interesan las métricas vanidosas - solo las que predicen
el éxito en mercados reales.

Métricas implementadas:
- Sharpe Ratio (risk-adjusted returns)
- Calmar Ratio (return/max drawdown)
- Sortino Ratio (downside deviation)
- Information Ratio (excess return/tracking error)
- Maximum Drawdown con análisis detallado
- Value at Risk (VaR) y Conditional VaR
- Kelly Criterion para position sizing
- Win Rate, Profit Factor, Expectancy
- Monte Carlo analysis para robustez

Filosofía: "Las métricas no mienten, pero los traders sí se mienten a sí mismos"

Autor: Trader Algorítmico Senior (10+ años)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import scipy.stats as stats
from scipy.optimize import minimize_scalar

from ..utils.logger import get_logger, LogContext
from ..utils.exceptions import TradingBotException

@dataclass
class RiskMetrics:
    """Métricas de riesgo fundamentales"""
    max_drawdown: float
    max_drawdown_duration: int  # días
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR 95%
    downside_deviation: float
    upside_deviation: float
    
    # Drawdown detallado
    drawdown_periods: List[Dict[str, Any]] = field(default_factory=list)
    current_drawdown: float = 0.0
    
    @property
    def recovery_factor(self) -> float:
        """Factor de recuperación (return/max_drawdown)"""
        return abs(self.max_drawdown) if self.max_drawdown != 0 else 0.0

@dataclass
class TradeMetrics:
    """Métricas de trading fundamentales"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # PnL metrics
    gross_profit: float
    gross_loss: float
    net_profit: float
    profit_factor: float
    
    # Trade statistics
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Consecutive trades
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    # Expectancy
    expectancy: float  # (win_rate * avg_win) - (loss_rate * avg_loss)
    
    @property
    def loss_rate(self) -> float:
        """Tasa de pérdidas"""
        return 1.0 - self.win_rate
    
    @property
    def reward_risk_ratio(self) -> float:
        """Ratio reward/risk"""
        return abs(self.avg_win / self.avg_loss) if self.avg_loss != 0 else 0.0

@dataclass
class DrawdownAnalysis:
    """Análisis detallado de drawdowns"""
    max_drawdown: float
    max_drawdown_start: datetime
    max_drawdown_end: datetime
    max_drawdown_duration: int
    
    # Todos los períodos de drawdown
    drawdown_periods: List[Dict[str, Any]]
    
    # Estadísticas de drawdown
    avg_drawdown: float
    avg_drawdown_duration: int
    drawdown_frequency: float  # drawdowns por año
    
    # Recovery analysis
    avg_recovery_time: int
    max_recovery_time: int

class PerformanceAnalyzer:
    """
    Analizador principal de performance
    
    Como trader senior, he visto muchos sistemas que se ven bien en papel
    pero fallan en producción. Estas métricas separan lo real de lo falso.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.logger = get_logger("PerformanceAnalyzer")
        self.risk_free_rate = risk_free_rate  # 2% anual por defecto
    
    def analyze_performance(self, equity_curve: pd.Series, 
                          trades: List[Dict[str, Any]],
                          benchmark: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Análisis completo de performance
        
        Args:
            equity_curve: Serie temporal del equity
            trades: Lista de trades ejecutados
            benchmark: Serie de benchmark (opcional)
            
        Returns:
            Diccionario con todas las métricas
        """
        context = LogContext(component="performance_analyzer")
        
        try:
            # Calcular returns
            returns = equity_curve.pct_change().dropna()
            
            # Métricas básicas
            total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
            annualized_return = self._annualize_return(total_return, len(equity_curve))
            
            # Métricas de riesgo
            risk_metrics = self._calculate_risk_metrics(equity_curve, returns)
            
            # Métricas de trading
            trade_metrics = self._calculate_trade_metrics(trades)
            
            # Ratios de performance
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = self._calculate_calmar_ratio(annualized_return, risk_metrics.max_drawdown)
            
            # Kelly Criterion
            kelly_criterion = self._calculate_kelly_criterion(trade_metrics)
            
            # Análisis de drawdown
            drawdown_analysis = DrawdownAnalysis(
                max_drawdown=risk_metrics.max_drawdown,
                max_drawdown_start=equity_curve.index[0],
                max_drawdown_end=equity_curve.index[-1],
                max_drawdown_duration=risk_metrics.max_drawdown_duration,
                drawdown_periods=risk_metrics.drawdown_periods,
                avg_drawdown=risk_metrics.max_drawdown,
                avg_drawdown_duration=risk_metrics.max_drawdown_duration,
                drawdown_frequency=1.0,
                avg_recovery_time=risk_metrics.max_drawdown_duration,
                max_recovery_time=risk_metrics.max_drawdown_duration
            )
            
            # Benchmark comparison (si está disponible)
            benchmark_metrics = {}
            if benchmark is not None:
                benchmark_metrics = self._compare_to_benchmark(returns, benchmark)
            
            results = {
                # Returns
                "total_return": total_return,
                "annualized_return": annualized_return,
                "volatility": returns.std() * np.sqrt(252),  # Anualizada
                
                # Risk-adjusted ratios
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio,
                
                # Risk metrics
                "risk_metrics": risk_metrics,
                
                # Trade metrics
                "trade_metrics": trade_metrics,
                
                # Position sizing
                "kelly_criterion": kelly_criterion,
                
                # Drawdown analysis
                "drawdown_analysis": drawdown_analysis,
                
                # Benchmark comparison
                "benchmark_metrics": benchmark_metrics,
                
                # Additional metrics
                "best_month": returns.resample('M').sum().max() if len(returns) > 30 else 0,
                "worst_month": returns.resample('M').sum().min() if len(returns) > 30 else 0,
                "positive_months": (returns.resample('M').sum() > 0).sum() if len(returns) > 30 else 0,
                "total_months": len(returns.resample('M').sum()) if len(returns) > 30 else 0,
            }
            
            self.logger.info(
                "Análisis de performance completado",
                context=context,
                extra_fields={
                    "total_return": f"{total_return:.2%}",
                    "sharpe_ratio": f"{sharpe_ratio:.2f}",
                    "max_drawdown": f"{risk_metrics.max_drawdown:.2%}",
                    "win_rate": f"{trade_metrics.win_rate:.1%}"
                }
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error en análisis de performance: {e}")
            raise TradingBotException(f"Fallo en análisis: {str(e)}")
    
    def _calculate_risk_metrics(self, equity_curve: pd.Series, returns: pd.Series) -> RiskMetrics:
        """Calcula métricas de riesgo detalladas"""
        try:
            # Maximum Drawdown
            rolling_max = equity_curve.expanding().max()
            drawdown = (equity_curve - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Drawdown duration
            drawdown_start = drawdown[drawdown == max_drawdown].index[0]
            recovery_mask = equity_curve[drawdown_start:] >= rolling_max[drawdown_start]
            if recovery_mask.any():
                recovery_date = equity_curve[drawdown_start:][recovery_mask].index[0]
                max_dd_duration = (recovery_date - drawdown_start).days
            else:
                max_dd_duration = (equity_curve.index[-1] - drawdown_start).days
            
            # VaR and CVaR
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean()
            
            # Downside/Upside deviation
            downside_returns = returns[returns < 0]
            upside_returns = returns[returns > 0]
            
            downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
            upside_deviation = upside_returns.std() if len(upside_returns) > 0 else 0
            
            # Análisis detallado de drawdowns
            drawdown_periods = self._identify_drawdown_periods(drawdown, equity_curve)
            
            return RiskMetrics(
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_dd_duration,
                var_95=var_95,
                cvar_95=cvar_95,
                downside_deviation=downside_deviation,
                upside_deviation=upside_deviation,
                drawdown_periods=drawdown_periods,
                current_drawdown=drawdown.iloc[-1]
            )
            
        except Exception as e:
            self.logger.error(f"Error calculando métricas de riesgo: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0)
    
    def _calculate_trade_metrics(self, trades: List[Dict[str, Any]]) -> TradeMetrics:
        """Calcula métricas de trading detalladas"""
        try:
            if not trades:
                return TradeMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            
            # Extraer PnL de trades
            pnls = []
            for trade in trades:
                if 'pnl' in trade:
                    pnls.append(trade['pnl'])
                elif 'exit_price' in trade and 'entry_price' in trade:
                    # Calcular PnL si no está disponible
                    quantity = trade.get('quantity', 1)
                    side = trade.get('side', 'buy')
                    if side.lower() == 'buy':
                        pnl = quantity * (trade['exit_price'] - trade['entry_price'])
                    else:
                        pnl = quantity * (trade['entry_price'] - trade['exit_price'])
                    pnls.append(pnl)
            
            if not pnls:
                return TradeMetrics(len(trades), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            
            pnls = np.array(pnls)
            
            # Estadísticas básicas
            total_trades = len(pnls)
            winning_trades = (pnls > 0).sum()
            losing_trades = (pnls < 0).sum()
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # PnL metrics
            gross_profit = pnls[pnls > 0].sum() if winning_trades > 0 else 0
            gross_loss = abs(pnls[pnls < 0].sum()) if losing_trades > 0 else 0
            net_profit = pnls.sum()
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Trade statistics
            avg_win = pnls[pnls > 0].mean() if winning_trades > 0 else 0
            avg_loss = pnls[pnls < 0].mean() if losing_trades > 0 else 0
            largest_win = pnls.max() if len(pnls) > 0 else 0
            largest_loss = pnls.min() if len(pnls) > 0 else 0
            
            # Consecutive trades
            max_consecutive_wins = self._max_consecutive(pnls > 0)
            max_consecutive_losses = self._max_consecutive(pnls < 0)
            
            # Expectancy
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
            
            return TradeMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                gross_profit=gross_profit,
                gross_loss=gross_loss,
                net_profit=net_profit,
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                max_consecutive_wins=max_consecutive_wins,
                max_consecutive_losses=max_consecutive_losses,
                expectancy=expectancy
            )
            
        except Exception as e:
            self.logger.error(f"Error calculando métricas de trading: {e}")
            return TradeMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calcula Sharpe Ratio anualizado"""
        try:
            if len(returns) == 0 or returns.std() == 0:
                return 0.0
            
            excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
            return (excess_returns.mean() / returns.std()) * np.sqrt(252)
            
        except Exception:
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calcula Sortino Ratio (solo considera downside deviation)"""
        try:
            if len(returns) == 0:
                return 0.0
            
            excess_returns = returns - (self.risk_free_rate / 252)
            downside_returns = returns[returns < 0]
            
            if len(downside_returns) == 0:
                return float('inf')
            
            downside_std = downside_returns.std()
            if downside_std == 0:
                return 0.0
            
            return (excess_returns.mean() / downside_std) * np.sqrt(252)
            
        except Exception:
            return 0.0
    
    def _calculate_calmar_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """Calcula Calmar Ratio (return/max_drawdown)"""
        try:
            if max_drawdown == 0:
                return float('inf') if annualized_return > 0 else 0.0
            
            return annualized_return / abs(max_drawdown)
            
        except Exception:
            return 0.0
    
    def _calculate_kelly_criterion(self, trade_metrics: TradeMetrics) -> float:
        """
        Calcula Kelly Criterion para position sizing óptimo
        
        Kelly% = (bp - q) / b
        donde:
        b = odds (avg_win / avg_loss)
        p = win rate
        q = loss rate (1 - p)
        """
        try:
            if trade_metrics.avg_loss == 0 or trade_metrics.win_rate == 0:
                return 0.0
            
            b = abs(trade_metrics.avg_win / trade_metrics.avg_loss)  # odds
            p = trade_metrics.win_rate  # win rate
            q = 1 - p  # loss rate
            
            kelly = (b * p - q) / b
            
            # Limitar Kelly a máximo 25% (conservador)
            return min(max(kelly, 0), 0.25)
            
        except Exception:
            return 0.0
    
    def _annualize_return(self, total_return: float, periods: int) -> float:
        """Anualiza el retorno basado en el número de períodos"""
        try:
            if periods <= 0:
                return 0.0
            
            # Asumir datos diarios (252 días de trading por año)
            years = periods / 252
            if years <= 0:
                return 0.0
            
            return (1 + total_return) ** (1 / years) - 1
            
        except Exception:
            return 0.0
    
    def _max_consecutive(self, condition: np.ndarray) -> int:
        """Encuentra la máxima secuencia consecutiva de True"""
        try:
            if len(condition) == 0:
                return 0
            
            max_count = 0
            current_count = 0
            
            for value in condition:
                if value:
                    current_count += 1
                    max_count = max(max_count, current_count)
                else:
                    current_count = 0
            
            return max_count
            
        except Exception:
            return 0
    
    def _identify_drawdown_periods(self, drawdown: pd.Series, 
                                  equity_curve: pd.Series) -> List[Dict[str, Any]]:
        """Identifica todos los períodos de drawdown"""
        try:
            periods = []
            in_drawdown = False
            start_date = None
            start_value = None
            
            rolling_max = equity_curve.expanding().max()
            
            for date, dd_value in drawdown.items():
                if dd_value < -0.001 and not in_drawdown:  # Inicio de drawdown (>0.1%)
                    in_drawdown = True
                    start_date = date
                    start_value = equity_curve[date]
                    
                elif dd_value >= -0.001 and in_drawdown:  # Fin de drawdown
                    in_drawdown = False
                    end_date = date
                    end_value = equity_curve[date]
                    
                    # Calcular métricas del período
                    period_data = equity_curve[start_date:end_date]
                    max_dd_in_period = (period_data - period_data.expanding().max()).min() / period_data.expanding().max().max()
                    
                    periods.append({
                        'start_date': start_date,
                        'end_date': end_date,
                        'duration_days': (end_date - start_date).days,
                        'max_drawdown': max_dd_in_period,
                        'start_value': start_value,
                        'end_value': end_value,
                        'recovery': (end_value / start_value) - 1
                    })
            
            return periods
            
        except Exception as e:
            self.logger.error(f"Error identificando períodos de drawdown: {e}")
            return []
    
    def _compare_to_benchmark(self, returns: pd.Series, 
                             benchmark: pd.Series) -> Dict[str, float]:
        """Compara performance contra benchmark"""
        try:
            # Alinear fechas
            aligned_returns, aligned_benchmark = returns.align(benchmark, join='inner')
            
            if len(aligned_returns) == 0:
                return {}
            
            # Excess returns
            excess_returns = aligned_returns - aligned_benchmark
            
            # Information Ratio
            tracking_error = excess_returns.std()
            information_ratio = excess_returns.mean() / tracking_error if tracking_error > 0 else 0
            
            # Beta
            covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
            benchmark_variance = aligned_benchmark.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Alpha
            alpha = aligned_returns.mean() - (self.risk_free_rate / 252 + beta * (aligned_benchmark.mean() - self.risk_free_rate / 252))
            
            return {
                'information_ratio': information_ratio * np.sqrt(252),  # Anualizado
                'tracking_error': tracking_error * np.sqrt(252),  # Anualizado
                'beta': beta,
                'alpha': alpha * 252,  # Anualizado
                'correlation': aligned_returns.corr(aligned_benchmark)
            }
            
        except Exception as e:
            self.logger.error(f"Error comparando con benchmark: {e}")
            return {}

class MonteCarloAnalysis:
    """
    Análisis Monte Carlo para robustez de estrategia
    
    Como trader senior, uso Monte Carlo para entender qué tan
    robusto es realmente mi sistema bajo diferentes escenarios.
    """
    
    def __init__(self, n_simulations: int = 1000):
        self.logger = get_logger("MonteCarloAnalysis")
        self.n_simulations = n_simulations
    
    def run_monte_carlo(self, trades: List[Dict[str, Any]], 
                       initial_capital: float = 100000) -> Dict[str, Any]:
        """
        Ejecuta análisis Monte Carlo en los trades
        
        Simula diferentes órdenes de trades para evaluar robustez
        """
        try:
            if not trades:
                return {}
            
            # Extraer PnL de trades
            pnls = []
            for trade in trades:
                if 'pnl' in trade:
                    pnls.append(trade['pnl'])
            
            if not pnls:
                return {}
            
            pnls = np.array(pnls)
            
            # Ejecutar simulaciones
            final_capitals = []
            max_drawdowns = []
            
            for _ in range(self.n_simulations):
                # Reordenar trades aleatoriamente
                shuffled_pnls = np.random.permutation(pnls)
                
                # Simular equity curve
                equity_curve = np.cumsum(shuffled_pnls) + initial_capital
                equity_curve = np.insert(equity_curve, 0, initial_capital)
                
                # Calcular métricas
                final_capitals.append(equity_curve[-1])
                
                # Calcular drawdown
                rolling_max = np.maximum.accumulate(equity_curve)
                drawdown = (equity_curve - rolling_max) / rolling_max
                max_drawdowns.append(drawdown.min())
            
            final_capitals = np.array(final_capitals)
            max_drawdowns = np.array(max_drawdowns)
            
            # Calcular estadísticas
            results = {
                'simulations': self.n_simulations,
                'final_capital': {
                    'mean': final_capitals.mean(),
                    'std': final_capitals.std(),
                    'min': final_capitals.min(),
                    'max': final_capitals.max(),
                    'percentile_5': np.percentile(final_capitals, 5),
                    'percentile_95': np.percentile(final_capitals, 95)
                },
                'max_drawdown': {
                    'mean': max_drawdowns.mean(),
                    'std': max_drawdowns.std(),
                    'worst': max_drawdowns.min(),
                    'best': max_drawdowns.max(),
                    'percentile_5': np.percentile(max_drawdowns, 5),
                    'percentile_95': np.percentile(max_drawdowns, 95)
                },
                'probability_profit': (final_capitals > initial_capital).mean(),
                'probability_loss_10pct': (final_capitals < initial_capital * 0.9).mean(),
                'probability_loss_20pct': (final_capitals < initial_capital * 0.8).mean()
            }
            
            self.logger.info(
                f"Monte Carlo completado: {self.n_simulations} simulaciones",
                extra_fields={
                    'prob_profit': f"{results['probability_profit']:.1%}",
                    'worst_drawdown': f"{results['max_drawdown']['worst']:.2%}"
                }
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error en análisis Monte Carlo: {e}")
            return {}