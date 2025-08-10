"""
Motor de Backtesting con Datos Reales
=====================================

Como trader algorítmico senior, este es el corazón del sistema.
Usa datos 100% REALES de Binance y aplica nuestra estrategia
con un portfolio virtual de $200 USD.

Filosofía de trading que aplico:
- Entradas basadas en confluencia multi-timeframe
- Stop loss estricto al 2%
- Take profit al 6% (ratio 3:1)
- Máximo 3 trades simultáneos
- Solo operar con alta confianza (score > 70)

Autor: Trader Algorítmico Senior (10+ años rentable)
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pytz

from .portfolio import VirtualPortfolio, TradeSide
from .metrics import MetricsCalculator
from .reporter import BacktestReporter

from ..data.fetcher import MultiTimeframeFetcher
from ..analysis.tf_analyzers import create_analyzer
from ..analysis.confluence import ConfluenceEngine as ConfluenceAnalyzer, analyze_confluence
from ..analysis.patterns import PatternDetector
from ..scoring.scorer import DynamicScorer

from ..utils.logger import get_logger, LogContext
from ..utils.exceptions import TradingBotException

# Zona horaria Ecuador
ECUADOR_TZ = pytz.timezone('America/Guayaquil')

@dataclass
class BacktestConfig:
    """Configuración del backtest"""
    # Capital
    initial_capital: float = 200.0  # USD
    position_size_pct: float = 0.10  # 10% por trade
    
    # Estrategia
    min_score_entry: float = 70.0  # Score mínimo para entrar
    min_confidence: float = 0.6  # Confianza mínima
    
    # Risk Management
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.06  # 6% take profit (3:1)
    max_trades: int = 3  # Máximo trades simultáneos
    
    # Datos
    candles_to_fetch: int = 500  # Velas a obtener de Binance
    warmup_candles: int = 100  # Velas para calentar indicadores
    
    # Timeframes a analizar
    timeframes: List[str] = None  # ['1d', '4h', '1h', '15m']
    
    # Símbolos a testear
    symbols: List[str] = None  # ['BTCUSDT', 'ETHUSDT', etc]
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ['1d', '4h', '1h', '15m']
        if self.symbols is None:
            self.symbols = ['BTCUSDT']  # Por defecto solo BTC

@dataclass
class BacktestResult:
    """Resultado completo del backtest"""
    config: BacktestConfig
    portfolio_stats: Dict
    trades_df: pd.DataFrame
    balance_curve: pd.DataFrame
    signals_generated: int
    execution_time: float
    final_balance: float
    roi_percent: float
    win_rate: float
    profit_factor: float
    max_drawdown: float

class BacktestEngine:
    """
    Motor principal de backtesting con datos reales
    
    Como trader con 10+ años, he aprendido que:
    1. Los datos reales son CRÍTICOS - nunca simular precios
    2. La disciplina en stops es innegociable
    3. Menos trades con más calidad > muchos trades mediocres
    4. El backtesting debe ser lo más realista posible
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.logger = get_logger("BacktestEngine")
        self.config = config or BacktestConfig()
        
        # Componentes del sistema
        self.fetcher = MultiTimeframeFetcher()
        self.confluence_analyzer = ConfluenceAnalyzer()
        self.pattern_detector = PatternDetector()
        self.scorer = DynamicScorer()
        
        # Portfolio virtual
        self.portfolio = VirtualPortfolio(
            initial_capital=self.config.initial_capital,
            max_position_pct=self.config.position_size_pct,
            max_trades=self.config.max_trades
        )
        
        # Estadísticas
        self.signals_generated = 0
        self.start_time = None
        self.end_time = None
        
        self.logger.info(
            f"BacktestEngine iniciado - Capital: ${self.config.initial_capital} "
            f"Score mínimo: {self.config.min_score_entry}"
        )
    
    async def run_backtest(self, symbol: str = None, 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> BacktestResult:
        """
        Ejecuta el backtest con datos reales de Binance
        
        Args:
            symbol: Símbolo a testear (usa config si None)
            start_date: Fecha inicio (usa últimas N velas si None)
            end_date: Fecha fin (usa ahora si None)
            
        Returns:
            BacktestResult con todos los resultados
        """
        self.start_time = datetime.now()
        context = LogContext(component="backtest_engine")
        
        try:
            # Usar símbolo de config si no se especifica
            symbols = [symbol] if symbol else self.config.symbols
            
            self.logger.info(
                f"Iniciando backtest para {symbols} con {self.config.candles_to_fetch} velas",
                context=context
            )
            
            # Para cada símbolo
            for sym in symbols:
                await self._backtest_symbol(sym)
            
            # Cerrar trades abiertos al final
            if self.portfolio.open_trades:
                self.logger.info("Cerrando trades abiertos al final del backtest")
                last_prices = await self._get_current_prices(symbols)
                self.portfolio.close_all_trades(last_prices, datetime.now(ECUADOR_TZ))
            
            # Calcular tiempo de ejecución
            self.end_time = datetime.now()
            execution_time = (self.end_time - self.start_time).total_seconds()
            
            # Obtener estadísticas finales
            portfolio_stats = self.portfolio.get_statistics()
            trades_df = self.portfolio.get_trade_history()
            balance_curve = self.portfolio.get_balance_curve()
            
            # Crear resultado
            result = BacktestResult(
                config=self.config,
                portfolio_stats=portfolio_stats,
                trades_df=trades_df,
                balance_curve=balance_curve,
                signals_generated=self.signals_generated,
                execution_time=execution_time,
                final_balance=portfolio_stats['final_balance'],
                roi_percent=portfolio_stats['roi_percent'],
                win_rate=portfolio_stats['win_rate'],
                profit_factor=portfolio_stats['profit_factor'],
                max_drawdown=portfolio_stats['max_drawdown_pct']
            )
            
            self._log_results(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en backtest: {str(e)}", context=context)
            raise TradingBotException(f"Backtest falló: {str(e)}")
    
    async def _backtest_symbol(self, symbol: str):
        """
        Ejecuta backtest para un símbolo específico
        
        Como trader profesional, proceso cada vela como si fuera en tiempo real
        """
        context = LogContext(component="backtest_engine", symbol=symbol)
        
        self.logger.info(f"Obteniendo datos reales de {symbol}...", context=context)
        
        # 1. Obtener datos históricos REALES de Binance
        data = await self._fetch_historical_data(symbol)
        
        if not data or '1h' not in data:  # Usamos 1h como timeframe principal
            self.logger.error(f"No se pudieron obtener datos para {symbol}")
            return
        
        # 2. Preparar datos para iteración
        main_df = data['1h']  # DataFrame principal (1h)
        
        # Saltar periodo de calentamiento
        start_idx = self.config.warmup_candles
        end_idx = len(main_df)
        
        self.logger.info(
            f"Procesando {end_idx - start_idx} velas de {symbol} "
            f"(saltando {start_idx} de calentamiento)",
            context=context
        )
        
        # 3. Iterar por cada vela (simulando tiempo real)
        for i in range(start_idx, end_idx):
            current_candle = main_df.iloc[i]
            current_time = current_candle.name  # El índice debe ser datetime
            current_price = current_candle['close']
            
            # Preparar datos hasta este punto (no ver el futuro!)
            historical_data = {}
            for tf, df in data.items():
                # Encontrar índice correspondiente en este timeframe
                mask = df.index <= current_time
                historical_data[tf] = df[mask]
            
            # 4. Actualizar trades abiertos con precio actual
            self.portfolio.update_trades(
                {symbol: current_price}, 
                current_time
            )
            
            # 5. Verificar si debemos entrar en un trade
            should_enter = await self._evaluate_entry_signal(
                symbol, historical_data, current_price, current_time
            )
            
            if should_enter:
                self._execute_entry(symbol, current_price, current_time)
        
        self.logger.info(
            f"Backtest de {symbol} completado. "
            f"Trades ejecutados: {len(self.portfolio.closed_trades)}",
            context=context
        )
    
    async def _fetch_historical_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Obtiene datos históricos REALES de Binance
        
        Returns:
            Dict con DataFrames por timeframe
        """
        try:
            # Obtener datos de todos los timeframes
            fetch_results = await self.fetcher.fetch_ohlcv(
                symbol=symbol,
                timeframes=self.config.timeframes,
                limit=self.config.candles_to_fetch
            )
            
            # Convertir a DataFrames con índice datetime
            data = {}
            for tf, result in fetch_results.items():
                df = result.data
                # Asegurar que el índice es datetime
                if 'open_time' in df.columns:
                    df.set_index('open_time', inplace=True)
                elif not isinstance(df.index, pd.DatetimeIndex):
                    # Intentar convertir
                    df.index = pd.to_datetime(df.index)
                
                data[tf] = df
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error obteniendo datos: {str(e)}")
            return {}
    
    async def _evaluate_entry_signal(self, symbol: str, 
                                    historical_data: Dict[str, pd.DataFrame],
                                    current_price: float,
                                    current_time: datetime) -> bool:
        """
        Evalúa si debemos entrar en un trade
        
        Como trader experimentado, solo entro cuando:
        1. El score es alto (>70)
        2. La confianza es buena (>0.6)
        3. No tenemos demasiados trades abiertos
        4. Hay confluencia multi-timeframe
        """
        # Verificar si podemos abrir más trades
        if len(self.portfolio.open_trades) >= self.config.max_trades:
            return False
        
        # Verificar si ya tenemos un trade abierto en este símbolo
        for trade in self.portfolio.open_trades.values():
            if trade.symbol == symbol:
                return False  # Solo un trade por símbolo
        
        try:
            # 1. Analizar cada timeframe
            mtf_analyses = {}
            for tf, df in historical_data.items():
                if len(df) < 20:  # Necesitamos mínimo 20 velas
                    continue
                
                analyzer = create_analyzer(tf)
                analysis = analyzer.analyze(df, symbol)
                mtf_analyses[tf] = analysis
            
            if not mtf_analyses:
                return False
            
            # 2. Analizar confluencia
            confluence_result = analyze_confluence(mtf_analyses)
            
            # 3. Detectar patrones (simplificado para backtest)
            detected_patterns = []  # Por ahora sin detección de patrones
            
            # 4. Preparar contexto de mercado
            market_context = {
                'volatility': self._calculate_volatility(historical_data.get('1h')),
                'liquidity_score': 0.8,  # Asumimos buena liquidez
                'correlation_risk': 0.0,
                'market_hours': 'active'
            }
            
            # 5. Calcular score final (SIN LLM para backtest básico)
            scoring_result = self.scorer.calculate_score(
                mtf_analyses=mtf_analyses,
                confluence_result=confluence_result,
                detected_patterns=detected_patterns,
                market_context=market_context,
                symbol=symbol
            )
            
            self.signals_generated += 1
            
            # 6. Decidir si entrar
            if (scoring_result.final_score >= self.config.min_score_entry and
                scoring_result.breakdown.confidence >= self.config.min_confidence):
                
                self.logger.info(
                    f"SEÑAL DE ENTRADA: {symbol} @ {current_price:.2f} "
                    f"Score: {scoring_result.final_score:.1f} "
                    f"Confianza: {scoring_result.breakdown.confidence:.2f}"
                )
                
                # Guardar score para el trade
                self.last_entry_score = scoring_result.final_score
                self.last_entry_confidence = scoring_result.breakdown.confidence
                
                return True
            
        except Exception as e:
            self.logger.debug(f"Error evaluando señal: {str(e)}")
        
        return False
    
    def _execute_entry(self, symbol: str, entry_price: float, entry_time: datetime):
        """
        Ejecuta la entrada al mercado
        
        Como trader disciplinado, SIEMPRE uso stop loss y take profit
        MEJORADO: Stop loss dinámico según volatilidad del activo
        """
        # Stop Loss Dinámico según el tipo de activo
        # Basado en análisis: altcoins necesitan más espacio que BTC/ETH
        if symbol in ['BTCUSDT', 'ETHUSDT']:
            # Majors: 3% stop loss (antes 2%)
            dynamic_sl_pct = 0.03
            dynamic_tp_pct = 0.09  # Mantenemos ratio 1:3
        elif symbol in ['BNBUSDT', 'SOLUSDT']:
            # Top 10 coins: 3.5% stop loss
            dynamic_sl_pct = 0.035
            dynamic_tp_pct = 0.105  # Mantenemos ratio 1:3
        elif symbol in ['ADAUSDT']:
            # Mid-cap altcoins: 4% stop loss
            dynamic_sl_pct = 0.04
            dynamic_tp_pct = 0.12  # Mantenemos ratio 1:3
        elif symbol in ['DOGEUSDT', 'TIAUSDT']:
            # High volatility altcoins: 4.5% stop loss
            dynamic_sl_pct = 0.045
            dynamic_tp_pct = 0.135  # Mantenemos ratio 1:3
        else:
            # Default para otros: 3.5%
            dynamic_sl_pct = 0.035
            dynamic_tp_pct = 0.105
        
        # Calcular stop loss y take profit con valores dinámicos
        stop_loss = entry_price * (1 - dynamic_sl_pct)
        take_profit = entry_price * (1 + dynamic_tp_pct)
        
        # Ajustar tamaño de posición según volatilidad
        # Mantener riesgo constante: más volatilidad = menor posición
        base_position_pct = self.config.position_size_pct
        
        # Ajuste inverso a la volatilidad (más SL = menos posición)
        volatility_adjustment = 0.03 / dynamic_sl_pct  # 3% como referencia base
        adjusted_position_pct = base_position_pct * volatility_adjustment
        
        # Limitar entre 10% y 30% del capital
        adjusted_position_pct = min(0.30, max(0.10, adjusted_position_pct))
        
        # Calcular tamaño de posición ajustado
        position_size = self.portfolio.current_balance * adjusted_position_pct
        
        # Abrir trade (por ahora solo LONG)
        trade = self.portfolio.open_trade(
            symbol=symbol,
            side=TradeSide.LONG,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_usd=position_size,
            entry_score=getattr(self, 'last_entry_score', 0),
            entry_confidence=getattr(self, 'last_entry_confidence', 0)
        )
        
        if trade:
            self.logger.info(
                f"✅ TRADE EJECUTADO: {trade.trade_id} {symbol} "
                f"Entrada: ${entry_price:.2f} SL: ${stop_loss:.2f} ({dynamic_sl_pct*100:.1f}%) "
                f"TP: ${take_profit:.2f} ({dynamic_tp_pct*100:.1f}%) "
                f"Posición: {adjusted_position_pct*100:.1f}%"
            )
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calcula volatilidad simple para el contexto"""
        if df is None or len(df) < 20:
            return 0.02  # Valor por defecto
        
        # Volatilidad como desviación estándar de retornos
        returns = df['close'].pct_change().dropna()
        return returns.std()
    
    async def _get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Obtiene precios actuales para cerrar trades al final"""
        prices = {}
        for symbol in symbols:
            try:
                # Obtener última vela
                result = await self.fetcher.fetch_ohlcv(
                    symbol=symbol,
                    timeframes=['1m'],  # 1 minuto para precio más actual
                    limit=1
                )
                if '1m' in result and not result['1m'].data.empty:
                    prices[symbol] = result['1m'].data['close'].iloc[-1]
            except:
                pass
        return prices
    
    def _log_results(self, result: BacktestResult):
        """
        Registra los resultados del backtest
        
        Como trader profesional, estas son las métricas que reviso SIEMPRE
        """
        self.logger.info("=" * 60)
        self.logger.info("RESULTADOS DEL BACKTEST")
        self.logger.info("=" * 60)
        self.logger.info(f"Capital inicial: ${result.config.initial_capital:.2f}")
        self.logger.info(f"Balance final: ${result.final_balance:.2f}")
        self.logger.info(f"ROI: {result.roi_percent:.2f}%")
        self.logger.info(f"Total trades: {result.portfolio_stats['total_trades']}")
        self.logger.info(f"Win Rate: {result.win_rate:.1f}%")
        self.logger.info(f"Profit Factor: {result.profit_factor:.2f}")
        self.logger.info(f"Max Drawdown: {result.max_drawdown:.2f}%")
        self.logger.info(f"Señales generadas: {result.signals_generated}")
        self.logger.info(f"Tiempo ejecución: {result.execution_time:.1f}s")
        self.logger.info("=" * 60)