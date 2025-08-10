"""
Multi-timeframe data fetcher with async support
Handles OHLCV data retrieval with intelligent retry logic and rate limiting
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd

from ..config.settings import config
from ..utils.logger import get_logger, LogContext
from ..utils.exceptions import (
    DataFetchException, 
    DataValidationException,
    ErrorCodes,
    create_exception
)
from ..config.symbols import get_active_symbols, get_symbol_config

# Import existing Binance client
from bnb.binance import RobotBinance

@dataclass
class FetchRequest:
    """Data fetch request configuration"""
    symbol: str
    timeframe: str
    limit: int = 500
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    priority: int = 1  # 1=high, 2=medium, 3=low

@dataclass
class FetchResult:
    """Data fetch result with metadata"""
    symbol: str
    timeframe: str
    data: pd.DataFrame
    fetch_time: datetime
    data_quality: float
    source: str = "binance"
    cached: bool = False

class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, max_requests: int = 1200, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.logger = get_logger("RateLimiter")
    
    async def acquire(self) -> None:
        """Acquire permission to make a request"""
        now = time.time()
        
        # Remove old requests outside the time window
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.time_window]
        
        # Check if we're at the limit
        if len(self.requests) >= self.max_requests:
            sleep_time = self.time_window - (now - self.requests[0]) + 1
            self.logger.warning(f"Rate limit reached, sleeping for {sleep_time:.1f}s")
            await asyncio.sleep(sleep_time)
            return await self.acquire()
        
        # Record this request
        self.requests.append(now)

class MultiTimeframeFetcher:
    """
    Advanced multi-timeframe data fetcher with async support
    Features: rate limiting, retry logic, request queuing, data validation
    """
    
    def __init__(self):
        self.logger = get_logger("MultiTimeframeFetcher")
        self.rate_limiter = RateLimiter()
        self.clients: Dict[str, RobotBinance] = {}
        self.request_queue = asyncio.Queue()
        self.active_requests = 0
        self.max_concurrent_requests = 5
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "retry_count": 0
        }
    
    def _get_client(self, symbol: str, timeframe: str) -> RobotBinance:
        """Get or create a Binance client for the symbol/timeframe"""
        key = f"{symbol}_{timeframe}"
        if key not in self.clients:
            self.clients[key] = RobotBinance(pair=symbol, temporality=timeframe)
        return self.clients[key]
    
    async def _fetch_single_with_retry(self, request: FetchRequest, 
                                     max_retries: int = 3) -> FetchResult:
        """Fetch data for a single symbol/timeframe with retry logic"""
        context = LogContext(
            component="data_fetcher",
            symbol=request.symbol,
            timeframe=request.timeframe
        )
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # Rate limiting
                await self.rate_limiter.acquire()
                
                # Get client and fetch data
                client = self._get_client(request.symbol, request.timeframe)
                
                self.logger.debug(
                    f"Fetching {request.symbol} {request.timeframe} (attempt {attempt + 1})",
                    context=context
                )
                
                # Fetch OHLCV data
                start_time = time.time()
                
                # Convert datetime to string if provided
                start_str = request.start_time.isoformat() if request.start_time else None
                end_str = request.end_time.isoformat() if request.end_time else None
                
                df = client.candlestick(
                    start_str=start_str,
                    end_str=end_str,
                    limit=request.limit
                )
                
                fetch_duration = time.time() - start_time
                
                if df.empty:
                    raise DataFetchException(
                        f"No data returned for {request.symbol} {request.timeframe}",
                        ErrorCodes.DATA_MISSING,
                        {"symbol": request.symbol, "timeframe": request.timeframe}
                    )
                
                # Calculate data quality score
                data_quality = self._calculate_data_quality(df)
                
                # Create result
                result = FetchResult(
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    data=df,
                    fetch_time=datetime.utcnow(),
                    data_quality=data_quality,
                    source="binance"
                )
                
                self.stats["successful_requests"] += 1
                
                self.logger.debug(
                    f"Successfully fetched {len(df)} candles in {fetch_duration:.2f}s "
                    f"(quality: {data_quality:.2f})",
                    context=context,
                    extra_fields={
                        "candles_count": len(df),
                        "fetch_duration": fetch_duration,
                        "data_quality": data_quality
                    }
                )
                
                return result
                
            except Exception as e:
                last_exception = e
                self.stats["retry_count"] += 1
                
                if attempt < max_retries:
                    delay = (2 ** attempt) + (attempt * 0.1)  # Exponential backoff with jitter
                    self.logger.warning(
                        f"Fetch attempt {attempt + 1} failed, retrying in {delay:.1f}s: {str(e)}",
                        context=context
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(
                        f"All fetch attempts failed for {request.symbol} {request.timeframe}",
                        context=context
                    )
        
        # All retries failed
        self.stats["failed_requests"] += 1
        raise DataFetchException(
            f"Failed to fetch data after {max_retries + 1} attempts: {str(last_exception)}",
            ErrorCodes.DATA_FETCH_FAILED,
            {
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "attempts": max_retries + 1,
                "last_error": str(last_exception)
            }
        )
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> float:
        """Calculate data quality score (0.0 to 1.0)"""
        if df.empty:
            return 0.0
        
        quality_score = 1.0
        
        # Check for missing values
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        quality_score -= missing_ratio * 0.5
        
        # Check for zero volumes (suspicious)
        zero_volume_ratio = (df['volume'] == 0).sum() / len(df)
        quality_score -= zero_volume_ratio * 0.3
        
        # Check for price anomalies (OHLC consistency)
        price_anomalies = 0
        for _, row in df.iterrows():
            if not (row['low'] <= row['open'] <= row['high'] and 
                   row['low'] <= row['close'] <= row['high']):
                price_anomalies += 1
        
        anomaly_ratio = price_anomalies / len(df)
        quality_score -= anomaly_ratio * 0.4
        
        # Check for gaps in time series (if index is datetime)
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
            expected_intervals = len(df) - 1
            actual_intervals = len(df) - 1  # Simplified - would need timeframe-specific logic
            gap_ratio = abs(expected_intervals - actual_intervals) / expected_intervals
            quality_score -= gap_ratio * 0.2
        
        return max(0.0, min(1.0, quality_score))
    
    async def fetch_ohlcv(self, symbol: str, timeframes: List[str], 
                         limit: int = 500,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> Dict[str, FetchResult]:
        """
        Fetch OHLCV data for multiple timeframes of a single symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframes: List of timeframes (e.g., ['1d', '4h', '1h', '15m'])
            limit: Number of candles to fetch
            start_time: Start time for historical data
            end_time: End time for historical data
            
        Returns:
            Dictionary mapping timeframe to FetchResult
        """
        context = LogContext(component="data_fetcher", symbol=symbol)
        
        self.logger.info(
            f"Fetching {symbol} data for timeframes: {', '.join(timeframes)}",
            context=context
        )
        
        # Create fetch requests
        requests = [
            FetchRequest(
                symbol=symbol,
                timeframe=tf,
                limit=limit,
                start_time=start_time,
                end_time=end_time
            )
            for tf in timeframes
        ]
        
        # Execute requests concurrently
        tasks = [self._fetch_single_with_retry(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        fetch_results = {}
        successful_fetches = 0
        
        for i, result in enumerate(results):
            timeframe = timeframes[i]
            
            if isinstance(result, Exception):
                self.logger.error(
                    f"Failed to fetch {symbol} {timeframe}: {str(result)}",
                    context=context
                )
                # Could store failed result or skip
                continue
            
            fetch_results[timeframe] = result
            successful_fetches += 1
        
        self.stats["total_requests"] += len(timeframes)
        
        self.logger.info(
            f"Completed {symbol} fetch: {successful_fetches}/{len(timeframes)} successful",
            context=context,
            extra_fields={
                "successful_fetches": successful_fetches,
                "total_timeframes": len(timeframes),
                "success_rate": successful_fetches / len(timeframes)
            }
        )
        
        return fetch_results
    
    async def fetch_multiple_symbols(self, symbols: List[str], 
                                   timeframes: List[str],
                                   limit: int = 500) -> Dict[str, Dict[str, FetchResult]]:
        """
        Fetch OHLCV data for multiple symbols and timeframes
        
        Args:
            symbols: List of trading symbols
            timeframes: List of timeframes
            limit: Number of candles to fetch
            
        Returns:
            Nested dictionary: {symbol: {timeframe: FetchResult}}
        """
        context = LogContext(component="data_fetcher")
        
        self.logger.info(
            f"Fetching data for {len(symbols)} symbols, {len(timeframes)} timeframes",
            context=context
        )
        
        # Create tasks for all symbol/timeframe combinations
        tasks = []
        symbol_timeframe_map = []
        
        for symbol in symbols:
            task = self.fetch_ohlcv(symbol, timeframes, limit)
            tasks.append(task)
            symbol_timeframe_map.append(symbol)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Organize results
        organized_results = {}
        total_successful = 0
        total_possible = len(symbols) * len(timeframes)
        
        for i, result in enumerate(results):
            symbol = symbol_timeframe_map[i]
            
            if isinstance(result, Exception):
                self.logger.error(
                    f"Failed to fetch data for {symbol}: {str(result)}",
                    context=context
                )
                organized_results[symbol] = {}
                continue
            
            organized_results[symbol] = result
            total_successful += len(result)
        
        success_rate = total_successful / total_possible if total_possible > 0 else 0
        
        self.logger.info(
            f"Multi-symbol fetch completed: {total_successful}/{total_possible} "
            f"successful ({success_rate:.1%})",
            context=context,
            extra_fields={
                "total_successful": total_successful,
                "total_possible": total_possible,
                "success_rate": success_rate,
                "symbols_count": len(symbols),
                "timeframes_count": len(timeframes)
            }
        )
        
        return organized_results
    
    async def fetch_market_context(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch additional market context data for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with market context information
        """
        context = LogContext(component="data_fetcher", symbol=symbol)
        
        try:
            # Get current price
            client = self._get_client(symbol, "1m")  # Use 1m for current price
            current_price = client.symbol_price(symbol)
            
            # Get 24h ticker data (would need to extend RobotBinance for this)
            # For now, return basic context
            market_context = {
                "current_price": current_price,
                "timestamp": datetime.utcnow(),
                "symbol": symbol,
                "data_source": "binance"
            }
            
            self.logger.debug(
                f"Fetched market context for {symbol}",
                context=context,
                extra_fields=market_context
            )
            
            return market_context
            
        except Exception as e:
            self.logger.error(
                f"Failed to fetch market context for {symbol}: {str(e)}",
                context=context
            )
            raise DataFetchException(
                f"Failed to fetch market context: {str(e)}",
                ErrorCodes.DATA_FETCH_FAILED,
                {"symbol": symbol}
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get fetcher statistics"""
        total_requests = self.stats["total_requests"]
        success_rate = (self.stats["successful_requests"] / total_requests 
                       if total_requests > 0 else 0)
        
        return {
            **self.stats,
            "success_rate": success_rate,
            "active_clients": len(self.clients),
            "queue_size": self.request_queue.qsize(),
            "active_requests": self.active_requests
        }
    
    def validate_data_quality(self, data: pd.DataFrame, 
                            min_quality: float = 0.95) -> bool:
        """
        Validate data quality meets minimum standards
        
        Args:
            data: DataFrame to validate
            min_quality: Minimum quality score required
            
        Returns:
            True if data quality is acceptable
        """
        if data.empty:
            return False
        
        quality_score = self._calculate_data_quality(data)
        return quality_score >= min_quality

# Global fetcher instance
fetcher = MultiTimeframeFetcher()

# Convenience functions
async def fetch_symbol_data(symbol: str, timeframes: List[str] = None, 
                          limit: int = 500) -> Dict[str, FetchResult]:
    """Convenience function to fetch data for a single symbol"""
    if timeframes is None:
        timeframes = config.timeframes
    
    return await fetcher.fetch_ohlcv(symbol, timeframes, limit)

async def fetch_all_active_symbols(timeframes: List[str] = None,
                                 limit: int = 500) -> Dict[str, Dict[str, FetchResult]]:
    """Convenience function to fetch data for all active symbols"""
    if timeframes is None:
        timeframes = config.timeframes
    
    symbols = get_active_symbols()
    return await fetcher.fetch_multiple_symbols(symbols, timeframes, limit)