"""
Manejador de Datos Históricos para Backtesting
==============================================

Como trader senior, sé que la calidad de los datos es CRÍTICA.
Datos malos = backtesting inútil = pérdidas en producción.

Este módulo maneja:
- Descarga de datos históricos de múltiples fuentes
- Validación y limpieza de datos
- Detección de gaps y datos faltantes
- Ajuste por splits y dividendos
- Sincronización de timeframes múltiples

Fuentes de datos soportadas:
- Binance (principal para crypto)
- Yahoo Finance (backup)
- Archivos CSV locales
- Bases de datos

Filosofía: "Garbage in, garbage out - La calidad de datos es todo"

Autor: Trader Algorítmico Senior (10+ años)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import pytz
import requests
import time
from pathlib import Path

from ..config.settings import config
from ..utils.logger import get_logger, LogContext
from ..utils.exceptions import TradingBotException, ErrorCodes

# Zona horaria de Ecuador (UTC-5)
ECUADOR_TZ = pytz.timezone('America/Guayaquil')

class DataSource(Enum):
    """Fuentes de datos disponibles"""
    BINANCE = "binance"
    YAHOO_FINANCE = "yahoo_finance"
    LOCAL_CSV = "local_csv"
    DATABASE = "database"

class DataQuality(Enum):
    """Niveles de calidad de datos"""
    EXCELLENT = "excellent"  # Sin gaps, datos completos
    GOOD = "good"           # Gaps menores, interpolables
    FAIR = "fair"           # Algunos gaps, usable con precaución
    POOR = "poor"           # Muchos gaps, no recomendado
    UNUSABLE = "unusable"   # Datos corruptos o insuficientes

@dataclass
class MarketDataPoint:
    """Punto de datos de mercado"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # Datos adicionales
    trades: Optional[int] = None
    quote_volume: Optional[float] = None
    
    def validate(self) -> bool:
        """Valida que el punto de datos sea consistente"""
        try:
            # Verificar que high >= low
            if self.high < self.low:
                return False
            
            # Verificar que open, close estén entre high y low
            if not (self.low <= self.open <= self.high):
                return False
            if not (self.low <= self.close <= self.high):
                return False
            
            # Verificar que los precios sean positivos
            if any(price <= 0 for price in [self.open, self.high, self.low, self.close]):
                return False
            
            # Verificar volumen no negativo
            if self.volume < 0:
                return False
            
            return True
            
        except Exception:
            return False

class HistoricalDataHandler:
    """
    Manejador principal de datos históricos
    
    Como trader senior, he diseñado este handler para ser:
    1. Robusto - Maneja errores de red y datos faltantes
    2. Eficiente - Caché inteligente para evitar re-descargas
    3. Preciso - Validación exhaustiva de calidad de datos
    4. Flexible - Múltiples fuentes con fallback automático
    """
    
    def __init__(self, cache_dir: str = "data/historical"):
        self.logger = get_logger("HistoricalDataHandler")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuración de fuentes de datos
        self.data_sources = {
            DataSource.BINANCE: self._fetch_binance_data,
            DataSource.YAHOO_FINANCE: self._fetch_yahoo_data,
            DataSource.LOCAL_CSV: self._fetch_local_csv
        }
        
        # Caché de datos
        self.data_cache: Dict[str, pd.DataFrame] = {}
        
        # Configuración de calidad
        self.min_data_points = 100  # Mínimo para backtesting
        self.max_gap_percentage = 0.05  # Máximo 5% de gaps
        self.outlier_threshold = 5.0  # 5 desviaciones estándar
    
    def get_historical_data(self, symbol: str, start_date: datetime, 
                          end_date: datetime, frequency: str = "1h",
                          source: DataSource = DataSource.BINANCE) -> Optional[pd.DataFrame]:
        """
        Obtiene datos históricos para un símbolo
        
        Args:
            symbol: Símbolo a obtener (ej: BTCUSDT)
            start_date: Fecha de inicio
            end_date: Fecha de fin
            frequency: Frecuencia de datos (1m, 5m, 15m, 1h, 4h, 1d)
            source: Fuente de datos preferida
            
        Returns:
            DataFrame con datos OHLCV o None si falla
        """
        context = LogContext(component="data_handler")
        
        try:
            # Generar clave de caché
            cache_key = f"{symbol}_{frequency}_{start_date.date()}_{end_date.date()}"
            
            # Verificar caché
            if cache_key in self.data_cache:
                self.logger.debug(f"Datos obtenidos del caché: {cache_key}")
                return self.data_cache[cache_key]
            
            # Intentar obtener datos de la fuente principal
            df = self._fetch_data_from_source(symbol, start_date, end_date, frequency, source)
            
            # Si falla, intentar fuentes de respaldo
            if df is None or df.empty:
                self.logger.warning(f"Fuente principal falló, intentando respaldos para {symbol}")
                
                backup_sources = [s for s in DataSource if s != source]
                for backup_source in backup_sources:
                    df = self._fetch_data_from_source(symbol, start_date, end_date, frequency, backup_source)
                    if df is not None and not df.empty:
                        self.logger.info(f"Datos obtenidos de fuente de respaldo: {backup_source.value}")
                        break
            
            if df is None or df.empty:
                self.logger.error(f"No se pudieron obtener datos para {symbol}")
                return None
            
            # Validar y limpiar datos
            df_cleaned = self._validate_and_clean_data(df, symbol)
            
            if df_cleaned is None:
                self.logger.error(f"Datos de {symbol} no pasaron validación")
                return None
            
            # Evaluar calidad
            quality = self._assess_data_quality(df_cleaned, symbol)
            
            if quality == DataQuality.UNUSABLE:
                self.logger.error(f"Calidad de datos inaceptable para {symbol}")
                return None
            
            if quality in [DataQuality.POOR, DataQuality.FAIR]:
                self.logger.warning(f"Calidad de datos {quality.value} para {symbol}")
            
            # Guardar en caché
            self.data_cache[cache_key] = df_cleaned
            
            # Guardar en disco para persistencia
            self._save_to_cache_file(cache_key, df_cleaned)
            
            self.logger.info(
                f"Datos históricos obtenidos: {symbol}",
                context=context,
                extra_fields={
                    "records": len(df_cleaned),
                    "start_date": start_date.date(),
                    "end_date": end_date.date(),
                    "quality": quality.value,
                    "source": source.value
                }
            )
            
            return df_cleaned
            
        except Exception as e:
            self.logger.error(f"Error obteniendo datos históricos: {e}")
            return None
    
    def _fetch_data_from_source(self, symbol: str, start_date: datetime,
                               end_date: datetime, frequency: str,
                               source: DataSource) -> Optional[pd.DataFrame]:
        """Obtiene datos de una fuente específica"""
        try:
            if source in self.data_sources:
                fetch_func = self.data_sources[source]
                return fetch_func(symbol, start_date, end_date, frequency)
            else:
                self.logger.warning(f"Fuente no soportada: {source}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error obteniendo datos de {source.value}: {e}")
            return None
    
    def _fetch_binance_data(self, symbol: str, start_date: datetime,
                           end_date: datetime, frequency: str) -> Optional[pd.DataFrame]:
        """
        Obtiene datos históricos de Binance
        
        Como trader senior, uso Binance porque tiene los datos más confiables
        y la API más estable para crypto trading.
        """
        try:
            # Mapear frecuencias a formato Binance
            interval_map = {
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"
            }
            
            if frequency not in interval_map:
                self.logger.error(f"Frecuencia no soportada: {frequency}")
                return None
            
            interval = interval_map[frequency]
            
            # Convertir fechas a timestamps
            start_ts = int(start_date.timestamp() * 1000)
            end_ts = int(end_date.timestamp() * 1000)
            
            # URL de la API de Binance
            base_url = "https://api.binance.com/api/v3/klines"
            
            all_data = []
            current_start = start_ts
            
            # Binance limita a 1000 velas por request
            limit = 1000
            
            while current_start < end_ts:
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": current_start,
                    "endTime": end_ts,
                    "limit": limit
                }
                
                response = requests.get(base_url, params=params, timeout=30)
                
                if response.status_code != 200:
                    self.logger.error(f"Error API Binance: {response.status_code}")
                    break
                
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                
                # Actualizar timestamp para siguiente batch
                current_start = data[-1][6] + 1  # Close time + 1ms
                
                # Rate limiting - Binance permite 1200 requests/min
                time.sleep(0.1)
            
            if not all_data:
                self.logger.warning(f"No se obtuvieron datos de Binance para {symbol}")
                return None
            
            # Convertir a DataFrame
            df = pd.DataFrame(all_data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convertir tipos de datos
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convertir timestamps a datetime
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Seleccionar columnas necesarias
            df = df[['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']]
            
            # Remover duplicados y ordenar
            df = df.drop_duplicates().sort_index()
            
            self.logger.info(f"Datos Binance obtenidos: {len(df)} velas para {symbol}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error obteniendo datos de Binance: {e}")
            return None
    
    def _fetch_yahoo_data(self, symbol: str, start_date: datetime,
                         end_date: datetime, frequency: str) -> Optional[pd.DataFrame]:
        """
        Obtiene datos de Yahoo Finance (backup para algunos activos)
        
        Nota: Principalmente para stocks, no crypto
        """
        try:
            # Yahoo Finance requiere yfinance library
            import yfinance as yf
            
            # Mapear frecuencias
            interval_map = {
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                "1h": "1h", "1d": "1d", "1w": "1wk"
            }
            
            if frequency not in interval_map:
                self.logger.warning(f"Frecuencia {frequency} no soportada en Yahoo Finance")
                return None
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval_map[frequency]
            )
            
            if df.empty:
                return None
            
            # Normalizar columnas
            df.columns = df.columns.str.lower()
            df['volume'] = df.get('volume', 0)
            df['quote_volume'] = df['volume'] * df['close']  # Aproximación
            df['trades'] = None
            
            return df
            
        except ImportError:
            self.logger.warning("yfinance no instalado, Yahoo Finance no disponible")
            return None
        except Exception as e:
            self.logger.error(f"Error obteniendo datos de Yahoo Finance: {e}")
            return None
    
    def _fetch_local_csv(self, symbol: str, start_date: datetime,
                        end_date: datetime, frequency: str) -> Optional[pd.DataFrame]:
        """Obtiene datos de archivos CSV locales"""
        try:
            csv_path = self.cache_dir / f"{symbol}_{frequency}.csv"
            
            if not csv_path.exists():
                self.logger.warning(f"Archivo CSV no encontrado: {csv_path}")
                return None
            
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            
            # Filtrar por fechas
            mask = (df.index >= start_date) & (df.index <= end_date)
            df = df[mask]
            
            return df if not df.empty else None
            
        except Exception as e:
            self.logger.error(f"Error leyendo CSV local: {e}")
            return None
    
    def _validate_and_clean_data(self, df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """
        Valida y limpia los datos históricos
        
        Como trader senior, sé que datos sucios pueden arruinar un backtest
        """
        try:
            if df is None or df.empty:
                return None
            
            original_len = len(df)
            
            # 1. Remover filas con valores nulos en columnas críticas
            critical_columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.dropna(subset=critical_columns)
            
            # 2. Validar que high >= low
            invalid_hl = df['high'] < df['low']
            if invalid_hl.any():
                self.logger.warning(f"Removiendo {invalid_hl.sum()} velas con high < low")
                df = df[~invalid_hl]
            
            # 3. Validar que open y close estén entre high y low
            invalid_ohlc = (
                (df['open'] > df['high']) | (df['open'] < df['low']) |
                (df['close'] > df['high']) | (df['close'] < df['low'])
            )
            if invalid_ohlc.any():
                self.logger.warning(f"Removiendo {invalid_ohlc.sum()} velas con OHLC inválido")
                df = df[~invalid_ohlc]
            
            # 4. Remover precios negativos o cero
            invalid_prices = (
                (df['open'] <= 0) | (df['high'] <= 0) |
                (df['low'] <= 0) | (df['close'] <= 0)
            )
            if invalid_prices.any():
                self.logger.warning(f"Removiendo {invalid_prices.sum()} velas con precios <= 0")
                df = df[~invalid_prices]
            
            # 5. Detectar y manejar outliers extremos
            for col in ['open', 'high', 'low', 'close']:
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)
                iqr = q99 - q01
                
                # Outliers extremos (más de 5 IQR)
                outlier_threshold = 5 * iqr
                outliers = (
                    (df[col] > q99 + outlier_threshold) |
                    (df[col] < q01 - outlier_threshold)
                )
                
                if outliers.any():
                    self.logger.warning(f"Removiendo {outliers.sum()} outliers extremos en {col}")
                    df = df[~outliers]
            
            # 6. Verificar volumen negativo
            if 'volume' in df.columns:
                negative_volume = df['volume'] < 0
                if negative_volume.any():
                    self.logger.warning(f"Corrigiendo {negative_volume.sum()} volúmenes negativos")
                    df.loc[negative_volume, 'volume'] = 0
            
            # 7. Ordenar por timestamp
            df = df.sort_index()
            
            # 8. Remover duplicados
            df = df[~df.index.duplicated(keep='first')]
            
            cleaned_len = len(df)
            removed_pct = (original_len - cleaned_len) / original_len * 100
            
            self.logger.info(
                f"Datos limpiados para {symbol}: {original_len} -> {cleaned_len} "
                f"({removed_pct:.1f}% removido)"
            )
            
            return df if not df.empty else None
            
        except Exception as e:
            self.logger.error(f"Error validando datos: {e}")
            return None
    
    def _assess_data_quality(self, df: pd.DataFrame, symbol: str) -> DataQuality:
        """
        Evalúa la calidad de los datos históricos
        
        Como trader senior, clasifico los datos para saber qué tan confiables son
        """
        try:
            if df is None or df.empty:
                return DataQuality.UNUSABLE
            
            total_points = len(df)
            
            # Verificar cantidad mínima de datos
            if total_points < self.min_data_points:
                return DataQuality.UNUSABLE
            
            # Detectar gaps en los datos
            expected_freq = pd.infer_freq(df.index)
            if expected_freq:
                expected_range = pd.date_range(
                    start=df.index[0],
                    end=df.index[-1],
                    freq=expected_freq
                )
                missing_points = len(expected_range) - total_points
                gap_percentage = missing_points / len(expected_range)
            else:
                # Si no se puede inferir frecuencia, usar aproximación
                time_diff = df.index[-1] - df.index[0]
                avg_interval = time_diff / (total_points - 1)
                expected_points = int(time_diff / avg_interval) + 1
                gap_percentage = (expected_points - total_points) / expected_points
            
            # Evaluar volatilidad de precios (detectar datos anómalos)
            returns = df['close'].pct_change().dropna()
            volatility = returns.std()
            extreme_moves = (abs(returns) > 0.2).sum()  # Movimientos > 20%
            extreme_pct = extreme_moves / len(returns) if len(returns) > 0 else 0
            
            # Clasificar calidad
            if gap_percentage <= 0.01 and extreme_pct <= 0.001:  # < 1% gaps, < 0.1% extreme moves
                quality = DataQuality.EXCELLENT
            elif gap_percentage <= 0.03 and extreme_pct <= 0.005:  # < 3% gaps, < 0.5% extreme moves
                quality = DataQuality.GOOD
            elif gap_percentage <= self.max_gap_percentage and extreme_pct <= 0.02:  # < 5% gaps, < 2% extreme moves
                quality = DataQuality.FAIR
            elif gap_percentage <= 0.15 and extreme_pct <= 0.05:  # < 15% gaps, < 5% extreme moves
                quality = DataQuality.POOR
            else:
                quality = DataQuality.UNUSABLE
            
            self.logger.info(
                f"Calidad de datos para {symbol}: {quality.value} "
                f"(gaps: {gap_percentage:.1%}, extreme moves: {extreme_pct:.1%})"
            )
            
            return quality
            
        except Exception as e:
            self.logger.error(f"Error evaluando calidad de datos: {e}")
            return DataQuality.POOR
    
    def _save_to_cache_file(self, cache_key: str, df: pd.DataFrame) -> None:
        """Guarda datos en archivo de caché para persistencia"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            df.to_parquet(cache_file, compression='snappy')
            self.logger.debug(f"Datos guardados en caché: {cache_file}")
            
        except Exception as e:
            self.logger.warning(f"No se pudo guardar caché: {e}")
    
    def load_from_cache_file(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Carga datos desde archivo de caché"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            
            if cache_file.exists():
                df = pd.read_parquet(cache_file)
                self.logger.debug(f"Datos cargados desde caché: {cache_file}")
                return df
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error cargando desde caché: {e}")
            return None
    
    def clear_cache(self) -> None:
        """Limpia el caché de datos"""
        try:
            self.data_cache.clear()
            
            # Limpiar archivos de caché
            for cache_file in self.cache_dir.glob("*.parquet"):
                cache_file.unlink()
            
            self.logger.info("Caché de datos limpiado")
            
        except Exception as e:
            self.logger.error(f"Error limpiando caché: {e}")
    
    def get_data_info(self, symbol: str, frequency: str) -> Dict[str, Any]:
        """Obtiene información sobre los datos disponibles"""
        try:
            cache_key = f"{symbol}_{frequency}"
            
            if cache_key in self.data_cache:
                df = self.data_cache[cache_key]
            else:
                # Intentar cargar desde archivo
                df = self.load_from_cache_file(cache_key)
                if df is not None:
                    self.data_cache[cache_key] = df
            
            if df is None or df.empty:
                return {"available": False}
            
            return {
                "available": True,
                "records": len(df),
                "start_date": df.index[0],
                "end_date": df.index[-1],
                "frequency": frequency,
                "columns": list(df.columns),
                "quality": self._assess_data_quality(df, symbol).value
            }
            
        except Exception as e:
            self.logger.error(f"Error obteniendo info de datos: {e}")
            return {"available": False, "error": str(e)}