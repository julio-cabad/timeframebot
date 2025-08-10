import time
import pandas as pd
from config.settings import BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TESTNET
from binance.um_futures import UMFutures
from smc.logger import get_logger

class RobotBinance:
    def __init__(self, pair: str, temporality: str):
        """Initialize Binance client with API credentials."""
        self.pair = pair.upper()
        self.temporality = temporality
        self.symbol = self.pair
        self.logger = get_logger("RobotBinance")
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initialize and configure Binance client."""
        try:
            
            api_key = BINANCE_API_KEY
            api_secret = BINANCE_API_SECRET

            if not api_key:
                self.logger.error("BINANCE_API_KEY no encontrada. Asegúrate de que está definida en tu archivo .env o en las variables de entorno.")
                raise ValueError("BINANCE_API_KEY no configurada.")
            if not api_secret:
                self.logger.error("BINANCE_API_SECRET no encontrada. Asegúrate de que está definida en tu archivo .env o en las variables de entorno.")
                raise ValueError("BINANCE_API_SECRET no configurada.")

            client = UMFutures(
                key=api_key,
                secret=api_secret,
                base_url="https://fapi.binance.com"  # Usando mainnet directamente
            )
            
            # Verificar la conexión
            time_res = client.time()
            self.logger.debug(f"Connected to Binance UMFutures: {'Mainnet' if not BINANCE_TESTNET else 'Testnet'}")
            self.logger.debug(f"Server time: {time_res['serverTime']}")
            
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance client: {str(e)}")
            raise

    def binance_client(self):
        return self.client

    def _request(self, func, **kwargs):
        """Wrapper para manejar errores en las peticiones a Binance."""
        retries = 0
        max_retries = 3
        delay = 5  # seconds

        while retries < max_retries:
            try:
                return func(**kwargs)
            except Exception as e:
                self.logger.error(f"Error en petición a Binance: {str(e)}")
                retries += 1
                if retries < max_retries:
                    self.logger.info(f"Reintentando en {delay} segundos... (Intento {retries}/{max_retries})")
                    time.sleep(delay)
                else:
                    self.logger.error("Se alcanzó el máximo de intentos")
                    raise

    def binance_account(self) -> dict:
        """Obtiene información de la cuenta"""
        return self._request(self.client.account)

    def symbol_price(self, target_symbol: str = None) -> float | None:
        """
        Get current price for the specified symbol.
        If target_symbol is None, it will attempt to use self.symbol (from constructor).
        """
        symbol_to_fetch = target_symbol if target_symbol else self.symbol
        
        if not symbol_to_fetch:
            self.logger.error("No symbol provided or configured to fetch price.")
            return None

        symbol_to_fetch = symbol_to_fetch.upper() # Ensure uppercase

        try:
            # self.logger.debug(f"Fetching price for symbol: {symbol_to_fetch}") # Optional debug log
            ticker = self.client.ticker_price(symbol=symbol_to_fetch)
            
            if ticker and isinstance(ticker, dict) and 'price' in ticker:
                price_str = ticker['price']
                try:
                    price = float(price_str)
                    if price > 0:
                        # self.logger.debug(f"Price for {symbol_to_fetch}: {price}") # Optional debug log
                        return price
                    else:
                        self.logger.warning(f"Received non-positive price for {symbol_to_fetch}: {price}")
                        return None
                except ValueError:
                    self.logger.warning(f"Could not convert price '{price_str}' to float for {symbol_to_fetch}.")
                    return None
            else:
                self.logger.warning(f"Invalid or incomplete price data received for {symbol_to_fetch} from API: {ticker}")
                return None
        except Exception as e:
            # Check if the error is from Binance API about invalid symbol
            if "Invalid symbol" in str(e):
                 self.logger.error(f"Error getting price for {symbol_to_fetch}: Invalid symbol according to Binance API.")
            else:
                 self.logger.error(f"Error getting price for {symbol_to_fetch}: {str(e)}")
            return None

    def open_orders(self, pair: str):
        """Obtiene las órdenes abiertas"""
        return self._request(self.client.get_open_orders, symbol=pair)

    def candlestick(self, start_str: str = None, end_str: str = None, limit: int = 1500) -> pd.DataFrame:
        """Get candlestick data for the symbol, optionally for a given date range."""
        try:
            params = {
                'symbol': self.symbol,
                'interval': self.temporality,
                'limit': limit
            }

            if start_str:
                dt_start = pd.to_datetime(start_str, utc=True)
                params['startTime'] = int(dt_start.timestamp() * 1000)
            
            if end_str:
                # Ensure the entire end_str day is included
                dt_end = pd.to_datetime(end_str, utc=True) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
                params['endTime'] = int(dt_end.timestamp() * 1000)

            klines = self.client.klines(**params)

            if not klines:
                # En lugar de ValueError, podrías retornar un DataFrame vacío o None
                # y manejarlo en el script que llama.
                self.logger.warning(f"No candlestick data received for {self.symbol}")
                return pd.DataFrame() # Devolver DataFrame vacío
                
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Seleccionar solo las columnas que necesitas antes de la conversión
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Convertir timestamp a datetime y establecerlo como índice
            # El timestamp de Binance es el tiempo de APERTURA de la vela, en milisegundos UTC
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True) # <--- CLAVE: Establecer como índice

            # Convertir a zona horaria de Ecuador (UTC-5) y remover timezone info
            

            # Convertir columnas de precios y volumen a tipos numéricos
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Verificar si hay valores NaN después de la conversión numérica y eliminarlos
            if df[numeric_columns].isna().any().any():
                self.logger.warning(f"Found NaN values in OHLCV data for {self.symbol} after numeric conversion. Dropping rows with NaNs.")
                df.dropna(subset=numeric_columns, inplace=True) # Modificar inplace
            
            if df.empty or len(df) < 2: # Chequear si el DataFrame está vacío después de limpiar
                self.logger.warning(f"Insufficient data for {self.symbol} after cleaning. DataFrame is empty or has less than 2 rows.")
                return pd.DataFrame() # Devolver DataFrame vacío
            
            # Ya no necesitas retornar 'timestamp' como columna porque es el índice
            return df[['open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            self.logger.error(f"Error in candlestick data retrieval for {self.symbol}: {e}")
            # Considera qué devolver en caso de error. ¿Un DataFrame vacío? ¿None?
            # Lanzar la excepción permite que el código que llama maneje el error.
            raise # Opcionalmente: return pd.DataFrame()

    def change_leverage(self, symbol: str, leverage: int):
        """Cambia el apalancamiento para un símbolo"""
        return self._request(
            self.client.change_leverage,
            symbol=symbol,
            leverage=leverage
        )

    def place_order(self, symbol: str, side: str, quantity: float):
        """
        Coloca una orden de mercado
        Args:
            symbol: Par de trading
            side: 'BUY' o 'SELL'
            quantity: Cantidad a operar
        """
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': quantity,
            'recvWindow': 60000
        }
        
        return self._request(self.client.new_order, **params)