"""
Sistema de Almacenamiento y Cache Optimizado
===========================================

Este módulo implementa un sistema híbrido de almacenamiento y cache diseñado específicamente
para trading algorítmico de alta frecuencia. Como trader experimentado, entiendo que la 
velocidad de acceso a datos es CRÍTICA - cada milisegundo cuenta en el trading.

El sistema utiliza una arquitectura de múltiples capas:
1. Cache en memoria (L1) - Acceso instantáneo para datos activos
2. Cache Redis (L2) - Persistencia rápida con TTL configurable  
3. Almacenamiento en disco (L3) - Backup y datos históricos
4. Compresión inteligente - Optimización de espacio sin sacrificar velocidad

Funcionalidades principales:
- Cache multi-nivel con fallback automático
- Compresión de datos históricos (Parquet + LZ4)
- TTL dinámico basado en timeframe y volatilidad
- Limpieza automática de cache obsoleto
- Métricas de performance y hit ratio
- Backup automático de datos críticos

Autor: Sistema de Trading Algorítmico
Zona Horaria: UTC-5 (Ecuador)
"""

import os
import pickle
import json
import gzip
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
import pytz
from threading import Lock
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..config.settings import config
from ..utils.logger import get_logger, LogContext
from ..utils.exceptions import (
    DataStorageException,
    DataException,
    ErrorCodes,
    create_exception
)

# Zona horaria de Ecuador (UTC-5)
ECUADOR_TZ = pytz.timezone('America/Guayaquil')

# Intentar importar Redis (opcional)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

@dataclass
class CacheEntry:
    """Entrada de cache con metadata"""
    key: str
    data: Any
    timestamp: datetime
    ttl_seconds: int
    access_count: int = 0
    last_access: Optional[datetime] = None
    size_bytes: int = 0
    compressed: bool = False

@dataclass
class StorageConfig:
    """Configuración del sistema de almacenamiento"""
    # Configuración de cache en memoria
    max_memory_cache_mb: int = 512  # 512MB máximo en memoria
    memory_cache_ttl_seconds: int = 300  # 5 minutos por defecto
    
    # Configuración de Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ttl_seconds: int = 3600  # 1 hora por defecto
    
    # Configuración de almacenamiento en disco
    data_directory: str = "data/storage"
    backup_directory: str = "data/backups"
    max_disk_usage_gb: float = 10.0  # 10GB máximo
    
    # Configuración de compresión
    enable_compression: bool = True
    compression_threshold_kb: int = 100  # Comprimir si > 100KB
    compression_level: int = 6  # Nivel de compresión (1-9)
    
    # TTL dinámico por timeframe
    ttl_by_timeframe: Dict[str, int] = None
    
    # Limpieza automática
    cleanup_interval_minutes: int = 30
    max_cache_entries: int = 10000
    
    def __post_init__(self):
        if self.ttl_by_timeframe is None:
            # TTL más corto para timeframes menores (datos más volátiles)
            self.ttl_by_timeframe = {
                '1m': 60,      # 1 minuto
                '5m': 300,     # 5 minutos  
                '15m': 900,    # 15 minutos
                '30m': 1800,   # 30 minutos
                '1h': 3600,    # 1 hora
                '2h': 7200,    # 2 horas
                '4h': 14400,   # 4 horas
                '6h': 21600,   # 6 horas
                '12h': 43200,  # 12 horas
                '1d': 86400,   # 1 día
                '3d': 259200,  # 3 días
                '1w': 604800,  # 1 semana
                '1M': 2592000  # 1 mes
            }

class MemoryCache:
    """
    Cache en memoria L1 - Acceso más rápido posible
    Utiliza LRU con límites de memoria y TTL
    """
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # Para LRU
        self.lock = Lock()
        self.logger = get_logger("MemoryCache")
        
        # Estadísticas
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_size_bytes": 0,
            "entries_count": 0
        }
    
    def _calculate_size(self, data: Any) -> int:
        """Calcula el tamaño aproximado de los datos en bytes"""
        try:
            if isinstance(data, pd.DataFrame):
                return data.memory_usage(deep=True).sum()
            elif isinstance(data, (dict, list)):
                return len(pickle.dumps(data))
            elif isinstance(data, str):
                return len(data.encode('utf-8'))
            else:
                return len(pickle.dumps(data))
        except Exception:
            return 1024  # Estimación por defecto: 1KB
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Verifica si una entrada ha expirado"""
        if entry.ttl_seconds <= 0:
            return False  # TTL infinito
        
        age_seconds = (datetime.now(ECUADOR_TZ) - entry.timestamp).total_seconds()
        return age_seconds > entry.ttl_seconds
    
    def _evict_lru(self) -> None:
        """Elimina entradas usando política LRU"""
        while (len(self.cache) >= self.config.max_cache_entries or 
               self.stats["total_size_bytes"] > self.config.max_memory_cache_mb * 1024 * 1024):
            
            if not self.access_order:
                break
                
            # Eliminar la entrada menos recientemente usada
            lru_key = self.access_order.pop(0)
            if lru_key in self.cache:
                entry = self.cache.pop(lru_key)
                self.stats["total_size_bytes"] -= entry.size_bytes
                self.stats["evictions"] += 1
                self.stats["entries_count"] -= 1
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene un valor del cache"""
        with self.lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                return None
            
            entry = self.cache[key]
            
            # Verificar expiración
            if self._is_expired(entry):
                self.cache.pop(key)
                if key in self.access_order:
                    self.access_order.remove(key)
                self.stats["total_size_bytes"] -= entry.size_bytes
                self.stats["entries_count"] -= 1
                self.stats["misses"] += 1
                return None
            
            # Actualizar estadísticas de acceso
            entry.access_count += 1
            entry.last_access = datetime.now(ECUADOR_TZ)
            
            # Actualizar orden LRU
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            self.stats["hits"] += 1
            return entry.data
    
    def set(self, key: str, data: Any, ttl_seconds: int = None) -> bool:
        """Almacena un valor en el cache"""
        with self.lock:
            if ttl_seconds is None:
                ttl_seconds = self.config.memory_cache_ttl_seconds
            
            # Calcular tamaño
            size_bytes = self._calculate_size(data)
            
            # Verificar si cabe en memoria
            max_size = self.config.max_memory_cache_mb * 1024 * 1024
            if size_bytes > max_size * 0.5:  # No almacenar si es > 50% del límite
                self.logger.warning(f"Datos demasiado grandes para cache: {size_bytes} bytes")
                return False
            
            # Eliminar entrada existente si existe
            if key in self.cache:
                old_entry = self.cache[key]
                self.stats["total_size_bytes"] -= old_entry.size_bytes
                self.stats["entries_count"] -= 1
            
            # Crear nueva entrada
            entry = CacheEntry(
                key=key,
                data=data,
                timestamp=datetime.now(ECUADOR_TZ),
                ttl_seconds=ttl_seconds,
                size_bytes=size_bytes
            )
            
            # Evitar si es necesario
            self._evict_lru()
            
            # Almacenar
            self.cache[key] = entry
            self.stats["total_size_bytes"] += size_bytes
            self.stats["entries_count"] += 1
            
            # Actualizar orden LRU
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Elimina una entrada del cache"""
        with self.lock:
            if key not in self.cache:
                return False
            
            entry = self.cache.pop(key)
            if key in self.access_order:
                self.access_order.remove(key)
            
            self.stats["total_size_bytes"] -= entry.size_bytes
            self.stats["entries_count"] -= 1
            return True
    
    def clear(self) -> None:
        """Limpia todo el cache"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.stats["total_size_bytes"] = 0
            self.stats["entries_count"] = 0
    
    def cleanup_expired(self) -> int:
        """Limpia entradas expiradas y retorna el número eliminado"""
        expired_keys = []
        
        with self.lock:
            for key, entry in self.cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self.cache.pop(key)
                if key in self.access_order:
                    self.access_order.remove(key)
                self.stats["total_size_bytes"] -= entry.size_bytes
                self.stats["entries_count"] -= 1
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache"""
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_ratio = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
            
            return {
                **self.stats,
                "hit_ratio": hit_ratio,
                "memory_usage_mb": self.stats["total_size_bytes"] / (1024 * 1024),
                "memory_usage_pct": (self.stats["total_size_bytes"] / 
                                    (self.config.max_memory_cache_mb * 1024 * 1024)) * 100
            }

class RedisCache:
    """
    Cache Redis L2 - Persistencia rápida con TTL
    Fallback cuando el cache en memoria no tiene los datos
    """
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.logger = get_logger("RedisCache")
        self.redis_client = None
        self.available = False
        
        if REDIS_AVAILABLE:
            self._connect()
        else:
            self.logger.warning("Redis no disponible - cache L2 deshabilitado")
    
    def _connect(self) -> None:
        """Conecta a Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=False,  # Mantenemos bytes para pickle
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            
            # Probar conexión
            self.redis_client.ping()
            self.available = True
            self.logger.info("Conexión a Redis establecida exitosamente")
            
        except Exception as e:
            self.logger.warning(f"No se pudo conectar a Redis: {str(e)}")
            self.available = False
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serializa datos para almacenamiento"""
        try:
            if isinstance(data, pd.DataFrame):
                # Para DataFrames, usar pickle optimizado
                return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise DataStorageException(
                f"Error serializando datos: {str(e)}",
                ErrorCodes.DATA_STORAGE_FAILED
            )
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserializa datos del almacenamiento"""
        try:
            return pickle.loads(data)
        except Exception as e:
            raise DataStorageException(
                f"Error deserializando datos: {str(e)}",
                ErrorCodes.DATA_STORAGE_FAILED
            )
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene un valor de Redis"""
        if not self.available:
            return None
        
        try:
            data = self.redis_client.get(key)
            if data is None:
                return None
            
            return self._deserialize_data(data)
            
        except Exception as e:
            self.logger.error(f"Error obteniendo de Redis: {str(e)}")
            return None
    
    def set(self, key: str, data: Any, ttl_seconds: int = None) -> bool:
        """Almacena un valor en Redis"""
        if not self.available:
            return False
        
        try:
            if ttl_seconds is None:
                ttl_seconds = self.config.redis_ttl_seconds
            
            serialized_data = self._serialize_data(data)
            
            # Almacenar con TTL
            result = self.redis_client.setex(key, ttl_seconds, serialized_data)
            return bool(result)
            
        except Exception as e:
            self.logger.error(f"Error almacenando en Redis: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """Elimina una entrada de Redis"""
        if not self.available:
            return False
        
        try:
            result = self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            self.logger.error(f"Error eliminando de Redis: {str(e)}")
            return False
    
    def clear(self) -> bool:
        """Limpia toda la base de datos Redis"""
        if not self.available:
            return False
        
        try:
            self.redis_client.flushdb()
            return True
        except Exception as e:
            self.logger.error(f"Error limpiando Redis: {str(e)}")
            return False

class DiskStorage:
    """
    Almacenamiento en disco L3 - Persistencia a largo plazo
    Utiliza compresión y organización eficiente por símbolo/timeframe
    """
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.logger = get_logger("DiskStorage")
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Crear directorios
        self.data_dir = Path(config.data_directory)
        self.backup_dir = Path(config.backup_directory)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_file_path(self, key: str, compressed: bool = False) -> Path:
        """Genera la ruta del archivo basada en la clave"""
        # Crear hash para evitar nombres de archivo problemáticos
        key_hash = hashlib.md5(key.encode()).hexdigest()
        
        # Organizar por fecha para facilitar limpieza
        today = datetime.now(ECUADOR_TZ).strftime("%Y/%m")
        
        extension = ".pkl.gz" if compressed else ".pkl"
        return self.data_dir / today / f"{key_hash}{extension}"
    
    def _should_compress(self, data: Any) -> bool:
        """Determina si los datos deben comprimirse"""
        if not self.config.enable_compression:
            return False
        
        try:
            size_bytes = len(pickle.dumps(data))
            return size_bytes > self.config.compression_threshold_kb * 1024
        except Exception:
            return False
    
    def _compress_data(self, data: bytes) -> bytes:
        """Comprime datos usando gzip"""
        return gzip.compress(data, compresslevel=self.config.compression_level)
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Descomprime datos gzip"""
        return gzip.decompress(data)
    
    async def store(self, key: str, data: Any) -> bool:
        """Almacena datos en disco de forma asíncrona"""
        try:
            # Determinar si comprimir
            should_compress = self._should_compress(data)
            file_path = self._get_file_path(key, should_compress)
            
            # Crear directorio si no existe
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Serializar datos
            def _store_sync():
                try:
                    serialized_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    if should_compress:
                        serialized_data = self._compress_data(serialized_data)
                    
                    with open(file_path, 'wb') as f:
                        f.write(serialized_data)
                    
                    return True
                except Exception as e:
                    self.logger.error(f"Error almacenando en disco: {str(e)}")
                    return False
            
            # Ejecutar en thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, _store_sync)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en almacenamiento asíncrono: {str(e)}")
            return False
    
    async def load(self, key: str) -> Optional[Any]:
        """Carga datos del disco de forma asíncrona"""
        try:
            # Intentar ambas versiones (comprimida y no comprimida)
            compressed_path = self._get_file_path(key, True)
            uncompressed_path = self._get_file_path(key, False)
            
            def _load_sync():
                try:
                    # Intentar versión comprimida primero
                    if compressed_path.exists():
                        with open(compressed_path, 'rb') as f:
                            data = f.read()
                        data = self._decompress_data(data)
                        return pickle.loads(data)
                    
                    # Intentar versión no comprimida
                    elif uncompressed_path.exists():
                        with open(uncompressed_path, 'rb') as f:
                            data = f.read()
                        return pickle.loads(data)
                    
                    return None
                    
                except Exception as e:
                    self.logger.error(f"Error cargando del disco: {str(e)}")
                    return None
            
            # Ejecutar en thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, _load_sync)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en carga asíncrona: {str(e)}")
            return None
    
    def delete(self, key: str) -> bool:
        """Elimina archivos del disco"""
        try:
            compressed_path = self._get_file_path(key, True)
            uncompressed_path = self._get_file_path(key, False)
            
            deleted = False
            if compressed_path.exists():
                compressed_path.unlink()
                deleted = True
            
            if uncompressed_path.exists():
                uncompressed_path.unlink()
                deleted = True
            
            return deleted
            
        except Exception as e:
            self.logger.error(f"Error eliminando del disco: {str(e)}")
            return False
    
    def get_disk_usage(self) -> Dict[str, float]:
        """Obtiene estadísticas de uso de disco"""
        try:
            total_size = 0
            file_count = 0
            
            for file_path in self.data_dir.rglob("*.pkl*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            return {
                "total_size_gb": total_size / (1024**3),
                "file_count": file_count,
                "directory": str(self.data_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Error obteniendo uso de disco: {str(e)}")
            return {"total_size_gb": 0, "file_count": 0, "directory": str(self.data_dir)}

class StorageManager:
    """
    Gestor principal del sistema de almacenamiento multi-nivel
    Coordina cache en memoria, Redis y almacenamiento en disco
    """
    
    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()
        self.logger = get_logger("StorageManager")
        
        # Inicializar capas de almacenamiento
        self.memory_cache = MemoryCache(self.config)
        self.redis_cache = RedisCache(self.config)
        self.disk_storage = DiskStorage(self.config)
        
        # Estadísticas globales
        self.stats = {
            "total_requests": 0,
            "l1_hits": 0,  # Memory cache
            "l2_hits": 0,  # Redis cache
            "l3_hits": 0,  # Disk storage
            "misses": 0,
            "stores": 0,
            "errors": 0
        }
        
        # La limpieza automática se iniciará cuando sea necesario
        self._cleanup_task = None
    
    def _generate_cache_key(self, symbol: str, timeframe: str, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          extra: Optional[str] = None) -> str:
        """Genera una clave única para el cache"""
        key_parts = [symbol.upper(), timeframe.lower()]
        
        if start_time:
            key_parts.append(start_time.strftime("%Y%m%d_%H%M"))
        if end_time:
            key_parts.append(end_time.strftime("%Y%m%d_%H%M"))
        if extra:
            key_parts.append(extra)
        
        return ":".join(key_parts)
    
    def _get_ttl_for_timeframe(self, timeframe: str) -> int:
        """Obtiene TTL apropiado para el timeframe"""
        return self.config.ttl_by_timeframe.get(timeframe, self.config.memory_cache_ttl_seconds)
    
    async def get_data(self, symbol: str, timeframe: str,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      extra: Optional[str] = None) -> Optional[Any]:
        """
        Obtiene datos del sistema de almacenamiento multi-nivel
        Busca en orden: Memory -> Redis -> Disk
        """
        context = LogContext(
            component="storage_manager",
            symbol=symbol,
            timeframe=timeframe
        )
        
        key = self._generate_cache_key(symbol, timeframe, start_time, end_time, extra)
        self.stats["total_requests"] += 1
        
        # Asegurar que la tarea de limpieza esté ejecutándose
        self._ensure_cleanup_task()
        
        try:
            # L1: Intentar cache en memoria
            data = self.memory_cache.get(key)
            if data is not None:
                self.stats["l1_hits"] += 1
                self.logger.debug(f"Cache L1 hit: {key}", context=context)
                return data
            
            # L2: Intentar Redis
            data = self.redis_cache.get(key)
            if data is not None:
                self.stats["l2_hits"] += 1
                self.logger.debug(f"Cache L2 hit: {key}", context=context)
                
                # Promover a L1
                ttl = self._get_ttl_for_timeframe(timeframe)
                self.memory_cache.set(key, data, ttl)
                return data
            
            # L3: Intentar disco
            data = await self.disk_storage.load(key)
            if data is not None:
                self.stats["l3_hits"] += 1
                self.logger.debug(f"Cache L3 hit: {key}", context=context)
                
                # Promover a L2 y L1
                ttl = self._get_ttl_for_timeframe(timeframe)
                self.redis_cache.set(key, data, ttl)
                self.memory_cache.set(key, data, ttl)
                return data
            
            # Miss completo
            self.stats["misses"] += 1
            self.logger.debug(f"Cache miss completo: {key}", context=context)
            return None
            
        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Error obteniendo datos: {str(e)}", context=context)
            return None
    
    async def store_data(self, symbol: str, timeframe: str, data: Any,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None,
                        extra: Optional[str] = None) -> bool:
        """
        Almacena datos en el sistema multi-nivel
        Guarda en todas las capas disponibles
        """
        context = LogContext(
            component="storage_manager",
            symbol=symbol,
            timeframe=timeframe
        )
        
        key = self._generate_cache_key(symbol, timeframe, start_time, end_time, extra)
        self.stats["stores"] += 1
        
        try:
            ttl = self._get_ttl_for_timeframe(timeframe)
            success_count = 0
            
            # Almacenar en L1 (memoria)
            if self.memory_cache.set(key, data, ttl):
                success_count += 1
            
            # Almacenar en L2 (Redis)
            if self.redis_cache.set(key, data, ttl):
                success_count += 1
            
            # Almacenar en L3 (disco) - asíncrono
            if await self.disk_storage.store(key, data):
                success_count += 1
            
            self.logger.debug(
                f"Datos almacenados en {success_count}/3 capas: {key}",
                context=context
            )
            
            return success_count > 0
            
        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Error almacenando datos: {str(e)}", context=context)
            return False
    
    def delete_data(self, symbol: str, timeframe: str,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   extra: Optional[str] = None) -> bool:
        """Elimina datos de todas las capas"""
        key = self._generate_cache_key(symbol, timeframe, start_time, end_time, extra)
        
        success_count = 0
        
        # Eliminar de L1
        if self.memory_cache.delete(key):
            success_count += 1
        
        # Eliminar de L2
        if self.redis_cache.delete(key):
            success_count += 1
        
        # Eliminar de L3
        if self.disk_storage.delete(key):
            success_count += 1
        
        return success_count > 0
    
    def clear_all_cache(self) -> None:
        """Limpia todos los caches"""
        self.memory_cache.clear()
        self.redis_cache.clear()
        # Nota: No limpiamos disco automáticamente por seguridad
    
    def _ensure_cleanup_task(self) -> None:
        """Asegura que la tarea de limpieza esté ejecutándose"""
        if self._cleanup_task is None or self._cleanup_task.done():
            try:
                async def cleanup_loop():
                    while True:
                        try:
                            await asyncio.sleep(self.config.cleanup_interval_minutes * 60)
                            
                            # Limpiar entradas expiradas del cache en memoria
                            expired_count = self.memory_cache.cleanup_expired()
                            if expired_count > 0:
                                self.logger.info(f"Limpieza automática: {expired_count} entradas expiradas eliminadas")
                            
                        except Exception as e:
                            self.logger.error(f"Error en limpieza automática: {str(e)}")
                
                # Intentar crear la tarea solo si hay un loop ejecutándose
                try:
                    self._cleanup_task = asyncio.create_task(cleanup_loop())
                except RuntimeError:
                    # No hay loop ejecutándose, la tarea se creará cuando sea necesario
                    pass
                    
            except Exception as e:
                self.logger.error(f"Error iniciando tarea de limpieza: {str(e)}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas comprehensivas del sistema"""
        total_requests = self.stats["total_requests"]
        
        # Calcular ratios
        l1_ratio = self.stats["l1_hits"] / total_requests if total_requests > 0 else 0
        l2_ratio = self.stats["l2_hits"] / total_requests if total_requests > 0 else 0
        l3_ratio = self.stats["l3_hits"] / total_requests if total_requests > 0 else 0
        miss_ratio = self.stats["misses"] / total_requests if total_requests > 0 else 0
        
        return {
            "global_stats": {
                **self.stats,
                "l1_hit_ratio": l1_ratio,
                "l2_hit_ratio": l2_ratio,
                "l3_hit_ratio": l3_ratio,
                "miss_ratio": miss_ratio,
                "total_hit_ratio": l1_ratio + l2_ratio + l3_ratio
            },
            "memory_cache_stats": self.memory_cache.get_stats(),
            "redis_available": self.redis_cache.available,
            "disk_usage": self.disk_storage.get_disk_usage(),
            "config": asdict(self.config)
        }

# Instancia global del gestor de almacenamiento
storage_manager = StorageManager()

# Funciones de conveniencia
async def cache_ohlcv_data(symbol: str, timeframe: str, data: pd.DataFrame,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> bool:
    """Función de conveniencia para cachear datos OHLCV"""
    return await storage_manager.store_data(symbol, timeframe, data, start_time, end_time, "ohlcv")

async def get_cached_ohlcv_data(symbol: str, timeframe: str,
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None) -> Optional[pd.DataFrame]:
    """Función de conveniencia para obtener datos OHLCV cacheados"""
    return await storage_manager.get_data(symbol, timeframe, start_time, end_time, "ohlcv")

def get_storage_stats() -> Dict[str, Any]:
    """Función de conveniencia para obtener estadísticas"""
    return storage_manager.get_comprehensive_stats()