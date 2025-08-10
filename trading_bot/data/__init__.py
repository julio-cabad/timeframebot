"""
Módulo de Gestión de Datos
=========================

Este módulo maneja toda la gestión de datos para el sistema de trading algorítmico:
- Fetching multi-timeframe con rate limiting inteligente
- Validación y control de calidad de datos OHLCV
- Storage y caching multi-nivel optimizado
- Preprocesamiento de indicadores técnicos (próximamente)

Componentes principales:
- fetcher: Obtención asíncrona de datos multi-timeframe
- validator: Sistema de validación y control de calidad
- storage: Cache multi-nivel (memoria + Redis + archivos)
- preprocessor: Indicadores y normalización (próximamente)

Zona Horaria: UTC-5 (Ecuador)
"""

from .fetcher import (
    MultiTimeframeFetcher,
    FetchRequest,
    FetchResult,
    RateLimiter,
    fetcher,
    fetch_symbol_data,
    fetch_all_active_symbols
)

from .validator import (
    DataValidator,
    ValidationConfig,
    ValidationRule,
    ValidationSeverity,
    ValidationIssue,
    ValidationResult,
    validator,
    validate_data,
    quick_validate
)

from .storage import (
    StorageManager,
    StorageConfig,
    MemoryCache,
    DiskStorage,
    CacheEntry,
    storage_manager,
    cache_ohlcv_data,
    get_cached_ohlcv_data,
    get_storage_stats
)

__all__ = [
    # Fetcher components
    "MultiTimeframeFetcher",
    "FetchRequest", 
    "FetchResult",
    "RateLimiter",
    "fetcher",
    "fetch_symbol_data",
    "fetch_all_active_symbols",
    
    # Validator components
    "DataValidator",
    "ValidationConfig",
    "ValidationRule",
    "ValidationSeverity", 
    "ValidationIssue",
    "ValidationResult",
    "validator",
    "validate_data",
    "quick_validate",
    
    # Storage components
    "StorageManager",
    "StorageConfig",
    "MemoryCache",
    "DiskStorage",
    "CacheEntry",
    "storage_manager",
    "cache_ohlcv_data",
    "get_cached_ohlcv_data",
    "get_storage_stats"
]