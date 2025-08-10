"""
Sistema de Validación y Control de Calidad de Datos
====================================================

Este módulo implementa un sistema robusto de validación de datos OHLCV para trading algorítmico.
Como trader con experiencia, sabemos que la calidad de los datos es CRÍTICA para el éxito.
Un solo dato malo puede arruinar una estrategia completa.

Funcionalidades principales:
- Validación de integridad OHLCV (Open, High, Low, Close, Volume)
- Detección de gaps temporales en series de tiempo
- Identificación de outliers y anomalías de precio
- Validación de consistencia entre timeframes
- Análisis de calidad de volumen
- Detección de datos sospechosos o manipulados

Autor: Sistema de Trading Algorítmico
Zona Horaria: UTC-5 (Ecuador)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass
from enum import Enum
import pytz

from ..config.settings import config
from ..utils.logger import get_logger, LogContext
from ..utils.exceptions import (
    DataValidationException,
    DataException,
    ErrorCodes,
    create_exception
)

# Zona horaria de Ecuador (UTC-5)
ECUADOR_TZ = pytz.timezone('America/Guayaquil')

class ValidationSeverity(Enum):
    """Niveles de severidad para problemas de validación"""
    INFO = "info"           # Información, no crítico
    WARNING = "warning"     # Advertencia, revisar pero no bloquear
    ERROR = "error"         # Error, datos no confiables
    CRITICAL = "critical"   # Crítico, datos completamente inválidos

class ValidationRule(Enum):
    """Reglas de validación disponibles"""
    OHLC_CONSISTENCY = "ohlc_consistency"           # H >= O,C >= L y L <= O,C <= H
    PRICE_CONTINUITY = "price_continuity"           # Continuidad entre velas
    VOLUME_SANITY = "volume_sanity"                 # Volumen debe ser >= 0
    TEMPORAL_GAPS = "temporal_gaps"                 # Gaps en series temporales
    OUTLIER_DETECTION = "outlier_detection"         # Detección de outliers
    DUPLICATE_TIMESTAMPS = "duplicate_timestamps"   # Timestamps duplicados
    FUTURE_TIMESTAMPS = "future_timestamps"         # Timestamps en el futuro
    ZERO_PRICE_CHECK = "zero_price_check"          # Precios en cero
    NEGATIVE_SPREAD = "negative_spread"             # Spread negativo H-L
    EXTREME_MOVEMENTS = "extreme_movements"         # Movimientos extremos

@dataclass
class ValidationIssue:
    """Representa un problema encontrado durante la validación"""
    rule: ValidationRule
    severity: ValidationSeverity
    message: str
    timestamp: Optional[datetime] = None
    row_index: Optional[int] = None
    value: Optional[float] = None
    expected_range: Optional[Tuple[float, float]] = None
    suggestion: Optional[str] = None

class ValidationResult(NamedTuple):
    """Resultado de la validación de datos"""
    is_valid: bool
    quality_score: float  # 0.0 a 1.0
    issues: List[ValidationIssue]
    summary: Dict[str, Any]
    processed_data: Optional[pd.DataFrame] = None

@dataclass
class ValidationConfig:
    """Configuración para el validador de datos"""
    # Tolerancias para detección de outliers (en desviaciones estándar)
    price_outlier_threshold: float = 4.0
    volume_outlier_threshold: float = 5.0
    
    # Máximo gap temporal permitido (en múltiplos del timeframe)
    max_temporal_gap_multiplier: float = 2.0
    
    # Máximo movimiento de precio permitido entre velas (porcentaje)
    max_price_movement_pct: float = 0.20  # 20%
    
    # Mínimo volumen esperado (percentil)
    min_volume_percentile: float = 0.01  # 1%
    
    # Reglas habilitadas por defecto
    enabled_rules: List[ValidationRule] = None
    
    # Severidad mínima para considerar datos inválidos
    min_severity_for_rejection: ValidationSeverity = ValidationSeverity.ERROR
    
    def __post_init__(self):
        if self.enabled_rules is None:
            self.enabled_rules = list(ValidationRule)

class DataValidator:
    """
    Validador avanzado de datos OHLCV para trading algorítmico
    
    Este validador implementa múltiples capas de validación basadas en
    experiencia real de trading. Cada regla está diseñada para detectar
    problemas que pueden afectar la rentabilidad de las estrategias.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.logger = get_logger("DataValidator")
        
        # Estadísticas de validación
        self.stats = {
            "total_validations": 0,
            "passed_validations": 0,
            "failed_validations": 0,
            "issues_by_rule": {rule.value: 0 for rule in ValidationRule},
            "issues_by_severity": {sev.value: 0 for sev in ValidationSeverity}
        }
    
    def validate_ohlcv_data(self, data: pd.DataFrame, symbol: str, 
                           timeframe: str) -> ValidationResult:
        """
        Valida un DataFrame de datos OHLCV completo
        
        Args:
            data: DataFrame con columnas OHLCV
            symbol: Símbolo del activo (ej: 'BTCUSDT')
            timeframe: Timeframe de los datos (ej: '1h', '4h')
            
        Returns:
            ValidationResult con el resultado completo de la validación
        """
        context = LogContext(
            component="data_validator",
            symbol=symbol,
            timeframe=timeframe
        )
        
        self.logger.debug(
            f"Iniciando validación de {len(data)} velas para {symbol} {timeframe}",
            context=context
        )
        
        if data.empty:
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                issues=[ValidationIssue(
                    rule=ValidationRule.OHLC_CONSISTENCY,
                    severity=ValidationSeverity.CRITICAL,
                    message="DataFrame vacío - no hay datos para validar"
                )],
                summary={"empty_data": True}
            )
        
        # Lista para acumular todos los problemas encontrados
        all_issues = []
        
        # Ejecutar todas las reglas de validación habilitadas
        for rule in self.config.enabled_rules:
            try:
                issues = self._apply_validation_rule(data, rule, symbol, timeframe)
                all_issues.extend(issues)
                
                # Actualizar estadísticas
                self.stats["issues_by_rule"][rule.value] += len(issues)
                for issue in issues:
                    self.stats["issues_by_severity"][issue.severity.value] += 1
                    
            except Exception as e:
                self.logger.error(
                    f"Error aplicando regla {rule.value}: {str(e)}",
                    context=context
                )
                all_issues.append(ValidationIssue(
                    rule=rule,
                    severity=ValidationSeverity.ERROR,
                    message=f"Error interno en validación: {str(e)}"
                ))
        
        # Calcular score de calidad y determinar si es válido
        quality_score = self._calculate_quality_score(all_issues, len(data))
        is_valid = self._determine_validity(all_issues)
        
        # Crear resumen
        summary = self._create_validation_summary(all_issues, data, symbol, timeframe)
        
        # Actualizar estadísticas globales
        self.stats["total_validations"] += 1
        if is_valid:
            self.stats["passed_validations"] += 1
        else:
            self.stats["failed_validations"] += 1
        
        result = ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            issues=all_issues,
            summary=summary,
            processed_data=data.copy() if is_valid else None
        )
        
        self.logger.info(
            f"Validación completada: {'✅ VÁLIDO' if is_valid else '❌ INVÁLIDO'} "
            f"(Score: {quality_score:.3f}, Issues: {len(all_issues)})",
            context=context,
            extra_fields={
                "is_valid": is_valid,
                "quality_score": quality_score,
                "issues_count": len(all_issues),
                "data_points": len(data)
            }
        )
        
        return result
    
    def _apply_validation_rule(self, data: pd.DataFrame, rule: ValidationRule,
                              symbol: str, timeframe: str) -> List[ValidationIssue]:
        """Aplica una regla específica de validación"""
        
        if rule == ValidationRule.OHLC_CONSISTENCY:
            return self._validate_ohlc_consistency(data)
        elif rule == ValidationRule.PRICE_CONTINUITY:
            return self._validate_price_continuity(data)
        elif rule == ValidationRule.VOLUME_SANITY:
            return self._validate_volume_sanity(data)
        elif rule == ValidationRule.TEMPORAL_GAPS:
            return self._validate_temporal_gaps(data, timeframe)
        elif rule == ValidationRule.OUTLIER_DETECTION:
            return self._validate_outliers(data)
        elif rule == ValidationRule.DUPLICATE_TIMESTAMPS:
            return self._validate_duplicate_timestamps(data)
        elif rule == ValidationRule.FUTURE_TIMESTAMPS:
            return self._validate_future_timestamps(data)
        elif rule == ValidationRule.ZERO_PRICE_CHECK:
            return self._validate_zero_prices(data)
        elif rule == ValidationRule.NEGATIVE_SPREAD:
            return self._validate_negative_spreads(data)
        elif rule == ValidationRule.EXTREME_MOVEMENTS:
            return self._validate_extreme_movements(data)
        else:
            return []
    
    def _validate_ohlc_consistency(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """
        Valida la consistencia OHLC: High >= Open,Close >= Low
        Esta es la regla MÁS IMPORTANTE - datos inconsistentes son completamente inútiles
        """
        issues = []
        
        for idx, row in data.iterrows():
            # Verificar que High sea el mayor
            if not (row['high'] >= row['open'] and row['high'] >= row['close'] and row['high'] >= row['low']):
                issues.append(ValidationIssue(
                    rule=ValidationRule.OHLC_CONSISTENCY,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"High ({row['high']}) no es el precio más alto de la vela",
                    timestamp=idx if isinstance(idx, datetime) else None,
                    row_index=idx if isinstance(idx, int) else None,
                    value=row['high'],
                    suggestion="Verificar fuente de datos - posible corrupción"
                ))
            
            # Verificar que Low sea el menor
            if not (row['low'] <= row['open'] and row['low'] <= row['close'] and row['low'] <= row['high']):
                issues.append(ValidationIssue(
                    rule=ValidationRule.OHLC_CONSISTENCY,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Low ({row['low']}) no es el precio más bajo de la vela",
                    timestamp=idx if isinstance(idx, datetime) else None,
                    row_index=idx if isinstance(idx, int) else None,
                    value=row['low'],
                    suggestion="Verificar fuente de datos - posible corrupción"
                ))
        
        return issues
    
    def _validate_price_continuity(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """
        Valida continuidad de precios entre velas consecutivas
        Gaps extremos pueden indicar problemas de datos o eventos de mercado
        """
        issues = []
        
        if len(data) < 2:
            return issues
        
        for i in range(1, len(data)):
            prev_close = data.iloc[i-1]['close']
            curr_open = data.iloc[i]['open']
            
            # Calcular gap como porcentaje
            gap_pct = abs(curr_open - prev_close) / prev_close
            
            # Gap mayor al umbral configurado
            if gap_pct > self.config.max_price_movement_pct:
                severity = ValidationSeverity.WARNING if gap_pct < 0.5 else ValidationSeverity.ERROR
                
                issues.append(ValidationIssue(
                    rule=ValidationRule.PRICE_CONTINUITY,
                    severity=severity,
                    message=f"Gap de precio extremo: {gap_pct:.2%} entre velas",
                    timestamp=data.index[i] if hasattr(data.index, 'to_pydatetime') else None,
                    row_index=i,
                    value=gap_pct,
                    expected_range=(0.0, self.config.max_price_movement_pct),
                    suggestion="Verificar si hubo evento de mercado o problema de datos"
                ))
        
        return issues
    
    def _validate_volume_sanity(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """
        Valida que los volúmenes sean sensatos
        Volumen es crucial para confirmar movimientos de precio
        """
        issues = []
        
        # Verificar volúmenes negativos
        negative_volume = data['volume'] < 0
        if negative_volume.any():
            for idx in data[negative_volume].index:
                issues.append(ValidationIssue(
                    rule=ValidationRule.VOLUME_SANITY,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Volumen negativo: {data.loc[idx, 'volume']}",
                    timestamp=idx if isinstance(idx, datetime) else None,
                    row_index=idx if isinstance(idx, int) else None,
                    value=data.loc[idx, 'volume'],
                    suggestion="Volumen negativo es imposible - corregir datos"
                ))
        
        # Verificar volúmenes extremadamente bajos (posibles errores)
        if len(data) > 10:  # Solo si tenemos suficientes datos
            volume_threshold = data['volume'].quantile(self.config.min_volume_percentile)
            zero_volume = data['volume'] == 0
            
            if zero_volume.sum() > len(data) * 0.1:  # Más del 10% con volumen cero
                issues.append(ValidationIssue(
                    rule=ValidationRule.VOLUME_SANITY,
                    severity=ValidationSeverity.WARNING,
                    message=f"Demasiadas velas con volumen cero: {zero_volume.sum()}/{len(data)}",
                    value=zero_volume.sum(),
                    suggestion="Verificar si el mercado estuvo cerrado o hay problemas de datos"
                ))
        
        return issues
    
    def _validate_temporal_gaps(self, data: pd.DataFrame, timeframe: str) -> List[ValidationIssue]:
        """
        Valida que no haya gaps temporales excesivos
        Los gaps pueden indicar datos faltantes o problemas de conectividad
        """
        issues = []
        
        if not isinstance(data.index, pd.DatetimeIndex) or len(data) < 2:
            return issues
        
        # Convertir timeframe a timedelta
        timeframe_delta = self._timeframe_to_timedelta(timeframe)
        if timeframe_delta is None:
            return issues
        
        max_gap = timeframe_delta * self.config.max_temporal_gap_multiplier
        
        for i in range(1, len(data)):
            time_diff = data.index[i] - data.index[i-1]
            
            if time_diff > max_gap:
                issues.append(ValidationIssue(
                    rule=ValidationRule.TEMPORAL_GAPS,
                    severity=ValidationSeverity.WARNING,
                    message=f"Gap temporal excesivo: {time_diff} (esperado: ~{timeframe_delta})",
                    timestamp=data.index[i],
                    row_index=i,
                    value=time_diff.total_seconds(),
                    expected_range=(0, max_gap.total_seconds()),
                    suggestion="Verificar si faltan datos en este período"
                ))
        
        return issues
    
    def _validate_outliers(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """
        Detecta outliers estadísticos en precios y volúmenes
        Outliers extremos pueden ser errores de datos o eventos excepcionales
        """
        issues = []
        
        if len(data) < 10:  # Necesitamos suficientes datos para estadísticas
            return issues
        
        # Detectar outliers en precios (usando precio típico)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        price_mean = typical_price.mean()
        price_std = typical_price.std()
        
        price_outliers = np.abs(typical_price - price_mean) > (self.config.price_outlier_threshold * price_std)
        
        for idx in data[price_outliers].index:
            issues.append(ValidationIssue(
                rule=ValidationRule.OUTLIER_DETECTION,
                severity=ValidationSeverity.WARNING,
                message=f"Outlier de precio detectado: precio típico {typical_price[idx]:.2f}",
                timestamp=idx if isinstance(idx, datetime) else None,
                row_index=idx if isinstance(idx, int) else None,
                value=typical_price[idx],
                expected_range=(price_mean - 3*price_std, price_mean + 3*price_std),
                suggestion="Verificar si hubo evento de mercado significativo"
            ))
        
        # Detectar outliers en volumen
        if data['volume'].std() > 0:  # Evitar división por cero
            volume_mean = data['volume'].mean()
            volume_std = data['volume'].std()
            
            volume_outliers = np.abs(data['volume'] - volume_mean) > (self.config.volume_outlier_threshold * volume_std)
            
            for idx in data[volume_outliers].index:
                issues.append(ValidationIssue(
                    rule=ValidationRule.OUTLIER_DETECTION,
                    severity=ValidationSeverity.INFO,
                    message=f"Outlier de volumen detectado: {data.loc[idx, 'volume']:.0f}",
                    timestamp=idx if isinstance(idx, datetime) else None,
                    row_index=idx if isinstance(idx, int) else None,
                    value=data.loc[idx, 'volume'],
                    suggestion="Volumen anómalo - posible evento de mercado"
                ))
        
        return issues
    
    def _validate_duplicate_timestamps(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Detecta timestamps duplicados"""
        issues = []
        
        if isinstance(data.index, pd.DatetimeIndex):
            duplicates = data.index.duplicated()
            if duplicates.any():
                for idx in data[duplicates].index:
                    issues.append(ValidationIssue(
                        rule=ValidationRule.DUPLICATE_TIMESTAMPS,
                        severity=ValidationSeverity.ERROR,
                        message=f"Timestamp duplicado: {idx}",
                        timestamp=idx,
                        suggestion="Eliminar velas duplicadas"
                    ))
        
        return issues
    
    def _validate_future_timestamps(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Detecta timestamps en el futuro"""
        issues = []
        
        if isinstance(data.index, pd.DatetimeIndex):
            # Obtener tiempo actual en zona horaria de Ecuador
            now_ecuador = datetime.now(ECUADOR_TZ)
            
            future_timestamps = data.index > now_ecuador
            if future_timestamps.any():
                for idx in data[future_timestamps].index:
                    issues.append(ValidationIssue(
                        rule=ValidationRule.FUTURE_TIMESTAMPS,
                        severity=ValidationSeverity.ERROR,
                        message=f"Timestamp en el futuro: {idx}",
                        timestamp=idx,
                        suggestion="Verificar sincronización de tiempo"
                    ))
        
        return issues
    
    def _validate_zero_prices(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Detecta precios en cero (imposibles en mercados reales)"""
        issues = []
        
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            zero_prices = data[col] <= 0
            if zero_prices.any():
                for idx in data[zero_prices].index:
                    issues.append(ValidationIssue(
                        rule=ValidationRule.ZERO_PRICE_CHECK,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Precio {col} en cero o negativo: {data.loc[idx, col]}",
                        timestamp=idx if isinstance(idx, datetime) else None,
                        row_index=idx if isinstance(idx, int) else None,
                        value=data.loc[idx, col],
                        suggestion="Precios <= 0 son imposibles - corregir datos"
                    ))
        
        return issues
    
    def _validate_negative_spreads(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Detecta spreads negativos (High < Low)"""
        issues = []
        
        negative_spreads = data['high'] < data['low']
        if negative_spreads.any():
            for idx in data[negative_spreads].index:
                issues.append(ValidationIssue(
                    rule=ValidationRule.NEGATIVE_SPREAD,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Spread negativo: High {data.loc[idx, 'high']} < Low {data.loc[idx, 'low']}",
                    timestamp=idx if isinstance(idx, datetime) else None,
                    row_index=idx if isinstance(idx, int) else None,
                    value=data.loc[idx, 'high'] - data.loc[idx, 'low'],
                    suggestion="High debe ser >= Low siempre"
                ))
        
        return issues
    
    def _validate_extreme_movements(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Detecta movimientos de precio extremos dentro de una vela"""
        issues = []
        
        for idx, row in data.iterrows():
            # Calcular rango de la vela como porcentaje del precio de apertura
            if row['open'] > 0:
                range_pct = (row['high'] - row['low']) / row['open']
                
                # Movimiento extremo (más del 50% en una vela)
                if range_pct > 0.5:
                    issues.append(ValidationIssue(
                        rule=ValidationRule.EXTREME_MOVEMENTS,
                        severity=ValidationSeverity.WARNING,
                        message=f"Movimiento extremo en vela: {range_pct:.1%}",
                        timestamp=idx if isinstance(idx, datetime) else None,
                        row_index=idx if isinstance(idx, int) else None,
                        value=range_pct,
                        suggestion="Verificar si hubo evento de mercado excepcional"
                    ))
        
        return issues
    
    def _timeframe_to_timedelta(self, timeframe: str) -> Optional[timedelta]:
        """Convierte un timeframe string a timedelta"""
        timeframe_map = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '2h': timedelta(hours=2),
            '4h': timedelta(hours=4),
            '6h': timedelta(hours=6),
            '8h': timedelta(hours=8),
            '12h': timedelta(hours=12),
            '1d': timedelta(days=1),
            '3d': timedelta(days=3),
            '1w': timedelta(weeks=1),
            '1M': timedelta(days=30),  # Aproximado
        }
        return timeframe_map.get(timeframe)
    
    def _calculate_quality_score(self, issues: List[ValidationIssue], 
                                data_points: int) -> float:
        """
        Calcula un score de calidad de 0.0 a 1.0
        Basado en la severidad y cantidad de problemas encontrados
        """
        if not issues:
            return 1.0
        
        # Pesos por severidad (más peso = mayor penalización)
        severity_weights = {
            ValidationSeverity.INFO: 0.01,
            ValidationSeverity.WARNING: 0.05,
            ValidationSeverity.ERROR: 0.20,
            ValidationSeverity.CRITICAL: 0.50
        }
        
        total_penalty = 0.0
        for issue in issues:
            penalty = severity_weights.get(issue.severity, 0.1)
            total_penalty += penalty
        
        # Normalizar por cantidad de datos
        normalized_penalty = total_penalty / max(data_points, 1)
        
        # Score final (mínimo 0.0)
        quality_score = max(0.0, 1.0 - normalized_penalty)
        
        return quality_score
    
    def _determine_validity(self, issues: List[ValidationIssue]) -> bool:
        """
        Determina si los datos son válidos basado en la severidad de los problemas
        """
        severity_order = {
            ValidationSeverity.INFO: 0,
            ValidationSeverity.WARNING: 1,
            ValidationSeverity.ERROR: 2,
            ValidationSeverity.CRITICAL: 3
        }
        
        min_severity_level = severity_order[self.config.min_severity_for_rejection]
        
        for issue in issues:
            issue_severity_level = severity_order[issue.severity]
            if issue_severity_level >= min_severity_level:
                return False
        return True
    
    def _create_validation_summary(self, issues: List[ValidationIssue], 
                                  data: pd.DataFrame, symbol: str, 
                                  timeframe: str) -> Dict[str, Any]:
        """Crea un resumen detallado de la validación"""
        
        # Contar issues por severidad
        severity_counts = {sev.value: 0 for sev in ValidationSeverity}
        rule_counts = {rule.value: 0 for rule in ValidationRule}
        
        for issue in issues:
            severity_counts[issue.severity.value] += 1
            rule_counts[issue.rule.value] += 1
        
        # Estadísticas básicas de los datos
        data_stats = {
            "total_candles": len(data),
            "date_range": {
                "start": data.index[0].isoformat() if len(data) > 0 and isinstance(data.index[0], datetime) else None,
                "end": data.index[-1].isoformat() if len(data) > 0 and isinstance(data.index[-1], datetime) else None
            },
            "price_range": {
                "min": float(data[['open', 'high', 'low', 'close']].min().min()),
                "max": float(data[['open', 'high', 'low', 'close']].max().max())
            },
            "volume_stats": {
                "total": float(data['volume'].sum()),
                "average": float(data['volume'].mean()),
                "max": float(data['volume'].max())
            }
        }
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "validation_timestamp": datetime.now(ECUADOR_TZ).isoformat(),
            "total_issues": len(issues),
            "issues_by_severity": severity_counts,
            "issues_by_rule": rule_counts,
            "data_statistics": data_stats,
            "validation_config": {
                "enabled_rules": [rule.value for rule in self.config.enabled_rules],
                "min_severity_for_rejection": self.config.min_severity_for_rejection.value
            }
        }
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas globales del validador"""
        total_validations = self.stats["total_validations"]
        success_rate = (self.stats["passed_validations"] / total_validations 
                       if total_validations > 0 else 0.0)
        
        return {
            **self.stats,
            "success_rate": success_rate,
            "most_common_issues": {
                rule: count for rule, count in self.stats["issues_by_rule"].items()
                if count > 0
            }
        }
    
    def create_validation_report(self, result: ValidationResult) -> str:
        """
        Crea un reporte legible de la validación
        Útil para debugging y análisis manual
        """
        report = []
        report.append("=" * 60)
        report.append("REPORTE DE VALIDACIÓN DE DATOS")
        report.append("=" * 60)
        
        summary = result.summary
        report.append(f"Símbolo: {summary['symbol']}")
        report.append(f"Timeframe: {summary['timeframe']}")
        report.append(f"Timestamp: {summary['validation_timestamp']}")
        report.append(f"Estado: {'✅ VÁLIDO' if result.is_valid else '❌ INVÁLIDO'}")
        report.append(f"Score de Calidad: {result.quality_score:.3f}")
        report.append(f"Total de Issues: {len(result.issues)}")
        report.append("")
        
        # Estadísticas de datos
        stats = summary["data_statistics"]
        report.append("ESTADÍSTICAS DE DATOS:")
        report.append(f"  Total de velas: {stats['total_candles']}")
        report.append(f"  Rango de fechas: {stats['date_range']['start']} a {stats['date_range']['end']}")
        report.append(f"  Rango de precios: ${stats['price_range']['min']:.2f} - ${stats['price_range']['max']:.2f}")
        report.append(f"  Volumen total: {stats['volume_stats']['total']:,.0f}")
        report.append("")
        
        # Issues por severidad
        if result.issues:
            report.append("ISSUES ENCONTRADOS:")
            for severity in ValidationSeverity:
                count = summary["issues_by_severity"][severity.value]
                if count > 0:
                    report.append(f"  {severity.value.upper()}: {count}")
            report.append("")
            
            # Detalles de issues críticos y errores
            critical_issues = [i for i in result.issues 
                             if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]]
            
            if critical_issues:
                report.append("ISSUES CRÍTICOS/ERRORES:")
                for issue in critical_issues[:10]:  # Mostrar máximo 10
                    report.append(f"  • {issue.message}")
                    if issue.suggestion:
                        report.append(f"    Sugerencia: {issue.suggestion}")
                report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)

# Instancia global del validador
validator = DataValidator()

# Funciones de conveniencia
def validate_data(data: pd.DataFrame, symbol: str, timeframe: str,
                 config: Optional[ValidationConfig] = None) -> ValidationResult:
    """Función de conveniencia para validar datos"""
    if config:
        temp_validator = DataValidator(config)
        return temp_validator.validate_ohlcv_data(data, symbol, timeframe)
    else:
        return validator.validate_ohlcv_data(data, symbol, timeframe)

def quick_validate(data: pd.DataFrame) -> bool:
    """Validación rápida - solo retorna True/False"""
    result = validator.validate_ohlcv_data(data, "UNKNOWN", "UNKNOWN")
    return result.is_valid