"""
Sistema de Gestión de Costos LLM
===============================

Como trader senior, el control de costos es crítico. Este sistema asegura que
nunca excedamos el presupuesto diario de $50 mientras maximizamos el valor
obtenido de cada llamada al LLM.

Características:
- Tracking preciso de costos por proveedor
- Presupuesto diario con alertas
- Optimización automática de uso
- Reportes detallados de ROI
- Predicción de costos futuros

El objetivo: Máximo valor analítico por dólar invertido.

Autor: Trader Algorítmico Senior (10+ años)
Zona Horaria: UTC-5 (Ecuador)
"""

import json
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pytz
from pathlib import Path

from .connector import LLMProvider, LLMResponse
from ..utils.logger import get_logger, LogContext
from ..utils.exceptions import TradingBotException

# Zona horaria de Ecuador (UTC-5)
ECUADOR_TZ = pytz.timezone('America/Guayaquil')

class BudgetExceededError(TradingBotException):
    """Error cuando se excede el presupuesto"""
    pass

class CostCategory(Enum):
    """Categorías de costos LLM"""
    TRADE_ANALYSIS = "trade_analysis"
    DIVERGENCE_CHECK = "divergence_check"
    MACRO_CONTEXT = "macro_context"
    RISK_ASSESSMENT = "risk_assessment"
    PATTERN_VALIDATION = "pattern_validation"
    OTHER = "other"

@dataclass
class CostEntry:
    """Entrada individual de costo"""
    timestamp: datetime
    provider: LLMProvider
    model: str
    category: CostCategory
    tokens_used: int
    cost_usd: float
    response_time_ms: int
    cached: bool = False
    
    # Métricas de valor
    trade_approved: Optional[bool] = None
    trade_pnl: Optional[float] = None  # PnL del trade si se ejecutó
    value_score: Optional[float] = None  # Score de valor (0-1)

@dataclass
class DailyCostSummary:
    """Resumen de costos diarios"""
    date: date
    total_cost: float
    total_requests: int
    total_tokens: int
    
    # Por proveedor
    cost_by_provider: Dict[LLMProvider, float] = field(default_factory=dict)
    requests_by_provider: Dict[LLMProvider, int] = field(default_factory=dict)
    
    # Por categoría
    cost_by_category: Dict[CostCategory, float] = field(default_factory=dict)
    requests_by_category: Dict[CostCategory, int] = field(default_factory=dict)
    
    # Métricas de valor
    trades_approved: int = 0
    trades_rejected: int = 0
    total_trade_pnl: float = 0.0
    roi_percentage: float = 0.0  # ROI = (PnL - Costo) / Costo * 100

class CostTracker:
    """
    Tracker de costos con análisis de ROI
    
    Como trader senior, mido el ROI de cada dólar invertido en LLM.
    Si el LLM no genera valor, lo desactivo.
    """
    
    def __init__(self, data_directory: str = "data/llm_costs"):
        self.logger = get_logger("CostTracker")
        self.data_dir = Path(data_directory)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Almacenamiento de costos
        self.cost_entries: List[CostEntry] = []
        self.daily_summaries: Dict[date, DailyCostSummary] = {}
        
        # Cargar datos existentes
        self._load_cost_data()
    
    def record_cost(self, response: LLMResponse, category: CostCategory) -> None:
        """
        Registra el costo de una respuesta LLM
        
        Args:
            response: Respuesta del LLM
            category: Categoría del análisis
        """
        entry = CostEntry(
            timestamp=response.timestamp,
            provider=response.provider,
            model=response.model,
            category=category,
            tokens_used=response.tokens_used,
            cost_usd=response.cost_usd,
            response_time_ms=response.response_time_ms,
            cached=response.cached
        )
        
        self.cost_entries.append(entry)
        self._update_daily_summary(entry)
        
        self.logger.debug(
            f"Costo registrado: {category.value}",
            extra_fields={
                "provider": response.provider.value,
                "cost_usd": response.cost_usd,
                "tokens": response.tokens_used,
                "cached": response.cached
            }
        )
    
    def update_trade_outcome(self, response_hash: str, approved: bool, 
                           pnl: Optional[float] = None) -> None:
        """
        Actualiza el resultado de un trade para calcular ROI
        
        Args:
            response_hash: Hash de la respuesta LLM
            approved: Si el trade fue aprobado
            pnl: PnL del trade si se ejecutó
        """
        # Buscar la entrada correspondiente
        for entry in reversed(self.cost_entries):  # Buscar desde la más reciente
            if hasattr(entry, 'response_hash') and entry.response_hash == response_hash:
                entry.trade_approved = approved
                entry.trade_pnl = pnl or 0.0
                
                # Calcular score de valor
                if approved and pnl is not None:
                    # Valor positivo si el PnL supera el costo
                    entry.value_score = min(1.0, max(0.0, (pnl - entry.cost_usd) / max(entry.cost_usd, 0.01)))
                else:
                    # Valor neutral si se rechazó (potencialmente evitó pérdidas)
                    entry.value_score = 0.5
                
                # Actualizar resumen diario
                daily_summary = self.daily_summaries.get(entry.timestamp.date())
                if daily_summary:
                    if approved:
                        daily_summary.trades_approved += 1
                    else:
                        daily_summary.trades_rejected += 1
                    
                    if pnl is not None:
                        daily_summary.total_trade_pnl += pnl
                        # Recalcular ROI
                        if daily_summary.total_cost > 0:
                            daily_summary.roi_percentage = (
                                (daily_summary.total_trade_pnl - daily_summary.total_cost) / 
                                daily_summary.total_cost * 100
                            )
                
                break
    
    def _update_daily_summary(self, entry: CostEntry) -> None:
        """Actualiza el resumen diario con una nueva entrada"""
        entry_date = entry.timestamp.date()
        
        if entry_date not in self.daily_summaries:
            self.daily_summaries[entry_date] = DailyCostSummary(
                date=entry_date,
                total_cost=0.0,
                total_requests=0,
                total_tokens=0
            )
        
        summary = self.daily_summaries[entry_date]
        
        # Actualizar totales
        summary.total_cost += entry.cost_usd
        summary.total_requests += 1
        summary.total_tokens += entry.tokens_used
        
        # Actualizar por proveedor
        if entry.provider not in summary.cost_by_provider:
            summary.cost_by_provider[entry.provider] = 0.0
            summary.requests_by_provider[entry.provider] = 0
        
        summary.cost_by_provider[entry.provider] += entry.cost_usd
        summary.requests_by_provider[entry.provider] += 1
        
        # Actualizar por categoría
        if entry.category not in summary.cost_by_category:
            summary.cost_by_category[entry.category] = 0.0
            summary.requests_by_category[entry.category] = 0
        
        summary.cost_by_category[entry.category] += entry.cost_usd
        summary.requests_by_category[entry.category] += 1
    
    def get_daily_cost(self, target_date: Optional[date] = None) -> float:
        """Obtiene el costo total para un día específico"""
        if target_date is None:
            target_date = datetime.now(ECUADOR_TZ).date()
        
        summary = self.daily_summaries.get(target_date)
        return summary.total_cost if summary else 0.0
    
    def get_daily_summary(self, target_date: Optional[date] = None) -> Optional[DailyCostSummary]:
        """Obtiene el resumen completo para un día específico"""
        if target_date is None:
            target_date = datetime.now(ECUADOR_TZ).date()
        
        return self.daily_summaries.get(target_date)
    
    def get_weekly_summary(self, weeks_back: int = 1) -> Dict[str, Any]:
        """Obtiene resumen semanal"""
        end_date = datetime.now(ECUADOR_TZ).date()
        start_date = end_date - timedelta(days=7 * weeks_back)
        
        total_cost = 0.0
        total_requests = 0
        total_pnl = 0.0
        provider_costs = {}
        category_costs = {}
        
        for current_date in [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]:
            summary = self.daily_summaries.get(current_date)
            if summary:
                total_cost += summary.total_cost
                total_requests += summary.total_requests
                total_pnl += summary.total_trade_pnl
                
                # Agregar por proveedor
                for provider, cost in summary.cost_by_provider.items():
                    provider_costs[provider.value] = provider_costs.get(provider.value, 0.0) + cost
                
                # Agregar por categoría
                for category, cost in summary.cost_by_category.items():
                    category_costs[category.value] = category_costs.get(category.value, 0.0) + cost
        
        roi = ((total_pnl - total_cost) / total_cost * 100) if total_cost > 0 else 0.0
        
        return {
            "period": f"{start_date} to {end_date}",
            "total_cost": total_cost,
            "total_requests": total_requests,
            "total_pnl": total_pnl,
            "roi_percentage": roi,
            "cost_by_provider": provider_costs,
            "cost_by_category": category_costs,
            "avg_cost_per_request": total_cost / total_requests if total_requests > 0 else 0.0
        }
    
    def _load_cost_data(self) -> None:
        """Carga datos de costos desde disco"""
        try:
            # Cargar entradas de costo
            entries_file = self.data_dir / "cost_entries.json"
            if entries_file.exists():
                with open(entries_file, 'r') as f:
                    entries_data = json.load(f)
                
                for entry_data in entries_data:
                    try:
                        entry = CostEntry(
                            timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                            provider=LLMProvider(entry_data["provider"]),
                            model=entry_data["model"],
                            category=CostCategory(entry_data["category"]),
                            tokens_used=entry_data["tokens_used"],
                            cost_usd=entry_data["cost_usd"],
                            response_time_ms=entry_data["response_time_ms"],
                            cached=entry_data.get("cached", False),
                            trade_approved=entry_data.get("trade_approved"),
                            trade_pnl=entry_data.get("trade_pnl"),
                            value_score=entry_data.get("value_score")
                        )
                        self.cost_entries.append(entry)
                    except Exception as e:
                        self.logger.debug(f"Error cargando entrada de costo: {e}")
            
            # Reconstruir resúmenes diarios
            self._rebuild_daily_summaries()
            
            self.logger.info(f"Datos de costo cargados: {len(self.cost_entries)} entradas")
            
        except Exception as e:
            self.logger.warning(f"Error cargando datos de costo: {e}")
    
    def _rebuild_daily_summaries(self) -> None:
        """Reconstruye resúmenes diarios desde las entradas"""
        self.daily_summaries.clear()
        
        for entry in self.cost_entries:
            self._update_daily_summary(entry)
    
    def save_cost_data(self) -> None:
        """Guarda datos de costos a disco"""
        try:
            # Asegurar que el directorio existe
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Guardar entradas de costo
            entries_file = self.data_dir / "cost_entries.json"
            entries_data = []
            
            for entry in self.cost_entries:
                entry_dict = {
                    "timestamp": entry.timestamp.isoformat(),
                    "provider": entry.provider.value,
                    "model": entry.model,
                    "category": entry.category.value,
                    "tokens_used": entry.tokens_used,
                    "cost_usd": entry.cost_usd,
                    "response_time_ms": entry.response_time_ms,
                    "cached": entry.cached,
                    "trade_approved": entry.trade_approved,
                    "trade_pnl": entry.trade_pnl,
                    "value_score": entry.value_score
                }
                entries_data.append(entry_dict)
            
            with open(entries_file, 'w') as f:
                json.dump(entries_data, f, indent=2)
            
            self.logger.debug(f"Datos de costo guardados: {len(entries_data)} entradas")
            
        except Exception as e:
            self.logger.error(f"Error guardando datos de costo: {e}")

class CostManager:
    """
    Gestor principal de costos con control de presupuesto
    
    Como trader senior, este es mi guardian de costos. Nunca permite
    que el LLM exceda el presupuesto diario de $50.
    """
    
    def __init__(self, max_daily_cost: float = 50.0, 
                 data_directory: str = "data/llm_costs"):
        self.logger = get_logger("CostManager")
        self.max_daily_cost = max_daily_cost
        self.cost_tracker = CostTracker(data_directory)
        
        # Alertas de presupuesto
        self.alert_thresholds = [0.5, 0.75, 0.9, 0.95]  # 50%, 75%, 90%, 95%
        self.alerts_sent_today = set()
    
    def check_budget_available(self, estimated_cost: float = 0.0) -> Tuple[bool, str]:
        """
        Verifica si hay presupuesto disponible
        
        Args:
            estimated_cost: Costo estimado de la próxima operación
            
        Returns:
            Tuple de (disponible, razón)
        """
        today_cost = self.cost_tracker.get_daily_cost()
        remaining_budget = self.max_daily_cost - today_cost
        
        if today_cost >= self.max_daily_cost:
            return False, f"Presupuesto diario agotado: ${today_cost:.2f}/${self.max_daily_cost:.2f}"
        
        if estimated_cost > remaining_budget:
            return False, f"Costo estimado (${estimated_cost:.2f}) excede presupuesto restante (${remaining_budget:.2f})"
        
        # Verificar alertas de presupuesto
        usage_percentage = today_cost / self.max_daily_cost
        for threshold in self.alert_thresholds:
            if usage_percentage >= threshold and threshold not in self.alerts_sent_today:
                self.logger.warning(
                    f"Alerta de presupuesto: {usage_percentage:.1%} del presupuesto diario usado",
                    extra_fields={
                        "daily_cost": today_cost,
                        "max_daily_cost": self.max_daily_cost,
                        "remaining": remaining_budget
                    }
                )
                self.alerts_sent_today.add(threshold)
        
        return True, f"Presupuesto disponible: ${remaining_budget:.2f}"
    
    def record_llm_usage(self, response: LLMResponse, category: CostCategory) -> None:
        """
        Registra el uso del LLM
        
        Args:
            response: Respuesta del LLM
            category: Categoría del análisis
        """
        self.cost_tracker.record_cost(response, category)
        
        # Verificar si excedimos el presupuesto
        today_cost = self.cost_tracker.get_daily_cost()
        if today_cost > self.max_daily_cost:
            self.logger.error(
                f"¡PRESUPUESTO EXCEDIDO! Costo diario: ${today_cost:.2f}",
                extra_fields={
                    "max_daily_cost": self.max_daily_cost,
                    "overage": today_cost - self.max_daily_cost
                }
            )
    
    def update_trade_outcome(self, response_hash: str, approved: bool, 
                           pnl: Optional[float] = None) -> None:
        """Actualiza el resultado de un trade para análisis de ROI"""
        self.cost_tracker.update_trade_outcome(response_hash, approved, pnl)
    
    def get_cost_efficiency_report(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Genera reporte de eficiencia de costos
        
        Args:
            days_back: Días hacia atrás para el análisis
            
        Returns:
            Reporte detallado de eficiencia
        """
        end_date = datetime.now(ECUADOR_TZ).date()
        start_date = end_date - timedelta(days=days_back)
        
        # Recopilar datos del período
        total_cost = 0.0
        total_requests = 0
        total_pnl = 0.0
        approved_trades = 0
        rejected_trades = 0
        
        provider_efficiency = {}
        category_efficiency = {}
        
        for entry in self.cost_tracker.cost_entries:
            if start_date <= entry.timestamp.date() <= end_date:
                total_cost += entry.cost_usd
                total_requests += 1
                
                if entry.trade_pnl is not None:
                    total_pnl += entry.trade_pnl
                
                if entry.trade_approved is True:
                    approved_trades += 1
                elif entry.trade_approved is False:
                    rejected_trades += 1
                
                # Eficiencia por proveedor
                provider = entry.provider.value
                if provider not in provider_efficiency:
                    provider_efficiency[provider] = {"cost": 0.0, "pnl": 0.0, "requests": 0}
                
                provider_efficiency[provider]["cost"] += entry.cost_usd
                provider_efficiency[provider]["pnl"] += entry.trade_pnl or 0.0
                provider_efficiency[provider]["requests"] += 1
                
                # Eficiencia por categoría
                category = entry.category.value
                if category not in category_efficiency:
                    category_efficiency[category] = {"cost": 0.0, "pnl": 0.0, "requests": 0}
                
                category_efficiency[category]["cost"] += entry.cost_usd
                category_efficiency[category]["pnl"] += entry.trade_pnl or 0.0
                category_efficiency[category]["requests"] += 1
        
        # Calcular ROI por proveedor
        for provider_data in provider_efficiency.values():
            if provider_data["cost"] > 0:
                provider_data["roi"] = (provider_data["pnl"] - provider_data["cost"]) / provider_data["cost"] * 100
            else:
                provider_data["roi"] = 0.0
        
        # Calcular ROI por categoría
        for category_data in category_efficiency.values():
            if category_data["cost"] > 0:
                category_data["roi"] = (category_data["pnl"] - category_data["cost"]) / category_data["cost"] * 100
            else:
                category_data["roi"] = 0.0
        
        overall_roi = ((total_pnl - total_cost) / total_cost * 100) if total_cost > 0 else 0.0
        
        return {
            "period": f"{start_date} to {end_date}",
            "summary": {
                "total_cost": total_cost,
                "total_requests": total_requests,
                "total_pnl": total_pnl,
                "overall_roi": overall_roi,
                "approved_trades": approved_trades,
                "rejected_trades": rejected_trades,
                "approval_rate": approved_trades / (approved_trades + rejected_trades) if (approved_trades + rejected_trades) > 0 else 0.0
            },
            "provider_efficiency": provider_efficiency,
            "category_efficiency": category_efficiency,
            "recommendations": self._generate_efficiency_recommendations(provider_efficiency, category_efficiency)
        }
    
    def _generate_efficiency_recommendations(self, provider_eff: Dict, category_eff: Dict) -> List[str]:
        """Genera recomendaciones basadas en eficiencia"""
        recommendations = []
        
        # Recomendar mejor proveedor
        if provider_eff:
            best_provider = max(provider_eff.items(), key=lambda x: x[1]["roi"])
            if best_provider[1]["roi"] > 0:
                recommendations.append(f"Proveedor más eficiente: {best_provider[0]} (ROI: {best_provider[1]['roi']:.1f}%)")
        
        # Recomendar mejor categoría
        if category_eff:
            best_category = max(category_eff.items(), key=lambda x: x[1]["roi"])
            if best_category[1]["roi"] > 0:
                recommendations.append(f"Categoría más valiosa: {best_category[0]} (ROI: {best_category[1]['roi']:.1f}%)")
        
        # Identificar categorías ineficientes
        inefficient_categories = [cat for cat, data in category_eff.items() if data["roi"] < -10]
        if inefficient_categories:
            recommendations.append(f"Considerar reducir uso en: {', '.join(inefficient_categories)}")
        
        return recommendations
    
    def __del__(self):
        """Destructor - guarda datos al finalizar"""
        try:
            self.cost_tracker.save_cost_data()
        except:
            pass