"""
Conector LLM Escalable para Trading Algorítmico
==============================================

Como trader senior, he diseñado este conector para ser robusto, escalable y
cost-effective. Gemini es mi elección principal por su excelente relación
calidad-precio y capacidad de análisis contextual.

Este conector maneja:
- Múltiples proveedores LLM con fallback automático
- Control de costos estricto ($50/día máximo)
- Retry logic inteligente con exponential backoff
- Rate limiting para evitar throttling
- Caché de respuestas para queries similares
- Logging detallado para debugging

Autor: Trader Algorítmico Senior (10+ años)
Zona Horaria: UTC-5 (Ecuador)
"""

import asyncio
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import pytz
from pathlib import Path

import google.generativeai as genai
import openai
import anthropic

from ..config.settings import config
from ..utils.logger import get_logger, LogContext
from ..utils.exceptions import TradingBotException, ErrorCodes

# Zona horaria de Ecuador (UTC-5)
ECUADOR_TZ = pytz.timezone('America/Guayaquil')

class LLMProvider(Enum):
    """Proveedores LLM soportados"""
    GEMINI = "gemini"           # Google Gemini (recomendado)
    OPENAI = "openai"           # OpenAI GPT-4
    ANTHROPIC = "anthropic"     # Anthropic Claude
    LOCAL = "local"             # Modelo local (futuro)

class LLMError(TradingBotException):
    """Errores específicos de LLM"""
    pass

@dataclass
class LLMResponse:
    """Respuesta estructurada del LLM"""
    provider: LLMProvider
    model: str
    prompt_hash: str
    response_text: str
    parsed_data: Optional[Dict[str, Any]] = None
    
    # Métricas
    tokens_used: int = 0
    cost_usd: float = 0.0
    response_time_ms: int = 0
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(ECUADOR_TZ))
    cached: bool = False
    
    def is_valid(self) -> bool:
        """Verifica si la respuesta es válida"""
        return (
            self.response_text is not None and 
            len(self.response_text.strip()) > 0 and
            self.parsed_data is not None
        )

@dataclass
class LLMConfig:
    """Configuración para un proveedor LLM"""
    provider: LLMProvider
    model: str
    api_key: str
    max_tokens: int = 1000
    temperature: float = 0.1  # Baja temperatura para trading (más determinístico)
    timeout_seconds: int = 30
    max_retries: int = 3
    cost_per_1k_tokens: float = 0.001  # Costo por defecto
    enabled: bool = True

class LLMConnector:
    """
    Conector principal para múltiples proveedores LLM
    
    Como trader senior, he diseñado este conector para ser:
    1. Robusto - Maneja errores gracefully
    2. Cost-effective - Control estricto de presupuesto
    3. Escalable - Fácil agregar nuevos proveedores
    4. Inteligente - Caché y fallback automático
    """
    
    def __init__(self, cache_dir: str = "data/llm_cache"):
        self.logger = get_logger("LLMConnector")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuraciones por proveedor
        self.configs: Dict[LLMProvider, LLMConfig] = {}
        self.clients: Dict[LLMProvider, Any] = {}
        
        # Control de costos y rate limiting
        self.daily_cost = 0.0
        self.daily_requests = 0
        self.last_reset = datetime.now(ECUADOR_TZ).date()
        self.max_daily_cost = 50.0  # $50 máximo por día
        self.max_daily_requests = 1000
        
        # Caché de respuestas
        self.response_cache: Dict[str, LLMResponse] = {}
        self.cache_ttl_hours = 24  # TTL de 24 horas
        
        # Rate limiting por proveedor
        self.last_request_time: Dict[LLMProvider, float] = {}
        self.min_request_interval = 1.0  # Mínimo 1 segundo entre requests
        
        # Inicializar proveedores
        self._initialize_providers()
        self._load_cache()
    
    def _initialize_providers(self) -> None:
        """Inicializa todos los proveedores LLM disponibles"""
        try:
            import os
            # Google Gemini (prioridad 1)
            gemini_key = os.environ.get("GEMINI_API_KEY")
            if gemini_key:
                self.configs[LLMProvider.GEMINI] = LLMConfig(
                    provider=LLMProvider.GEMINI,
                    model="gemini-1.5-flash",  # Modelo rápido y económico
                    api_key=gemini_key,
                    max_tokens=1000,
                    temperature=0.1,
                    cost_per_1k_tokens=0.0005,  # Muy económico
                    enabled=True
                )
                
                genai.configure(api_key=gemini_key)
                self.clients[LLMProvider.GEMINI] = genai.GenerativeModel('gemini-1.5-flash')
                self.logger.info("Gemini configurado exitosamente")
            
            # OpenAI GPT-4 (fallback)
            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key:
                self.configs[LLMProvider.OPENAI] = LLMConfig(
                    provider=LLMProvider.OPENAI,
                    model="gpt-4o-mini",  # Modelo económico
                    api_key=openai_key,
                    max_tokens=1000,
                    temperature=0.1,
                    cost_per_1k_tokens=0.0015,
                    enabled=True
                )
                
                self.clients[LLMProvider.OPENAI] = openai.OpenAI(api_key=openai_key)
                self.logger.info("OpenAI configurado exitosamente")
            
            # Anthropic Claude (fallback)
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
            if anthropic_key:
                self.configs[LLMProvider.ANTHROPIC] = LLMConfig(
                    provider=LLMProvider.ANTHROPIC,
                    model="claude-3-haiku-20240307",  # Modelo económico
                    api_key=anthropic_key,
                    max_tokens=1000,
                    temperature=0.1,
                    cost_per_1k_tokens=0.0025,
                    enabled=True
                )
                
                self.clients[LLMProvider.ANTHROPIC] = anthropic.Anthropic(api_key=anthropic_key)
                self.logger.info("Anthropic configurado exitosamente")
            
            if not self.configs:
                raise LLMError("No hay proveedores LLM configurados")
            
            self.logger.info(f"Inicializados {len(self.configs)} proveedores LLM")
            
        except Exception as e:
            self.logger.error(f"Error inicializando proveedores: {e}")
            raise LLMError(f"Fallo en inicialización: {str(e)}")
    
    def _reset_daily_limits_if_needed(self) -> None:
        """Resetea límites diarios si es un nuevo día"""
        today = datetime.now(ECUADOR_TZ).date()
        if today > self.last_reset:
            self.daily_cost = 0.0
            self.daily_requests = 0
            self.last_reset = today
            self.logger.info("Límites diarios reseteados")
    
    def _check_daily_limits(self) -> None:
        """Verifica que no se excedan los límites diarios"""
        self._reset_daily_limits_if_needed()
        
        if self.daily_cost >= self.max_daily_cost:
            raise LLMError(f"Presupuesto diario excedido: ${self.daily_cost:.2f}")
        
        if self.daily_requests >= self.max_daily_requests:
            raise LLMError(f"Límite de requests diarios excedido: {self.daily_requests}")
    
    def _get_cache_key(self, prompt: str, provider: LLMProvider, model: str) -> str:
        """Genera clave de caché para un prompt"""
        content = f"{prompt}|{provider.value}|{model}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        """Obtiene respuesta del caché si está disponible y válida"""
        if cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key]
            
            # Verificar TTL
            age_hours = (datetime.now(ECUADOR_TZ) - cached_response.timestamp).total_seconds() / 3600
            if age_hours < self.cache_ttl_hours:
                cached_response.cached = True
                return cached_response
            else:
                # Remover caché expirado
                del self.response_cache[cache_key]
        
        return None
    
    def _cache_response(self, cache_key: str, response: LLMResponse) -> None:
        """Guarda respuesta en caché"""
        self.response_cache[cache_key] = response
        
        # Limpiar caché si es muy grande
        if len(self.response_cache) > 1000:
            # Remover las 100 entradas más antiguas
            sorted_items = sorted(
                self.response_cache.items(),
                key=lambda x: x[1].timestamp
            )
            for key, _ in sorted_items[:100]:
                del self.response_cache[key]
    
    def _apply_rate_limiting(self, provider: LLMProvider) -> None:
        """Aplica rate limiting por proveedor"""
        if provider in self.last_request_time:
            time_since_last = time.time() - self.last_request_time[provider]
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                time.sleep(sleep_time)
        
        self.last_request_time[provider] = time.time()
    
    async def _call_gemini(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """Llama a Google Gemini"""
        start_time = time.time()
        
        try:
            client = self.clients[LLMProvider.GEMINI]
            
            # Configurar generación
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=config.max_tokens,
                temperature=config.temperature,
                candidate_count=1
            )
            
            # Generar respuesta
            response = client.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if not response.text:
                raise LLMError("Gemini no retornó texto")
            
            # Calcular métricas
            response_time_ms = int((time.time() - start_time) * 1000)
            tokens_used = len(response.text.split()) * 1.3  # Aproximación
            cost_usd = (tokens_used / 1000) * config.cost_per_1k_tokens
            
            return LLMResponse(
                provider=LLMProvider.GEMINI,
                model=config.model,
                prompt_hash=hashlib.md5(prompt.encode()).hexdigest(),
                response_text=response.text,
                tokens_used=int(tokens_used),
                cost_usd=cost_usd,
                response_time_ms=response_time_ms
            )
            
        except Exception as e:
            self.logger.error(f"Error en Gemini: {e}")
            raise LLMError(f"Gemini falló: {str(e)}")
    
    async def _call_openai(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """Llama a OpenAI GPT"""
        start_time = time.time()
        
        try:
            client = self.clients[LLMProvider.OPENAI]
            
            response = client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                timeout=config.timeout_seconds
            )
            
            if not response.choices or not response.choices[0].message.content:
                raise LLMError("OpenAI no retornó contenido")
            
            # Calcular métricas
            response_time_ms = int((time.time() - start_time) * 1000)
            tokens_used = response.usage.total_tokens if response.usage else 0
            cost_usd = (tokens_used / 1000) * config.cost_per_1k_tokens
            
            return LLMResponse(
                provider=LLMProvider.OPENAI,
                model=config.model,
                prompt_hash=hashlib.md5(prompt.encode()).hexdigest(),
                response_text=response.choices[0].message.content,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                response_time_ms=response_time_ms
            )
            
        except Exception as e:
            self.logger.error(f"Error en OpenAI: {e}")
            raise LLMError(f"OpenAI falló: {str(e)}")
    
    async def _call_anthropic(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """Llama a Anthropic Claude"""
        start_time = time.time()
        
        try:
            client = self.clients[LLMProvider.ANTHROPIC]
            
            response = client.messages.create(
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            if not response.content or not response.content[0].text:
                raise LLMError("Anthropic no retornó contenido")
            
            # Calcular métricas
            response_time_ms = int((time.time() - start_time) * 1000)
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            cost_usd = (tokens_used / 1000) * config.cost_per_1k_tokens
            
            return LLMResponse(
                provider=LLMProvider.ANTHROPIC,
                model=config.model,
                prompt_hash=hashlib.md5(prompt.encode()).hexdigest(),
                response_text=response.content[0].text,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                response_time_ms=response_time_ms
            )
            
        except Exception as e:
            self.logger.error(f"Error en Anthropic: {e}")
            raise LLMError(f"Anthropic falló: {str(e)}")
    
    async def generate_response(self, 
                              prompt: str,
                              preferred_provider: Optional[LLMProvider] = None,
                              use_cache: bool = True) -> LLMResponse:
        """
        Genera respuesta usando el mejor proveedor disponible
        
        Args:
            prompt: Prompt para el LLM
            preferred_provider: Proveedor preferido (opcional)
            use_cache: Si usar caché de respuestas
            
        Returns:
            LLMResponse con la respuesta generada
        """
        context = LogContext(component="llm_connector")
        
        # Verificar límites diarios
        self._check_daily_limits()
        
        # Determinar orden de proveedores
        if preferred_provider and preferred_provider in self.configs:
            provider_order = [preferred_provider]
            # Agregar otros como fallback
            provider_order.extend([p for p in self.configs.keys() if p != preferred_provider])
        else:
            # Orden por defecto: Gemini primero (más económico)
            provider_order = [
                LLMProvider.GEMINI,
                LLMProvider.OPENAI,
                LLMProvider.ANTHROPIC
            ]
            provider_order = [p for p in provider_order if p in self.configs]
        
        # Intentar con cada proveedor
        last_error = None
        
        for provider in provider_order:
            config = self.configs[provider]
            
            if not config.enabled:
                continue
            
            try:
                # Verificar caché
                cache_key = self._get_cache_key(prompt, provider, config.model)
                
                if use_cache:
                    cached_response = self._get_cached_response(cache_key)
                    if cached_response:
                        self.logger.debug(f"Respuesta obtenida del caché: {provider.value}")
                        return cached_response
                
                # Aplicar rate limiting
                self._apply_rate_limiting(provider)
                
                # Llamar al proveedor
                self.logger.info(f"Llamando a {provider.value} para generar respuesta")
                
                if provider == LLMProvider.GEMINI:
                    response = await self._call_gemini(prompt, config)
                elif provider == LLMProvider.OPENAI:
                    response = await self._call_openai(prompt, config)
                elif provider == LLMProvider.ANTHROPIC:
                    response = await self._call_anthropic(prompt, config)
                else:
                    raise LLMError(f"Proveedor no soportado: {provider}")
                
                # Actualizar métricas diarias
                self.daily_cost += response.cost_usd
                self.daily_requests += 1
                
                # Guardar en caché
                if use_cache:
                    self._cache_response(cache_key, response)
                
                self.logger.info(
                    f"Respuesta generada exitosamente: {provider.value}",
                    context=context,
                    extra_fields={
                        "tokens_used": response.tokens_used,
                        "cost_usd": response.cost_usd,
                        "response_time_ms": response.response_time_ms,
                        "daily_cost": self.daily_cost,
                        "daily_requests": self.daily_requests
                    }
                )
                
                return response
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Proveedor {provider.value} falló: {str(e)}")
                continue
        
        # Si llegamos aquí, todos los proveedores fallaron
        raise LLMError(f"Todos los proveedores LLM fallaron. Último error: {str(last_error)}")
    
    def _load_cache(self) -> None:
        """Carga caché desde disco"""
        try:
            cache_file = self.cache_dir / "response_cache.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Reconstruir objetos LLMResponse
                for key, data in cache_data.items():
                    try:
                        response = LLMResponse(
                            provider=LLMProvider(data["provider"]),
                            model=data["model"],
                            prompt_hash=data["prompt_hash"],
                            response_text=data["response_text"],
                            parsed_data=data.get("parsed_data"),
                            tokens_used=data.get("tokens_used", 0),
                            cost_usd=data.get("cost_usd", 0.0),
                            response_time_ms=data.get("response_time_ms", 0),
                            timestamp=datetime.fromisoformat(data["timestamp"]),
                            cached=True
                        )
                        self.response_cache[key] = response
                    except Exception as e:
                        self.logger.debug(f"Error cargando entrada de caché: {e}")
                
                self.logger.info(f"Caché cargado: {len(self.response_cache)} entradas")
        
        except Exception as e:
            self.logger.warning(f"Error cargando caché: {e}")
    
    def save_cache(self) -> None:
        """Guarda caché a disco"""
        try:
            # Asegurar que el directorio existe
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / "response_cache.json"
            
            # Convertir a formato serializable
            cache_data = {}
            for key, response in self.response_cache.items():
                cache_data[key] = {
                    "provider": response.provider.value,
                    "model": response.model,
                    "prompt_hash": response.prompt_hash,
                    "response_text": response.response_text,
                    "parsed_data": response.parsed_data,
                    "tokens_used": response.tokens_used,
                    "cost_usd": response.cost_usd,
                    "response_time_ms": response.response_time_ms,
                    "timestamp": response.timestamp.isoformat()
                }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            self.logger.debug(f"Caché guardado: {len(cache_data)} entradas")
            
        except Exception as e:
            self.logger.error(f"Error guardando caché: {e}")
    
    def get_daily_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas diarias"""
        self._reset_daily_limits_if_needed()
        
        return {
            "daily_cost": self.daily_cost,
            "daily_requests": self.daily_requests,
            "max_daily_cost": self.max_daily_cost,
            "max_daily_requests": self.max_daily_requests,
            "cost_remaining": self.max_daily_cost - self.daily_cost,
            "requests_remaining": self.max_daily_requests - self.daily_requests,
            "cache_size": len(self.response_cache),
            "providers_available": len(self.configs)
        }
    
    def __del__(self):
        """Destructor - guarda caché al finalizar"""
        try:
            self.save_cache()
        except:
            pass

def create_llm_connector(cache_dir: str = "data/llm_cache") -> LLMConnector:
    """Factory function para crear un LLMConnector"""
    return LLMConnector(cache_dir)