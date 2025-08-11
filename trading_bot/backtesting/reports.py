"""
Sistema de Reportes para Backtesting
===================================

Como trader senior, los reportes son mi herramienta de diagnóstico.
Un buen reporte me dice exactamente qué está funcionando y qué no.

Tipos de reportes:
- Reporte ejecutivo (métricas clave)
- Reporte detallado (análisis completo)
- Reporte visual (gráficos y charts)
- Reporte de comparación (vs benchmark)
- Reporte de recomendaciones (actionable insights)

Filosofía: "Los números no mienten, pero hay que saber leerlos"

Autor: Trader Algorítmico Senior (10+ años)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json

# Imports para visualización
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from ..utils.logger import get_logger, LogContext
from .metrics import PerformanceAnalyzer, MonteCarloAnalysis

class ReportType(Enum):
    """Tipos de reportes disponibles"""
    EXECUTIVE = "executive"      # Resumen ejecutivo
    DETAILED = "detailed"        # Análisis detallado
    VISUAL = "visual"           # Gráficos y visualizaciones
    COMPARISON = "comparison"    # Comparación con benchmark
    RECOMMENDATIONS = "recommendations"  # Recomendaciones actionables

@dataclass
class ReportConfig:
    """Configuración para generación de reportes"""
    report_types: List[ReportType]
    output_dir: str = "reports"
    include_charts: bool = True
    chart_format: str = "png"  # png, svg, pdf
    include_monte_carlo: bool = True
    monte_carlo_simulations: int = 1000
    
    # Configuración de formato
    currency_symbol: str = "$"
    decimal_places: int = 2
    percentage_places: int = 1

class BacktestReporter:
    """
    Generador principal de reportes de backtesting
    
    Como trader senior, he diseñado este sistema para generar
    reportes que realmente ayuden a tomar decisiones.
    """
    
    def __init__(self, config: ReportConfig):
        self.config = config
        self.logger = get_logger("BacktestReporter")
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar analizadores
        self.performance_analyzer = PerformanceAnalyzer()
        self.monte_carlo = MonteCarloAnalysis(config.monte_carlo_simulations)
    
    def generate_comprehensive_report(self, 
                                    equity_curve: pd.Series,
                                    trades: List[Dict[str, Any]],
                                    config_used: Dict[str, Any],
                                    benchmark: Optional[pd.Series] = None,
                                    report_name: str = None) -> Dict[str, str]:
        """
        Genera reporte comprensivo de backtesting
        
        Returns:
            Dict con paths de archivos generados
        """
        context = LogContext(component="backtest_reporter")
        
        try:
            if report_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_name = f"backtest_report_{timestamp}"
            
            # Análisis de performance
            performance_results = self.performance_analyzer.analyze_performance(
                equity_curve, trades, benchmark
            )
            
            # Monte Carlo (si está habilitado)
            monte_carlo_results = {}
            if self.config.include_monte_carlo and trades:
                monte_carlo_results = self.monte_carlo.run_monte_carlo(
                    trades, equity_curve.iloc[0]
                )
            
            generated_files = {}
            
            # Generar reportes según configuración
            for report_type in self.config.report_types:
                if report_type == ReportType.EXECUTIVE:
                    file_path = self._generate_executive_report(
                        report_name, performance_results, config_used
                    )
                    generated_files['executive'] = file_path
                
                elif report_type == ReportType.DETAILED:
                    file_path = self._generate_detailed_report(
                        report_name, performance_results, trades, 
                        config_used, monte_carlo_results
                    )
                    generated_files['detailed'] = file_path
                
                elif report_type == ReportType.VISUAL and PLOTTING_AVAILABLE:
                    file_path = self._generate_visual_report(
                        report_name, equity_curve, trades, performance_results
                    )
                    generated_files['visual'] = file_path
                
                elif report_type == ReportType.COMPARISON and benchmark is not None:
                    file_path = self._generate_comparison_report(
                        report_name, performance_results, benchmark
                    )
                    generated_files['comparison'] = file_path
                
                elif report_type == ReportType.RECOMMENDATIONS:
                    file_path = self._generate_recommendations_report(
                        report_name, performance_results, trades
                    )
                    generated_files['recommendations'] = file_path
            
            # Generar reporte JSON con todos los datos
            json_file = self._generate_json_report(
                report_name, performance_results, trades, 
                config_used, monte_carlo_results
            )
            generated_files['json'] = json_file
            
            self.logger.info(
                f"Reportes generados: {report_name}",
                context=context,
                extra_fields={
                    "files_generated": len(generated_files),
                    "report_types": [rt.value for rt in self.config.report_types]
                }
            )
            
            return generated_files
            
        except Exception as e:
            self.logger.error(f"Error generando reportes: {e}")
            return {}
    
    def _generate_executive_report(self, report_name: str, 
                                 performance_results: Dict[str, Any],
                                 config_used: Dict[str, Any]) -> str:
        """Genera reporte ejecutivo conciso"""
        try:
            file_path = self.output_dir / f"{report_name}_executive.txt"
            
            trade_metrics = performance_results['trade_metrics']
            risk_metrics = performance_results['risk_metrics']
            
            # Determinar calificación
            rating = self._calculate_strategy_rating(performance_results)
            
            content = f"""
================================================================================
                    REPORTE EJECUTIVO DE BACKTESTING
================================================================================

📊 RESUMEN EJECUTIVO
----------------------------------------
Capital Inicial:        {self.config.currency_symbol}{config_used.get('initial_capital', 0):,.2f}
Balance Final:          {self.config.currency_symbol}{config_used.get('initial_capital', 0) * (1 + performance_results['total_return']):,.2f}
Ganancia/Pérdida:       {self.config.currency_symbol}{config_used.get('initial_capital', 0) * performance_results['total_return']:,.2f}
ROI:                    {performance_results['total_return']:.2%}
ROI Anualizado:         {performance_results['annualized_return']:.2%}
Calificación:           {rating}

📈 MÉTRICAS CLAVE
----------------------------------------
Win Rate:               {trade_metrics.win_rate:.1%}
Profit Factor:          {trade_metrics.profit_factor:.2f}
Sharpe Ratio:           {performance_results['sharpe_ratio']:.2f}
Calmar Ratio:           {performance_results['calmar_ratio']:.2f}
Kelly Criterion:        {performance_results['kelly_criterion']:.1%}

🛡️ GESTIÓN DE RIESGO
----------------------------------------
Max Drawdown:           {risk_metrics.max_drawdown:.2%}
Duración Max DD:        {risk_metrics.max_drawdown_duration} días
VaR (95%):             {risk_metrics.var_95:.2%}
CVaR (95%):            {risk_metrics.cvar_95:.2%}

💰 ESTADÍSTICAS DE TRADING
----------------------------------------
Total de Trades:        {trade_metrics.total_trades}
Trades Ganadores:       {trade_metrics.winning_trades} ({trade_metrics.win_rate:.1%})
Trades Perdedores:      {trade_metrics.losing_trades}
Ganancia Promedio:      {self.config.currency_symbol}{trade_metrics.avg_win:.2f}
Pérdida Promedio:       {self.config.currency_symbol}{trade_metrics.avg_loss:.2f}
Expectancy:            {self.config.currency_symbol}{trade_metrics.expectancy:.2f}

⚙️ CONFIGURACIÓN UTILIZADA
----------------------------------------
Símbolos:               {', '.join(config_used.get('symbols', []))}
Score Mínimo:           {config_used.get('min_score_threshold', 'N/A')}
Stop Loss:              {config_used.get('stop_loss_pct', 0):.1%}
Take Profit:            {config_used.get('take_profit_pct', 0):.1%}
Tamaño Posición:        {config_used.get('position_size_pct', 0):.1%}

💡 RECOMENDACIONES RÁPIDAS
----------------------------------------
{self._get_quick_recommendations(performance_results)}

================================================================================
Reporte generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Por: Sistema de Trading Algorítmico
================================================================================
"""
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error generando reporte ejecutivo: {e}")
            return ""
    
    def _generate_detailed_report(self, report_name: str,
                                performance_results: Dict[str, Any],
                                trades: List[Dict[str, Any]],
                                config_used: Dict[str, Any],
                                monte_carlo_results: Dict[str, Any]) -> str:
        """Genera reporte detallado completo"""
        try:
            file_path = self.output_dir / f"{report_name}_detailed.txt"
            
            trade_metrics = performance_results['trade_metrics']
            risk_metrics = performance_results['risk_metrics']
            
            content = f"""
================================================================================
                    REPORTE DETALLADO DE BACKTESTING
================================================================================

📊 ANÁLISIS DE PERFORMANCE COMPLETO
----------------------------------------

1. RETORNOS Y VOLATILIDAD
   • Retorno Total:              {performance_results['total_return']:.2%}
   • Retorno Anualizado:         {performance_results['annualized_return']:.2%}
   • Volatilidad Anualizada:     {performance_results['volatility']:.2%}
   • Mejor Mes:                  {performance_results.get('best_month', 0):.2%}
   • Peor Mes:                   {performance_results.get('worst_month', 0):.2%}
   • Meses Positivos:            {performance_results.get('positive_months', 0)}/{performance_results.get('total_months', 0)}

2. RATIOS DE PERFORMANCE
   • Sharpe Ratio:               {performance_results['sharpe_ratio']:.3f}
   • Sortino Ratio:              {performance_results['sortino_ratio']:.3f}
   • Calmar Ratio:               {performance_results['calmar_ratio']:.3f}
   • Information Ratio:          {performance_results.get('benchmark_metrics', {}).get('information_ratio', 'N/A')}

3. ANÁLISIS DE RIESGO DETALLADO
   • Maximum Drawdown:           {risk_metrics.max_drawdown:.2%}
   • Duración Max Drawdown:      {risk_metrics.max_drawdown_duration} días
   • Drawdown Actual:            {risk_metrics.current_drawdown:.2%}
   • Value at Risk (95%):        {risk_metrics.var_95:.2%}
   • Conditional VaR (95%):      {risk_metrics.cvar_95:.2%}
   • Desviación Downside:        {risk_metrics.downside_deviation:.2%}
   • Desviación Upside:          {risk_metrics.upside_deviation:.2%}
   • Períodos de Drawdown:       {len(risk_metrics.drawdown_periods)}

4. ESTADÍSTICAS DE TRADING DETALLADAS
   • Total de Trades:            {trade_metrics.total_trades}
   • Trades Ganadores:           {trade_metrics.winning_trades} ({trade_metrics.win_rate:.2%})
   • Trades Perdedores:          {trade_metrics.losing_trades} ({trade_metrics.loss_rate:.2%})
   • Ganancia Bruta:             {self.config.currency_symbol}{trade_metrics.gross_profit:,.2f}
   • Pérdida Bruta:              {self.config.currency_symbol}{trade_metrics.gross_loss:,.2f}
   • Ganancia Neta:              {self.config.currency_symbol}{trade_metrics.net_profit:,.2f}
   • Profit Factor:              {trade_metrics.profit_factor:.3f}
   • Ganancia Promedio:          {self.config.currency_symbol}{trade_metrics.avg_win:.2f}
   • Pérdida Promedio:           {self.config.currency_symbol}{trade_metrics.avg_loss:.2f}
   • Mayor Ganancia:             {self.config.currency_symbol}{trade_metrics.largest_win:.2f}
   • Mayor Pérdida:              {self.config.currency_symbol}{trade_metrics.largest_loss:.2f}
   • Ratio Reward/Risk:          {trade_metrics.reward_risk_ratio:.2f}
   • Expectancy:                 {self.config.currency_symbol}{trade_metrics.expectancy:.2f}
   • Max Wins Consecutivas:      {trade_metrics.max_consecutive_wins}
   • Max Losses Consecutivas:    {trade_metrics.max_consecutive_losses}

5. POSITION SIZING Y KELLY
   • Kelly Criterion:            {performance_results['kelly_criterion']:.2%}
   • Kelly Recomendado:          {min(performance_results['kelly_criterion'] * 0.5, 0.10):.2%} (50% del Kelly)
   • Tamaño Actual Usado:        {config_used.get('position_size_pct', 0):.2%}

"""
            
            # Agregar análisis Monte Carlo si está disponible
            if monte_carlo_results:
                content += f"""
6. ANÁLISIS MONTE CARLO ({monte_carlo_results['simulations']} simulaciones)
   • Capital Final Promedio:     {self.config.currency_symbol}{monte_carlo_results['final_capital']['mean']:,.2f}
   • Capital Final (5%-95%):     {self.config.currency_symbol}{monte_carlo_results['final_capital']['percentile_5']:,.2f} - {self.config.currency_symbol}{monte_carlo_results['final_capital']['percentile_95']:,.2f}
   • Peor Caso:                  {self.config.currency_symbol}{monte_carlo_results['final_capital']['min']:,.2f}
   • Mejor Caso:                 {self.config.currency_symbol}{monte_carlo_results['final_capital']['max']:,.2f}
   • Probabilidad de Ganancia:   {monte_carlo_results['probability_profit']:.1%}
   • Prob. Pérdida >10%:         {monte_carlo_results['probability_loss_10pct']:.1%}
   • Prob. Pérdida >20%:         {monte_carlo_results['probability_loss_20pct']:.1%}
   • Peor Drawdown Simulado:     {monte_carlo_results['max_drawdown']['worst']:.2%}

"""
            
            # Agregar top trades
            if trades:
                content += self._generate_top_trades_analysis(trades)
            
            # Agregar análisis de drawdowns
            if risk_metrics.drawdown_periods:
                content += self._generate_drawdown_analysis(risk_metrics.drawdown_periods)
            
            # Agregar recomendaciones detalladas
            content += f"""
💡 RECOMENDACIONES DETALLADAS
----------------------------------------
{self._get_detailed_recommendations(performance_results, config_used)}

================================================================================
Reporte generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Por: Sistema de Trading Algorítmico Avanzado
================================================================================
"""
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error generando reporte detallado: {e}")
            return ""
    
    def _generate_visual_report(self, report_name: str,
                              equity_curve: pd.Series,
                              trades: List[Dict[str, Any]],
                              performance_results: Dict[str, Any]) -> str:
        """Genera reporte visual con gráficos"""
        try:
            if not PLOTTING_AVAILABLE:
                self.logger.warning("Matplotlib no disponible, saltando reporte visual")
                return ""
            
            # Configurar estilo
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Análisis Visual de Backtesting - {report_name}', fontsize=16, fontweight='bold')
            
            # 1. Equity Curve
            ax1 = axes[0, 0]
            ax1.plot(equity_curve.index, equity_curve.values, linewidth=2, color='blue')
            ax1.set_title('Curva de Equity', fontweight='bold')
            ax1.set_ylabel('Capital ($)')
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            # 2. Drawdown
            ax2 = axes[0, 1]
            rolling_max = equity_curve.expanding().max()
            drawdown = (equity_curve - rolling_max) / rolling_max * 100
            ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.7, color='red')
            ax2.set_title('Drawdown (%)', fontweight='bold')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            # 3. Distribución de Returns
            ax3 = axes[1, 0]
            returns = equity_curve.pct_change().dropna() * 100
            ax3.hist(returns, bins=50, alpha=0.7, color='green', edgecolor='black')
            ax3.axvline(returns.mean(), color='red', linestyle='--', label=f'Media: {returns.mean():.2f}%')
            ax3.set_title('Distribución de Returns Diarios', fontweight='bold')
            ax3.set_xlabel('Return (%)')
            ax3.set_ylabel('Frecuencia')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Trade PnL Distribution
            ax4 = axes[1, 1]
            if trades:
                pnls = [trade.get('pnl', 0) for trade in trades if 'pnl' in trade]
                if pnls:
                    ax4.hist(pnls, bins=30, alpha=0.7, color='purple', edgecolor='black')
                    ax4.axvline(np.mean(pnls), color='red', linestyle='--', label=f'Media: ${np.mean(pnls):.2f}')
                    ax4.set_title('Distribución de PnL por Trade', fontweight='bold')
                    ax4.set_xlabel('PnL ($)')
                    ax4.set_ylabel('Frecuencia')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Guardar gráfico
            file_path = self.output_dir / f"{report_name}_visual.{self.config.chart_format}"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error generando reporte visual: {e}")
            return ""
    
    def _generate_json_report(self, report_name: str,
                            performance_results: Dict[str, Any],
                            trades: List[Dict[str, Any]],
                            config_used: Dict[str, Any],
                            monte_carlo_results: Dict[str, Any]) -> str:
        """Genera reporte en formato JSON para procesamiento automático"""
        try:
            file_path = self.output_dir / f"{report_name}_data.json"
            
            # Preparar datos para JSON (convertir objetos no serializables)
            json_data = {
                'report_metadata': {
                    'name': report_name,
                    'generated_at': datetime.now().isoformat(),
                    'generator': 'Advanced Trading Bot Backtesting System'
                },
                'configuration': config_used,
                'performance_metrics': self._serialize_performance_results(performance_results),
                'monte_carlo_analysis': monte_carlo_results,
                'trade_summary': {
                    'total_trades': len(trades),
                    'trade_details': trades[:100] if len(trades) > 100 else trades  # Limitar para tamaño
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error generando reporte JSON: {e}")
            return ""
    
    def _calculate_strategy_rating(self, performance_results: Dict[str, Any]) -> str:
        """Calcula calificación de la estrategia basada en métricas clave"""
        try:
            trade_metrics = performance_results['trade_metrics']
            risk_metrics = performance_results['risk_metrics']
            
            score = 0
            
            # Win Rate (25 puntos máximo)
            if trade_metrics.win_rate >= 0.65:
                score += 25
            elif trade_metrics.win_rate >= 0.55:
                score += 20
            elif trade_metrics.win_rate >= 0.45:
                score += 15
            elif trade_metrics.win_rate >= 0.35:
                score += 10
            else:
                score += 5
            
            # Profit Factor (25 puntos máximo)
            if trade_metrics.profit_factor >= 2.5:
                score += 25
            elif trade_metrics.profit_factor >= 2.0:
                score += 20
            elif trade_metrics.profit_factor >= 1.5:
                score += 15
            elif trade_metrics.profit_factor >= 1.2:
                score += 10
            else:
                score += 5
            
            # Sharpe Ratio (25 puntos máximo)
            sharpe = performance_results['sharpe_ratio']
            if sharpe >= 2.0:
                score += 25
            elif sharpe >= 1.5:
                score += 20
            elif sharpe >= 1.0:
                score += 15
            elif sharpe >= 0.5:
                score += 10
            else:
                score += 5
            
            # Max Drawdown (25 puntos máximo)
            max_dd = abs(risk_metrics.max_drawdown)
            if max_dd <= 0.05:
                score += 25
            elif max_dd <= 0.10:
                score += 20
            elif max_dd <= 0.15:
                score += 15
            elif max_dd <= 0.25:
                score += 10
            else:
                score += 5
            
            # Determinar calificación
            if score >= 90:
                return "⭐⭐⭐⭐⭐ EXCELENTE - Sistema listo para producción"
            elif score >= 80:
                return "⭐⭐⭐⭐ MUY BUENO - Optimizar parámetros menores"
            elif score >= 70:
                return "⭐⭐⭐ BUENO - Revisar gestión de riesgo"
            elif score >= 60:
                return "⭐⭐ REGULAR - Revisar estrategia"
            else:
                return "⭐ POBRE - Rediseñar completamente"
            
        except Exception:
            return "❓ NO EVALUABLE - Error en cálculo"
    
    def _get_quick_recommendations(self, performance_results: Dict[str, Any]) -> str:
        """Genera recomendaciones rápidas basadas en métricas"""
        try:
            recommendations = []
            
            trade_metrics = performance_results['trade_metrics']
            risk_metrics = performance_results['risk_metrics']
            
            # Win Rate
            if trade_metrics.win_rate < 0.50:
                recommendations.append("• Win rate bajo - Revisar condiciones de entrada")
            
            # Profit Factor
            if trade_metrics.profit_factor < 1.5:
                recommendations.append("• Profit factor bajo - Mejorar ratio reward/risk")
            
            # Drawdown
            if abs(risk_metrics.max_drawdown) > 0.20:
                recommendations.append("• Drawdown alto - Implementar mejor gestión de riesgo")
            
            # Sharpe Ratio
            if performance_results['sharpe_ratio'] < 1.0:
                recommendations.append("• Sharpe bajo - Optimizar risk-adjusted returns")
            
            # Expectancy
            if trade_metrics.expectancy <= 0:
                recommendations.append("• Expectancy negativa - Sistema no rentable")
            
            if not recommendations:
                recommendations.append("• Sistema muestra métricas sólidas - Continuar optimización")
            
            return "\n".join(recommendations)
            
        except Exception:
            return "• Error generando recomendaciones"
    
    def _get_detailed_recommendations(self, performance_results: Dict[str, Any], 
                                   config_used: Dict[str, Any]) -> str:
        """Genera recomendaciones detalladas y actionables"""
        try:
            recommendations = []
            
            trade_metrics = performance_results['trade_metrics']
            risk_metrics = performance_results['risk_metrics']
            
            recommendations.append("ANÁLISIS Y RECOMENDACIONES ESPECÍFICAS:")
            recommendations.append("")
            
            # Análisis de Win Rate
            if trade_metrics.win_rate < 0.50:
                recommendations.append(f"1. WIN RATE BAJO ({trade_metrics.win_rate:.1%})")
                recommendations.append("   • Revisar condiciones de entrada - posiblemente muy agresivas")
                recommendations.append("   • Considerar filtros adicionales de calidad de señal")
                recommendations.append("   • Evaluar si el score mínimo es demasiado bajo")
                recommendations.append("")
            
            # Análisis de Profit Factor
            if trade_metrics.profit_factor < 2.0:
                recommendations.append(f"2. PROFIT FACTOR SUBÓPTIMO ({trade_metrics.profit_factor:.2f})")
                recommendations.append("   • Target: >2.0 para sistemas robustos")
                recommendations.append("   • Revisar niveles de take profit - posiblemente muy conservadores")
                recommendations.append("   • Evaluar niveles de stop loss - posiblemente muy amplios")
                recommendations.append("")
            
            # Análisis de Drawdown
            if abs(risk_metrics.max_drawdown) > 0.15:
                recommendations.append(f"3. DRAWDOWN EXCESIVO ({risk_metrics.max_drawdown:.1%})")
                recommendations.append("   • Implementar límites de drawdown diario más estrictos")
                recommendations.append("   • Reducir tamaño de posición durante períodos de pérdidas")
                recommendations.append("   • Considerar filtros de régimen de mercado")
                recommendations.append("")
            
            # Análisis de Position Sizing
            kelly = performance_results['kelly_criterion']
            current_size = config_used.get('position_size_pct', 0)
            if kelly > 0 and current_size > kelly:
                recommendations.append(f"4. POSITION SIZING AGRESIVO")
                recommendations.append(f"   • Kelly Criterion sugiere: {kelly:.1%}")
                recommendations.append(f"   • Tamaño actual: {current_size:.1%}")
                recommendations.append(f"   • Recomendado: {min(kelly * 0.5, 0.10):.1%} (50% del Kelly)")
                recommendations.append("")
            
            # Análisis de Expectancy
            if trade_metrics.expectancy > 0:
                recommendations.append(f"5. EXPECTANCY POSITIVA (${trade_metrics.expectancy:.2f})")
                recommendations.append("   • Sistema matemáticamente rentable")
                recommendations.append("   • Enfocar en consistencia y gestión de riesgo")
                recommendations.append("")
            else:
                recommendations.append(f"5. EXPECTANCY NEGATIVA (${trade_metrics.expectancy:.2f})")
                recommendations.append("   • Sistema no rentable en su forma actual")
                recommendations.append("   • Requiere rediseño fundamental de la estrategia")
                recommendations.append("")
            
            return "\n".join(recommendations)
            
        except Exception:
            return "Error generando recomendaciones detalladas"
    
    def _serialize_performance_results(self, performance_results: Dict[str, Any]) -> Dict[str, Any]:
        """Convierte objetos de performance a formato serializable"""
        try:
            serialized = {}
            
            for key, value in performance_results.items():
                if hasattr(value, '__dict__'):
                    # Convertir dataclass a dict
                    serialized[key] = value.__dict__.copy()
                    # Convertir datetime objects a strings
                    for k, v in serialized[key].items():
                        if isinstance(v, datetime):
                            serialized[key][k] = v.isoformat()
                else:
                    serialized[key] = value
            
            return serialized
            
        except Exception as e:
            self.logger.error(f"Error serializando resultados: {e}")
            return {}
    
    def _generate_top_trades_analysis(self, trades: List[Dict[str, Any]]) -> str:
        """Genera análisis de mejores y peores trades"""
        try:
            if not trades:
                return ""
            
            # Filtrar trades con PnL
            trades_with_pnl = [t for t in trades if 'pnl' in t]
            if not trades_with_pnl:
                return ""
            
            # Ordenar por PnL
            sorted_trades = sorted(trades_with_pnl, key=lambda x: x['pnl'], reverse=True)
            
            content = f"""
7. ANÁLISIS DE MEJORES Y PEORES TRADES
   
   TOP 5 MEJORES TRADES:
"""
            for i, trade in enumerate(sorted_trades[:5]):
                symbol = trade.get('symbol', 'N/A')
                pnl = trade.get('pnl', 0)
                pnl_pct = trade.get('pnl_pct', 0)
                content += f"   {i+1}. {symbol}: ${pnl:.2f} ({pnl_pct:.1%})\n"
            
            content += f"""
   TOP 5 PEORES TRADES:
"""
            for i, trade in enumerate(sorted_trades[-5:]):
                symbol = trade.get('symbol', 'N/A')
                pnl = trade.get('pnl', 0)
                pnl_pct = trade.get('pnl_pct', 0)
                content += f"   {i+1}. {symbol}: ${pnl:.2f} ({pnl_pct:.1%})\n"
            
            return content
            
        except Exception:
            return ""
    
    def _generate_drawdown_analysis(self, drawdown_periods: List[Dict[str, Any]]) -> str:
        """Genera análisis detallado de períodos de drawdown"""
        try:
            if not drawdown_periods:
                return ""
            
            content = f"""
8. ANÁLISIS DETALLADO DE DRAWDOWNS

   Total de Períodos de Drawdown: {len(drawdown_periods)}
   
   PEORES DRAWDOWNS:
"""
            
            # Ordenar por drawdown
            sorted_periods = sorted(drawdown_periods, key=lambda x: x['max_drawdown'])
            
            for i, period in enumerate(sorted_periods[:3]):
                start = period['start_date']
                end = period['end_date']
                duration = period['duration_days']
                max_dd = period['max_drawdown']
                recovery = period.get('recovery', 0)
                
                content += f"""   {i+1}. Período: {start} - {end}
      Duración: {duration} días
      Max Drawdown: {max_dd:.2%}
      Recuperación: {recovery:.2%}
      
"""
            
            return content
            
        except Exception:
            return ""

def create_backtest_report(equity_curve: pd.Series,
                         trades: List[Dict[str, Any]],
                         config_used: Dict[str, Any],
                         benchmark: Optional[pd.Series] = None,
                         output_dir: str = "reports",
                         report_name: str = None) -> Dict[str, str]:
    """
    Función de conveniencia para generar reporte completo
    
    Args:
        equity_curve: Serie temporal del equity
        trades: Lista de trades ejecutados
        config_used: Configuración utilizada en el backtest
        benchmark: Serie de benchmark opcional
        output_dir: Directorio de salida
        report_name: Nombre del reporte
        
    Returns:
        Dict con paths de archivos generados
    """
    config = ReportConfig(
        report_types=[
            ReportType.EXECUTIVE,
            ReportType.DETAILED,
            ReportType.VISUAL,
            ReportType.RECOMMENDATIONS
        ],
        output_dir=output_dir,
        include_charts=True,
        include_monte_carlo=True
    )
    
    reporter = BacktestReporter(config)
    return reporter.generate_comprehensive_report(
        equity_curve, trades, config_used, benchmark, report_name
    )