"""
Generador de Reportes de Backtesting
====================================

Como trader profesional, los reportes claros son ESENCIALES.
Este módulo genera reportes detallados en múltiples formatos.

Incluye:
- Resumen ejecutivo con métricas clave
- Análisis detallado de trades
- Gráficos de equity curve
- Distribución de PnL
- Análisis de drawdown

Autor: Trader Algorítmico Senior
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Any
import json
from pathlib import Path

from .metrics import PerformanceMetrics

class BacktestReporter:
    """
    Generador de reportes profesionales de backtesting
    
    Como trader con 10+ años, sé que un buen reporte debe ser:
    - Claro y conciso
    - Enfocado en métricas que importan
    - Visual cuando sea posible
    - Actionable (que permita tomar decisiones)
    """
    
    def __init__(self, output_dir: str = "backtesting/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, backtest_result: Any, 
                       format: str = "text",
                       save_to_file: bool = True) -> str:
        """
        Genera reporte completo del backtesting
        
        Args:
            backtest_result: Resultado del backtest
            format: Formato del reporte ('text', 'json', 'html')
            save_to_file: Si guardar en archivo
            
        Returns:
            String con el reporte formateado
        """
        if format == "text":
            report = self._generate_text_report(backtest_result)
        elif format == "json":
            report = self._generate_json_report(backtest_result)
        elif format == "html":
            report = self._generate_html_report(backtest_result)
        else:
            report = self._generate_text_report(backtest_result)
        
        if save_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_report_{timestamp}.{format}"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                f.write(report)
            
            print(f"📊 Reporte guardado en: {filepath}")
        
        return report
    
    def _generate_text_report(self, result: Any) -> str:
        """
        Genera reporte en formato texto
        
        Este es mi formato favorito - simple, claro y directo
        """
        stats = result.portfolio_stats
        
        report = []
        report.append("=" * 80)
        report.append("                    REPORTE DE BACKTESTING")
        report.append("=" * 80)
        report.append("")
        
        # Resumen Ejecutivo
        report.append("📊 RESUMEN EJECUTIVO")
        report.append("-" * 40)
        report.append(f"Capital Inicial:        ${result.config.initial_capital:,.2f}")
        report.append(f"Balance Final:          ${stats['final_balance']:,.2f}")
        report.append(f"Ganancia/Pérdida:       ${stats['total_pnl']:,.2f}")
        report.append(f"ROI:                    {stats['roi_percent']:.2f}%")
        
        # Calificación
        rating = self._get_strategy_rating(stats)
        report.append(f"Calificación:           {rating}")
        report.append("")
        
        # Estadísticas de Trading
        report.append("📈 ESTADÍSTICAS DE TRADING")
        report.append("-" * 40)
        report.append(f"Total de Trades:        {stats['total_trades']}")
        report.append(f"Trades Ganadores:       {stats['winning_trades']} ({stats['win_rate']:.1f}%)")
        report.append(f"Trades Perdedores:      {stats['losing_trades']}")
        report.append(f"Profit Factor:          {stats['profit_factor']:.2f}")
        report.append(f"Ganancia Promedio:      ${stats['avg_win']:.2f}")
        report.append(f"Pérdida Promedio:       ${stats['avg_loss']:.2f}")
        report.append("")
        
        # Gestión de Riesgo
        report.append("🛡️ GESTIÓN DE RIESGO")
        report.append("-" * 40)
        report.append(f"Max Drawdown:           {stats['max_drawdown_pct']:.2f}%")
        report.append(f"Balance Máximo:         ${stats['peak_balance']:,.2f}")
        report.append(f"Balance Mínimo:         ${stats['lowest_balance']:,.2f}")
        report.append("")
        
        # Costos
        report.append("💰 COSTOS DE OPERACIÓN")
        report.append("-" * 40)
        report.append(f"Comisiones Totales:     ${stats['total_fees']:.2f}")
        report.append(f"% de Ganancias:         {(stats['total_fees']/abs(stats['total_pnl'])*100 if stats['total_pnl'] != 0 else 0):.1f}%")
        report.append("")
        
        # Configuración usada
        report.append("⚙️ CONFIGURACIÓN DEL BACKTEST")
        report.append("-" * 40)
        report.append(f"Símbolos:               {', '.join(result.config.symbols)}")
        report.append(f"Timeframes:             {', '.join(result.config.timeframes)}")
        report.append(f"Velas analizadas:       {result.config.candles_to_fetch}")
        report.append(f"Score mínimo entrada:   {result.config.min_score_entry}")
        report.append(f"Stop Loss:              {result.config.stop_loss_pct*100:.1f}%")
        report.append(f"Take Profit:            {result.config.take_profit_pct*100:.1f}%")
        report.append(f"Tamaño posición:        {result.config.position_size_pct*100:.1f}%")
        report.append(f"Max trades simultáneos: {result.config.max_trades}")
        report.append("")
        
        # Detalles de ejecución
        report.append("⏱️ DETALLES DE EJECUCIÓN")
        report.append("-" * 40)
        report.append(f"Señales generadas:      {result.signals_generated}")
        report.append(f"Tiempo de ejecución:    {result.execution_time:.1f} segundos")
        report.append("")
        
        # Top trades
        if not result.trades_df.empty:
            report.append("🏆 TOP 5 MEJORES TRADES")
            report.append("-" * 40)
            top_trades = result.trades_df.nlargest(5, 'pnl_usd')[['symbol', 'pnl_usd', 'pnl_percent']]
            for _, trade in top_trades.iterrows():
                report.append(f"  {trade['symbol']}: ${trade['pnl_usd']:.2f} ({trade['pnl_percent']:.1f}%)")
            report.append("")
            
            report.append("📉 TOP 5 PEORES TRADES")
            report.append("-" * 40)
            worst_trades = result.trades_df.nsmallest(5, 'pnl_usd')[['symbol', 'pnl_usd', 'pnl_percent']]
            for _, trade in worst_trades.iterrows():
                report.append(f"  {trade['symbol']}: ${trade['pnl_usd']:.2f} ({trade['pnl_percent']:.1f}%)")
            report.append("")
        
        # Recomendaciones
        report.append("💡 RECOMENDACIONES")
        report.append("-" * 40)
        recommendations = self._generate_recommendations(stats)
        for rec in recommendations:
            report.append(f"• {rec}")
        report.append("")
        
        # Footer
        report.append("=" * 80)
        report.append(f"Reporte generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("Por: Sistema de Trading Algorítmico v1.0")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _generate_json_report(self, result: Any) -> str:
        """Genera reporte en formato JSON para análisis programático"""
        
        report_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "version": "1.0"
            },
            "configuration": {
                "initial_capital": result.config.initial_capital,
                "symbols": result.config.symbols,
                "timeframes": result.config.timeframes,
                "min_score_entry": result.config.min_score_entry,
                "stop_loss_pct": result.config.stop_loss_pct,
                "take_profit_pct": result.config.take_profit_pct,
                "position_size_pct": result.config.position_size_pct,
                "max_trades": result.config.max_trades
            },
            "performance": {
                "final_balance": result.portfolio_stats['final_balance'],
                "total_pnl": result.portfolio_stats['total_pnl'],
                "roi_percent": result.portfolio_stats['roi_percent'],
                "total_trades": result.portfolio_stats['total_trades'],
                "win_rate": result.portfolio_stats['win_rate'],
                "profit_factor": result.portfolio_stats['profit_factor'],
                "max_drawdown_pct": result.portfolio_stats['max_drawdown_pct']
            },
            "trades": result.trades_df.to_dict('records') if not result.trades_df.empty else [],
            "balance_curve": result.balance_curve.to_dict('records') if not result.balance_curve.empty else [],
            "execution": {
                "signals_generated": result.signals_generated,
                "execution_time": result.execution_time
            }
        }
        
        return json.dumps(report_data, indent=2, default=str)
    
    def _generate_html_report(self, result: Any) -> str:
        """
        Genera reporte en formato HTML con gráficos
        
        TODO: Implementar con plotly para gráficos interactivos
        """
        # Por ahora, versión simple HTML
        text_report = self._generate_text_report(result)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report</title>
            <style>
                body {{ font-family: monospace; margin: 20px; background: #1a1a1a; color: #00ff00; }}
                pre {{ background: #0a0a0a; padding: 20px; border-radius: 5px; }}
                h1 {{ color: #00ff00; }}
            </style>
        </head>
        <body>
            <h1>Backtest Report</h1>
            <pre>{text_report}</pre>
        </body>
        </html>
        """
        
        return html
    
    def _get_strategy_rating(self, stats: Dict) -> str:
        """
        Califica la estrategia basado en métricas clave
        
        Como trader experimentado, estas son mis criterios:
        """
        score = 0
        
        # Win rate (max 25 puntos)
        if stats['win_rate'] >= 70:
            score += 25
        elif stats['win_rate'] >= 60:
            score += 20
        elif stats['win_rate'] >= 50:
            score += 15
        elif stats['win_rate'] >= 40:
            score += 10
        else:
            score += 5
        
        # Profit factor (max 25 puntos)
        if stats['profit_factor'] >= 2.0:
            score += 25
        elif stats['profit_factor'] >= 1.5:
            score += 20
        elif stats['profit_factor'] >= 1.2:
            score += 15
        elif stats['profit_factor'] >= 1.0:
            score += 10
        else:
            score += 0
        
        # ROI (max 25 puntos)
        roi = stats['roi_percent']
        if roi >= 50:
            score += 25
        elif roi >= 30:
            score += 20
        elif roi >= 15:
            score += 15
        elif roi >= 5:
            score += 10
        elif roi >= 0:
            score += 5
        else:
            score += 0
        
        # Max Drawdown (max 25 puntos)
        dd = stats['max_drawdown_pct']
        if dd <= 5:
            score += 25
        elif dd <= 10:
            score += 20
        elif dd <= 15:
            score += 15
        elif dd <= 20:
            score += 10
        elif dd <= 30:
            score += 5
        else:
            score += 0
        
        # Calificación final
        if score >= 85:
            return "⭐⭐⭐⭐⭐ EXCELENTE - Lista para producción"
        elif score >= 70:
            return "⭐⭐⭐⭐ MUY BUENA - Optimizar parámetros"
        elif score >= 55:
            return "⭐⭐⭐ BUENA - Necesita mejoras"
        elif score >= 40:
            return "⭐⭐ REGULAR - Revisar estrategia"
        else:
            return "⭐ POBRE - No recomendada"
    
    def _generate_recommendations(self, stats: Dict) -> list:
        """
        Genera recomendaciones basadas en los resultados
        
        Como mentor, estas son mis sugerencias automáticas
        """
        recommendations = []
        
        # Win rate
        if stats['win_rate'] < 40:
            recommendations.append("Win rate muy bajo - Revisar condiciones de entrada")
        elif stats['win_rate'] < 50:
            recommendations.append("Mejorar filtros de entrada para aumentar win rate")
        
        # Profit factor
        if stats['profit_factor'] < 1.0:
            recommendations.append("Profit factor < 1 - La estrategia pierde dinero")
        elif stats['profit_factor'] < 1.5:
            recommendations.append("Aumentar ratio reward/risk o mejorar win rate")
        
        # Drawdown
        if stats['max_drawdown_pct'] > 20:
            recommendations.append("Drawdown muy alto - Reducir tamaño de posición")
        elif stats['max_drawdown_pct'] > 15:
            recommendations.append("Considerar stops más ajustados para reducir drawdown")
        
        # Número de trades
        if stats['total_trades'] < 10:
            recommendations.append("Pocos trades - Aumentar período de backtest")
        elif stats['total_trades'] < 30:
            recommendations.append("Muestra pequeña - Validar con más datos")
        
        # ROI
        if stats['roi_percent'] < 0:
            recommendations.append("ROI negativo - Revisar toda la estrategia")
        elif stats['roi_percent'] < 10:
            recommendations.append("ROI bajo - Optimizar puntos de entrada/salida")
        
        # Ratio win/loss
        if stats['avg_win'] > 0 and stats['avg_loss'] > 0:
            ratio = stats['avg_win'] / stats['avg_loss']
            if ratio < 1.5:
                recommendations.append("Mejorar ratio ganancia/pérdida (actualmente < 1.5)")
        
        if not recommendations:
            recommendations.append("Estrategia con buen performance - Considerar paper trading")
            recommendations.append("Validar con diferentes períodos de mercado")
            recommendations.append("Implementar gestión de riesgo adicional antes de ir live")
        
        return recommendations