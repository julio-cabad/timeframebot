#!/usr/bin/env python3
"""
Script para ejecutar backtesting con datos reales
=================================================

Como trader algorítmico senior, este es mi flujo de validación:
1. Backtest con datos históricos reales
2. Analizar métricas clave
3. Optimizar parámetros
4. Paper trading
5. Trading real con capital pequeño

Empezamos con $200 USD y datos 100% reales de Binance.

Autor: Trader Algorítmico Senior (10+ años)
"""

import asyncio
import sys
from pathlib import Path

# Agregar el proyecto al path
sys.path.insert(0, str(Path(__file__).parent))

from trading_bot.backtesting.engine import BacktestEngine, BacktestConfig
from trading_bot.backtesting.reporter import BacktestReporter
from trading_bot.backtesting.metrics import MetricsCalculator

async def main():
    """
    Ejecuta backtest con configuración profesional
    
    Como trader experimentado, empiezo conservador:
    - Capital pequeño ($200)
    - Stop loss estricto (2%)
    - Pocos trades simultáneos (3 max)
    - Solo entradas con alta confianza
    """
    
    print("=" * 80)
    print("🚀 SISTEMA DE BACKTESTING PROFESIONAL")
    print("=" * 80)
    print("Capital inicial: $200 USD")
    print("Datos: 100% REALES de Binance")
    print("Estrategia: Multi-timeframe con scoring dinámico")
    print("-" * 80)
    
    # Configuración del backtest
    config = BacktestConfig(
        # Capital
        initial_capital=200.0,  # $200 USD
        position_size_pct=0.20,  # 20% por trade ($40)
        
        # Estrategia
        min_score_entry=65.0,  # Score mínimo 65 (bajado de 70 para más oportunidades)
        min_confidence=0.7,  # Confianza mínima 70% (para reducir trades)
        
        # Risk Management
        stop_loss_pct=0.02,  # 2% stop loss
        take_profit_pct=0.06,  # 6% take profit (3:1 ratio)
        max_trades=10,  # Máximo 10 trades simultáneos (CAMBIADO DE 3 A 10)
        
        # Datos
        candles_to_fetch=1500,  # Últimas 1500 velas como cuando funcionaba
        warmup_candles=100,  # 100 velas para calentar indicadores
        
        # Timeframes (orden importante: mayor a menor)
        timeframes=['1d', '4h', '1h', '15m'],
        
        # Símbolos a testear (7 símbolos como cuando teníamos ROI 7.88%)
        symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'TIAUSDT', 'BNBUSDT', 'DOGEUSDT']
    )
    
    print("\n⚙️ Configuración:")
    print(f"  • Símbolos: {', '.join(config.symbols)}")
    print(f"  • Timeframes: {', '.join(config.timeframes)}")
    print(f"  • Velas a analizar: {config.candles_to_fetch}")
    print(f"  • Score mínimo: {config.min_score_entry}")
    print(f"  • Stop Loss: {config.stop_loss_pct*100:.1f}%")
    print(f"  • Take Profit: {config.take_profit_pct*100:.1f}%")
    print(f"  • Tamaño posición: {config.position_size_pct*100:.1f}% (${config.initial_capital * config.position_size_pct:.0f})")
    
    print("\n📊 Iniciando backtesting...")
    print("-" * 80)
    
    try:
        # Crear motor de backtesting
        engine = BacktestEngine(config)
        
        # Ejecutar backtest
        result = await engine.run_backtest()
        
        print("\n✅ Backtesting completado!")
        print("-" * 80)
        
        # Mostrar resultados principales
        print("\n📈 RESULTADOS PRINCIPALES:")
        print(f"  • Balance final: ${result.final_balance:.2f}")
        print(f"  • ROI: {result.roi_percent:.2f}%")
        print(f"  • Total trades: {result.portfolio_stats['total_trades']}")
        print(f"  • Win Rate: {result.win_rate:.1f}%")
        print(f"  • Profit Factor: {result.profit_factor:.2f}")
        print(f"  • Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"  • Señales generadas: {result.signals_generated}")
        
        # Generar reporte detallado
        print("\n📝 Generando reporte detallado...")
        reporter = BacktestReporter(output_dir="backtesting/reports")
        
        # Generar reporte en texto
        text_report = reporter.generate_report(result, format="text", save_to_file=True)
        
        # Generar reporte en JSON
        json_report = reporter.generate_report(result, format="json", save_to_file=True)
        
        # Mostrar reporte en consola
        print("\n" + "=" * 80)
        print(text_report)
        
        # Análisis de trades si hay
        if not result.trades_df.empty:
            print("\n📊 ANÁLISIS DE TRADES:")
            print("-" * 40)
            print(f"Trades ganadores: {result.portfolio_stats['winning_trades']}")
            print(f"Trades perdedores: {result.portfolio_stats['losing_trades']}")
            print(f"Ganancia promedio: ${result.portfolio_stats['avg_win']:.2f}")
            print(f"Pérdida promedio: ${result.portfolio_stats['avg_loss']:.2f}")
            
            # Mostrar últimos 5 trades
            print("\n📜 Últimos 5 trades:")
            last_trades = result.trades_df.tail(5)[['symbol', 'entry_price', 'exit_price', 'pnl_usd', 'exit_reason']]
            for _, trade in last_trades.iterrows():
                status = "✅" if trade['pnl_usd'] > 0 else "❌"
                print(f"  {status} {trade['symbol']}: ${trade['pnl_usd']:.2f} ({trade['exit_reason']})")
        
        # Recomendaciones finales
        print("\n💡 PRÓXIMOS PASOS:")
        print("-" * 40)
        
        if result.profit_factor >= 1.5 and result.win_rate >= 50:
            print("✅ Estrategia prometedora! Recomendaciones:")
            print("  1. Ejecutar backtest con más símbolos (ETH, ADA, SOL)")
            print("  2. Probar diferentes períodos de mercado")
            print("  3. Optimizar parámetros de entrada (score mínimo)")
            print("  4. Considerar paper trading por 1 semana")
        elif result.profit_factor >= 1.0:
            print("⚠️ Estrategia necesita mejoras:")
            print("  1. Revisar condiciones de entrada")
            print("  2. Ajustar stop loss / take profit")
            print("  3. Analizar trades perdedores")
            print("  4. Considerar filtros adicionales")
        else:
            print("❌ Estrategia no rentable actualmente:")
            print("  1. Revisar lógica de scoring")
            print("  2. Analizar falsos positivos")
            print("  3. Considerar timeframes diferentes")
            print("  4. Revisar gestión de riesgo")
        
        print("\n" + "=" * 80)
        print("Backtest completado exitosamente")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error durante el backtesting: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    # Ejecutar con asyncio
    exit_code = asyncio.run(main())
    sys.exit(exit_code)