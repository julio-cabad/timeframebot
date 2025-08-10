#!/usr/bin/env python3
"""
Análisis Cronológico y Decisiones Críticas
==========================================

Script para ordenar trades cronológicamente y analizar 
performance por símbolo para tomar decisiones críticas.

Autor: Trader Algorítmico Senior
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json

def load_trades_csv():
    """Carga el CSV de trades más reciente"""
    csv_file = "trades_analysis_20250810_130210.csv"
    
    if not Path(csv_file).exists():
        print(f"❌ No se encontró el archivo {csv_file}")
        return None
    
    # Cargar CSV
    df = pd.read_csv(csv_file)
    
    # Limpiar datos
    df['Fecha/Hora Entrada'] = pd.to_datetime(df['Fecha/Hora Entrada'], format='%Y-%m-%d %H:%M')
    df['Fecha/Hora Salida'] = pd.to_datetime(df['Fecha/Hora Salida'], format='%Y-%m-%d %H:%M')
    
    # Convertir columnas numéricas
    df['Ganancia/Pérdida ($)'] = df['Ganancia/Pérdida ($)'].str.replace('$', '').astype(float)
    df['G/P (%)'] = df['G/P (%)'].str.replace('%', '').astype(float)
    df['Tamaño Posición ($)'] = df['Tamaño Posición ($)'].str.replace('$', '').astype(float)
    
    return df

def analyze_chronologically(df):
    """Ordena trades cronológicamente y muestra tabla completa"""
    
    print("\n" + "="*120)
    print("📅 TRADES ORDENADOS CRONOLÓGICAMENTE (Junio 17 → Agosto 10, 2025)")
    print("="*120)
    
    # Ordenar por fecha de entrada
    df_sorted = df.sort_values('Fecha/Hora Entrada')
    
    # Resetear índice para numeración correcta
    df_sorted = df_sorted.reset_index(drop=True)
    
    # Crear tabla formateada
    print("\n┌─────┬────────────────────┬──────────┬──────┬───────────────┬───────────────┬────────────────────┬──────────┬─────────┬─────────────┬─────────────┐")
    print("│ No. │ Fecha/Hora Entrada │ Símbolo  │ Lado │ Precio Entrada│ Precio Salida │ Fecha/Hora Salida  │   G/P    │  G/P %  │ Razón      │ Balance Acc │")
    print("├─────┼────────────────────┼──────────┼──────┼───────────────┼───────────────┼────────────────────┼──────────┼─────────┼─────────────┼─────────────┤")
    
    balance = 200.0  # Capital inicial
    
    for idx, row in df_sorted.iterrows():
        balance += row['Ganancia/Pérdida ($)']
        
        # Formatear valores
        fecha_entrada = row['Fecha/Hora Entrada'].strftime('%Y-%m-%d %H:%M')
        fecha_salida = row['Fecha/Hora Salida'].strftime('%Y-%m-%d %H:%M')
        simbolo = row['Símbolo']
        lado = row['Lado']
        precio_entrada = f"{row['Precio Entrada']:.2f}"
        precio_salida = f"{row['Precio Salida']:.2f}"
        ganancia = row['Ganancia/Pérdida ($)']
        ganancia_pct = row['G/P (%)']
        razon = row['Razón Salida']
        
        # Color para ganancia/pérdida
        if ganancia > 0:
            ganancia_str = f"+${ganancia:.2f}"
            emoji = "✅"
        else:
            ganancia_str = f"-${abs(ganancia):.2f}"
            emoji = "❌"
        
        print(f"│ {idx+1:3d} │ {fecha_entrada} │ {simbolo:8s} │ {lado:4s} │ {precio_entrada:>13s} │ {precio_salida:>13s} │ {fecha_salida} │ {emoji} {ganancia_str:>7s} │ {ganancia_pct:>6.1f}% │ {razon:11s} │ ${balance:>9.2f} │")
    
    print("└─────┴────────────────────┴──────────┴──────┴───────────────┴───────────────┴────────────────────┴──────────┴─────────┴─────────────┴─────────────┘")
    
    return df_sorted

def analyze_by_symbol(df):
    """Analiza performance por símbolo para decisiones críticas"""
    
    print("\n" + "="*120)
    print("⚠️ ANÁLISIS POR SÍMBOLO - DECISIONES CRÍTICAS")
    print("="*120)
    
    # Agrupar por símbolo
    symbol_stats = []
    
    for symbol in df['Símbolo'].unique():
        symbol_trades = df[df['Símbolo'] == symbol]
        
        wins = symbol_trades[symbol_trades['Ganancia/Pérdida ($)'] > 0]
        losses = symbol_trades[symbol_trades['Ganancia/Pérdida ($)'] <= 0]
        
        total_trades = len(symbol_trades)
        total_wins = len(wins)
        total_losses = len(losses)
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = wins['Ganancia/Pérdida ($)'].sum() if len(wins) > 0 else 0
        total_loss = abs(losses['Ganancia/Pérdida ($)'].sum()) if len(losses) > 0 else 0
        net_pnl = symbol_trades['Ganancia/Pérdida ($)'].sum()
        
        avg_win = total_profit / total_wins if total_wins > 0 else 0
        avg_loss = total_loss / total_losses if total_losses > 0 else 0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        symbol_stats.append({
            'Símbolo': symbol,
            'Trades': total_trades,
            'Ganadores': total_wins,
            'Perdedores': total_losses,
            'Win Rate (%)': win_rate,
            'Ganancia Total ($)': total_profit,
            'Pérdida Total ($)': total_loss,
            'PnL Neto ($)': net_pnl,
            'Profit Factor': profit_factor,
            'Promedio Ganancia ($)': avg_win,
            'Promedio Pérdida ($)': avg_loss
        })
    
    # Crear DataFrame y ordenar por Win Rate
    stats_df = pd.DataFrame(symbol_stats)
    stats_df = stats_df.sort_values('Win Rate (%)', ascending=False)
    
    # Mostrar tabla de estadísticas
    print("\n📊 ESTADÍSTICAS POR SÍMBOLO:")
    print("─" * 120)
    print(f"{'Símbolo':<10} {'Trades':>7} {'Ganadores':>10} {'Perdedores':>11} {'Win Rate':>10} {'PnL Neto':>12} {'Profit Factor':>14} {'Decisión':>15}")
    print("─" * 120)
    
    decisions = []
    
    for _, row in stats_df.iterrows():
        symbol = row['Símbolo']
        trades = row['Trades']
        wins = row['Ganadores']
        losses = row['Perdedores']
        win_rate = row['Win Rate (%)']
        pnl = row['PnL Neto ($)']
        pf = row['Profit Factor']
        
        # Decisión basada en métricas
        if win_rate < 30:
            decision = "🚫 ELIMINAR"
            reason = "Win rate muy bajo"
        elif pf < 1.0:
            decision = "⚠️ REVISAR"
            reason = "Profit Factor < 1"
        elif win_rate >= 50 and pf > 1.5:
            decision = "✅ MANTENER"
            reason = "Buenas métricas"
        elif trades < 3:
            decision = "📊 + DATOS"
            reason = "Pocos trades"
        else:
            decision = "⚠️ OPTIMIZAR"
            reason = "Métricas medianas"
        
        decisions.append((symbol, decision, reason, win_rate, pnl))
        
        # Formatear Profit Factor
        pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
        
        print(f"{symbol:<10} {trades:>7} {wins:>10} {losses:>11} {win_rate:>9.1f}% ${pnl:>11.2f} {pf_str:>14} {decision:>15}")
    
    print("─" * 120)
    
    return decisions

def print_critical_decisions(decisions):
    """Imprime las decisiones críticas a tomar"""
    
    print("\n" + "="*120)
    print("🎯 DECISIONES CRÍTICAS RECOMENDADAS")
    print("="*120)
    
    # Separar por tipo de decisión
    eliminar = [d for d in decisions if "ELIMINAR" in d[1]]
    revisar = [d for d in decisions if "REVISAR" in d[1]]
    mantener = [d for d in decisions if "MANTENER" in d[1]]
    optimizar = [d for d in decisions if "OPTIMIZAR" in d[1]]
    
    if eliminar:
        print("\n🚫 SÍMBOLOS A ELIMINAR INMEDIATAMENTE:")
        print("─" * 60)
        for symbol, decision, reason, win_rate, pnl in eliminar:
            print(f"  • {symbol}: Win Rate {win_rate:.1f}%, PnL ${pnl:.2f}")
            print(f"    Razón: {reason}")
            print(f"    ACCIÓN: Remover de config.symbols en run_backtest.py")
    
    if revisar:
        print("\n⚠️ SÍMBOLOS QUE REQUIEREN REVISIÓN:")
        print("─" * 60)
        for symbol, decision, reason, win_rate, pnl in revisar:
            print(f"  • {symbol}: Win Rate {win_rate:.1f}%, PnL ${pnl:.2f}")
            print(f"    Razón: {reason}")
            print(f"    ACCIÓN: Ajustar stop loss dinámico o filtros")
    
    if mantener:
        print("\n✅ SÍMBOLOS CON BUEN DESEMPEÑO:")
        print("─" * 60)
        for symbol, decision, reason, win_rate, pnl in mantener:
            print(f"  • {symbol}: Win Rate {win_rate:.1f}%, PnL ${pnl:.2f}")
            print(f"    Razón: {reason}")
            print(f"    ACCIÓN: Mantener en la estrategia")
    
    if optimizar:
        print("\n🔧 SÍMBOLOS A OPTIMIZAR:")
        print("─" * 60)
        for symbol, decision, reason, win_rate, pnl in optimizar:
            print(f"  • {symbol}: Win Rate {win_rate:.1f}%, PnL ${pnl:.2f}")
            print(f"    Razón: {reason}")
            print(f"    ACCIÓN: Ajustar parámetros específicos")
    
    # Resumen de cambios propuestos
    print("\n" + "="*120)
    print("📝 CONFIGURACIÓN PROPUESTA PARA PRÓXIMO BACKTEST")
    print("="*120)
    
    symbols_to_keep = [d[0] for d in decisions if "MANTENER" in d[1] or "OPTIMIZAR" in d[1]]
    
    print("\nModificar en run_backtest.py:")
    print("─" * 60)
    print("config.symbols = [")
    for symbol in symbols_to_keep:
        print(f"    '{symbol}',")
    print("]")
    
    print("\n💡 RECOMENDACIONES ADICIONALES:")
    print("─" * 60)
    print("1. Aumentar min_score a 72 para reducir señales falsas")
    print("2. Implementar filtro de tendencia macro (daily trend)")
    print("3. Añadir filtro de correlación BTC para altcoins")
    print("4. Considerar reducir position_size_pct a 15% para mejor gestión de riesgo")
    print("5. Implementar trailing stop para proteger ganancias")

def analyze_stop_losses(df):
    """Analiza los stop losses ejecutados"""
    
    print("\n" + "="*120)
    print("🛑 ANÁLISIS DE STOP LOSSES")
    print("="*120)
    
    # Filtrar trades cerrados por stop loss
    sl_trades = df[df['Razón Salida'] == 'stop_loss']
    
    # Agrupar por símbolo
    print("\nStop Losses por Símbolo:")
    print("─" * 60)
    
    for symbol in sl_trades['Símbolo'].unique():
        symbol_sl = sl_trades[sl_trades['Símbolo'] == symbol]
        avg_loss = symbol_sl['G/P (%)'].mean()
        
        print(f"\n{symbol}:")
        print(f"  • Total SL ejecutados: {len(symbol_sl)}")
        print(f"  • Pérdida promedio: {avg_loss:.2f}%")
        
        # Verificar si los SL están en el rango esperado
        if symbol == 'BTCUSDT':
            expected_sl = -2.0
        elif symbol in ['ETHUSDT', 'BNBUSDT']:
            expected_sl = -2.5
        else:
            expected_sl = -3.0
        
        if abs(avg_loss) > abs(expected_sl) * 1.5:
            print(f"  ⚠️ ALERTA: Stop loss más grande de lo esperado (esperado: {expected_sl}%)")

def main():
    """Función principal"""
    
    print("="*120)
    print("🔍 ANÁLISIS CRONOLÓGICO Y DECISIONES CRÍTICAS")
    print("="*120)
    
    # Cargar datos
    df = load_trades_csv()
    
    if df is None or df.empty:
        print("❌ No hay datos para analizar")
        return
    
    # 1. Análisis cronológico
    df_sorted = analyze_chronologically(df)
    
    # 2. Análisis por símbolo
    decisions = analyze_by_symbol(df)
    
    # 3. Decisiones críticas
    print_critical_decisions(decisions)
    
    # 4. Análisis de stop losses
    analyze_stop_losses(df)
    
    # 5. Resumen final
    print("\n" + "="*120)
    print("📈 RESUMEN EJECUTIVO")
    print("="*120)
    
    total_trades = len(df)
    winners = len(df[df['Ganancia/Pérdida ($)'] > 0])
    losers = len(df[df['Ganancia/Pérdida ($)'] <= 0])
    total_pnl = df['Ganancia/Pérdida ($)'].sum()
    win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
    
    print(f"\n• Período: Junio 17 - Agosto 10, 2025")
    print(f"• Total Trades: {total_trades}")
    print(f"• Ganadores: {winners} ({win_rate:.1f}%)")
    print(f"• Perdedores: {losers} ({100-win_rate:.1f}%)")
    print(f"• PnL Total: ${total_pnl:.2f}")
    print(f"• Balance Final: ${200 + total_pnl:.2f}")
    print(f"• ROI: {(total_pnl/200*100):.2f}%")
    
    print("\n✅ Análisis completado")
    print("="*120)

if __name__ == "__main__":
    main()