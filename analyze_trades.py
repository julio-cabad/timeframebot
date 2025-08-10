#!/usr/bin/env python3
"""
Analizador de Trades del Backtest
=================================

Script para extraer y mostrar todos los trades ejecutados
en formato tabla para verificación visual en TradingView.

Autor: Trader Algorítmico Senior
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import glob

def load_latest_backtest_report():
    """Carga el reporte más reciente de backtesting"""
    reports_dir = Path("backtesting/reports")
    
    # Buscar el archivo JSON más reciente
    json_files = glob.glob(str(reports_dir / "backtest_report_*.json"))
    
    if not json_files:
        print("❌ No se encontraron reportes de backtesting")
        return None
    
    # Obtener el más reciente
    latest_file = max(json_files, key=lambda x: Path(x).stat().st_mtime)
    
    print(f"📂 Cargando reporte: {Path(latest_file).name}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def format_trades_table(report_data):
    """Formatea los trades en una tabla detallada"""
    
    trades = report_data.get('trades', [])
    
    if not trades:
        print("❌ No hay trades en el reporte")
        return None
    
    # Crear DataFrame con los trades
    df = pd.DataFrame(trades)
    
    # Seleccionar y renombrar columnas relevantes
    columns_mapping = {
        'entry_time': 'Fecha/Hora Entrada',
        'symbol': 'Símbolo',
        'side': 'Lado',
        'entry_price': 'Precio Entrada',
        'exit_price': 'Precio Salida',
        'stop_loss': 'Stop Loss',
        'take_profit': 'Take Profit',
        'exit_time': 'Fecha/Hora Salida',
        'pnl_usd': 'Ganancia/Pérdida ($)',
        'pnl_percent': 'G/P (%)',
        'exit_reason': 'Razón Salida',
        'position_size_usd': 'Tamaño Posición ($)'
    }
    
    # Filtrar solo las columnas que necesitamos
    available_columns = [col for col in columns_mapping.keys() if col in df.columns]
    df_filtered = df[available_columns].copy()
    
    # Renombrar columnas
    df_filtered.rename(columns={k: columns_mapping[k] for k in available_columns}, inplace=True)
    
    # Formatear fechas para mejor legibilidad
    if 'Fecha/Hora Entrada' in df_filtered.columns:
        df_filtered['Fecha/Hora Entrada'] = pd.to_datetime(df_filtered['Fecha/Hora Entrada']).dt.strftime('%Y-%m-%d %H:%M')
    if 'Fecha/Hora Salida' in df_filtered.columns:
        df_filtered['Fecha/Hora Salida'] = pd.to_datetime(df_filtered['Fecha/Hora Salida']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Formatear números
    numeric_columns = ['Precio Entrada', 'Precio Salida', 'Stop Loss', 'Take Profit', 
                      'Ganancia/Pérdida ($)', 'G/P (%)', 'Tamaño Posición ($)']
    
    for col in numeric_columns:
        if col in df_filtered.columns:
            if 'Precio' in col or 'Stop' in col or 'Take' in col:
                df_filtered[col] = df_filtered[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            elif '$' in col:
                df_filtered[col] = df_filtered[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
            elif '%' in col:
                df_filtered[col] = df_filtered[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
    
    # Agregar columna de suma acumulada
    if 'pnl_usd' in df.columns:
        df_filtered['Balance Acumulado'] = df['pnl_usd'].cumsum() + 200  # Capital inicial $200
        df_filtered['Balance Acumulado'] = df_filtered['Balance Acumulado'].apply(lambda x: f"${x:.2f}")
    
    return df_filtered

def print_trades_summary(df):
    """Imprime resumen de trades ganadores y perdedores"""
    
    print("\n" + "="*120)
    print("📊 RESUMEN DE TRADES")
    print("="*120)
    
    # Contar ganadores y perdedores
    if 'Ganancia/Pérdida ($)' in df.columns:
        # Convertir de string a float para análisis
        pnl_values = df['Ganancia/Pérdida ($)'].str.replace('$', '').astype(float)
        
        winners = pnl_values > 0
        losers = pnl_values < 0
        
        print(f"\n✅ Trades Ganadores: {winners.sum()}")
        print(f"❌ Trades Perdedores: {losers.sum()}")
        print(f"📈 Total Trades: {len(df)}")
        print(f"💰 Ganancia Total: ${pnl_values.sum():.2f}")
        print(f"📊 Win Rate: {(winners.sum()/len(df)*100):.1f}%")

def print_trades_for_tradingview(df):
    """Imprime trades en formato para verificar en TradingView"""
    
    print("\n" + "="*120)
    print("🔍 TRADES DETALLADOS PARA VERIFICAR EN TRADINGVIEW")
    print("="*120)
    
    # Separar ganadores y perdedores
    df_copy = df.copy()
    df_copy['pnl_numeric'] = df_copy['Ganancia/Pérdida ($)'].str.replace('$', '').astype(float)
    winners = df_copy[df_copy['pnl_numeric'] > 0]
    losers = df_copy[df_copy['pnl_numeric'] <= 0]
    
    print(f"\n📊 RESUMEN: {len(winners)} GANADORES / {len(losers)} PERDEDORES")
    print("="*120)
    
    # Primero mostrar TODOS los perdedores
    print("\n❌ TRADES PERDEDORES (TODOS)")
    print("-"*80)
    for idx, row in losers.iterrows():
        print(f"\nTrade #{idx+1} - PERDEDOR")
        print(f"  📅 Entrada: {row.get('Fecha/Hora Entrada', 'N/A')}")
        print(f"  🪙 Símbolo: {row.get('Símbolo', 'N/A')}")
        print(f"  💵 Precio Entrada: {row.get('Precio Entrada', 'N/A')}")
        print(f"  🛑 Stop Loss: {row.get('Stop Loss', 'N/A')}")
        print(f"  🎯 Take Profit: {row.get('Take Profit', 'N/A')}")
        print(f"  📅 Salida: {row.get('Fecha/Hora Salida', 'N/A')}")
        print(f"  💵 Precio Salida: {row.get('Precio Salida', 'N/A')}")
        print(f"  ❌ Pérdida: {row.get('Ganancia/Pérdida ($)', 'N/A')} ({row.get('G/P (%)', 'N/A')})")
        print(f"  💼 Balance después: {row.get('Balance Acumulado', 'N/A')}")
        print(f"  📝 Razón: {row.get('Razón Salida', 'N/A')}")
    
    # Luego mostrar TODOS los ganadores
    print("\n✅ TRADES GANADORES (TODOS)")
    print("-"*80)
    for idx, row in winners.iterrows():
        print(f"\nTrade #{idx+1} - GANADOR")
        print(f"  📅 Entrada: {row.get('Fecha/Hora Entrada', 'N/A')}")
        print(f"  🪙 Símbolo: {row.get('Símbolo', 'N/A')}")
        print(f"  💵 Precio Entrada: {row.get('Precio Entrada', 'N/A')}")
        print(f"  🛑 Stop Loss: {row.get('Stop Loss', 'N/A')}")
        print(f"  🎯 Take Profit: {row.get('Take Profit', 'N/A')}")
        print(f"  📅 Salida: {row.get('Fecha/Hora Salida', 'N/A')}")
        print(f"  💵 Precio Salida: {row.get('Precio Salida', 'N/A')}")
        print(f"  ✅ Ganancia: {row.get('Ganancia/Pérdida ($)', 'N/A')} ({row.get('G/P (%)', 'N/A')})")
        print(f"  💼 Balance después: {row.get('Balance Acumulado', 'N/A')}")
        print(f"  📝 Razón: {row.get('Razón Salida', 'N/A')}")

def save_to_csv(df):
    """Guarda los trades en CSV para análisis externo"""
    filename = f"trades_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"\n💾 Trades guardados en: {filename}")
    return filename

def main():
    """Función principal"""
    
    print("="*120)
    print("🔍 ANALIZADOR DE TRADES DEL BACKTEST")
    print("="*120)
    
    # Cargar reporte
    report = load_latest_backtest_report()
    
    if not report:
        return
    
    # Formatear trades
    df_trades = format_trades_table(report)
    
    if df_trades is None or df_trades.empty:
        print("❌ No hay trades para analizar")
        return
    
    # Mostrar tabla completa
    print("\n" + "="*120)
    print("📊 TABLA COMPLETA DE TRADES")
    print("="*120)
    
    # Configurar pandas para mostrar todas las columnas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print("\n", df_trades.to_string(index=True))
    
    # Mostrar resumen
    print_trades_summary(df_trades)
    
    # Mostrar trades para TradingView
    print_trades_for_tradingview(df_trades)
    
    # Guardar en CSV
    csv_file = save_to_csv(df_trades)
    
    # Información adicional
    print("\n" + "="*120)
    print("📝 INSTRUCCIONES PARA TRADINGVIEW:")
    print("="*120)
    print("1. Abre TradingView y busca el símbolo correspondiente")
    print("2. Ajusta el timeframe a 1H (principal del sistema)")
    print("3. Navega a la fecha/hora de entrada del trade")
    print("4. Marca la entrada con una línea horizontal en el precio")
    print("5. Marca el Stop Loss (línea roja) y Take Profit (línea verde)")
    print("6. Verifica si el análisis multi-timeframe tenía sentido")
    print("\n💡 TIP: Usa el CSV generado para importar los datos a Excel")
    
    print("\n" + "="*120)
    print("✅ Análisis completado")
    print("="*120)

if __name__ == "__main__":
    main()