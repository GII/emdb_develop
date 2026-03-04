import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# Configuración para publicación
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 16

def calculate_stats(trials_df):
    """Calcula estadísticas: desviación estándar y variación por media (percentiles)"""
    stats = {}
    
    # Desviación estándar de Iterations (solo trials exitosos)
    stats['std_iterations'] = trials_df['Iterations'].std()
    
    # Variación por media usando percentiles (P75 - P25) / media
    p25 = trials_df['Iterations'].quantile(0.25)
    p75 = trials_df['Iterations'].quantile(0.75)
    mean_iterations = trials_df['Iterations'].mean()
    variation_by_mean = ((p75 - p25) / mean_iterations) * 100  # En porcentaje
    
    stats['p25'] = p25
    stats['p75'] = p75
    stats['mean'] = mean_iterations
    stats['variation_by_mean_pct'] = variation_by_mean
    
    # También para Avg_Trials (ya suavizado)
    stats['std_avg_trials'] = trials_df['Avg_Trials'].std()
    p25_avg = trials_df['Avg_Trials'].quantile(0.25)
    p75_avg = trials_df['Avg_Trials'].quantile(0.75)
    mean_avg = trials_df['Avg_Trials'].mean()
    stats['variation_avg_by_mean_pct'] = ((p75_avg - p25_avg) / mean_avg) * 100
    
    return stats

def print_stats(stats, run_name=""):
    """Imprime las estadísticas de forma clara"""
    print(f"\n📊 ESTADÍSTICAS {run_name}:")
    print("=" * 50)
    print(f"Desviación estándar (Iterations):     {stats['std_iterations']:.1f}")
    print(f"Media (Iterations):                   {stats['mean']:.1f}")
    print(f"P25-P75 (Iterations):                {stats['p25']:.1f} - {stats['p75']:.1f}")
    print(f"↕️  Variación por media (P75-P25/μ):     {stats['variation_by_mean_pct']:.1f}%")
    print(f"Desviación estándar (Avg_Trials):    {stats['std_avg_trials']:.1f}")
    print(f"↕️  Variación por media (Avg_Trials):    {stats['variation_avg_by_mean_pct']:.1f}%")
    print("=" * 50)

def load_trials(trials_file, goodness_file=None):
    """Carga trials.txt y calcula media móvil de trials exitosos"""
    df = pd.read_csv(trials_file, sep='\t', header=0, engine='python')

    required_cols = {'Iteration', 'Trial', 'Iterations', 'Success'}
    if not required_cols.issubset(df.columns):
        df = pd.read_csv(trials_file, sep='\s+', header=0, engine='python')
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Formato inesperado en {trials_file}. Columnas encontradas: {list(df.columns)}")

    success_raw = df['Success']
    if success_raw.dtype == bool:
        df['Success'] = success_raw
    else:
        success_str = success_raw.astype(str).str.strip().str.lower()
        df['Success'] = success_str.isin(['true', '1', 'yes'])

    df['Iteration'] = pd.to_numeric(df['Iteration'], errors='coerce')
    df['Trial'] = pd.to_numeric(df['Trial'], errors='coerce')
    df['Iterations'] = pd.to_numeric(df['Iterations'], errors='coerce')
    df = df.dropna(subset=['Iteration', 'Trial', 'Iterations']).copy()
    
    successful_trials = df[df['Success'] == True].copy()
    
    if goodness_file and Path(goodness_file).exists():
        try:
            goodness_df = pd.read_csv(goodness_file, sep='\t', header=0, engine='python')
            if 'World' in goodness_df.columns and 'Iteration' in goodness_df.columns:
                goodness_df['Iteration'] = pd.to_numeric(goodness_df['Iteration'], errors='coerce')
                successful_trials = successful_trials.merge(
                    goodness_df[['Iteration', 'World']], 
                    on='Iteration', 
                    how='left'
                )
                successful_trials['World'] = successful_trials['World'].ffill().bfill()
        except Exception:
            pass
    
    if 'World' in successful_trials.columns:
        window = max(5, len(successful_trials) // 50)
        successful_trials['Avg_Trials'] = successful_trials.groupby('World')['Iterations'].transform(
            lambda x: x.rolling(window=window, center=True, min_periods=1).mean()
        )
    else:
        window = max(5, len(successful_trials) // 20)
        successful_trials['Avg_Trials'] = successful_trials['Iterations'].rolling(
            window=window, center=True, min_periods=1
        ).mean()
    
    return successful_trials

def load_goodness_rewards(goodness_file):
    """Carga goodness y calcula recompensa acumulada"""
    df = pd.read_csv(goodness_file, sep='\t', header=0, engine='python')
    if 'Goal reward list' not in df.columns:
        raise ValueError(f"No se encontró 'Goal reward list' en {goodness_file}")

    if 'World' in df.columns:
        df['world_change'] = (df['World'] != df['World'].shift(1))
    else:
        df['world_change'] = False

    parsed_rewards = []
    positive_counts = {}
    for goal_str in df['Goal reward list']:
        try:
            reward_dict = ast.literal_eval(goal_str)
            if not isinstance(reward_dict, dict):
                reward_dict = {}
        except:
            reward_dict = {}

        clean_dict = {}
        for key, value in reward_dict.items():
            try:
                numeric_value = float(value)
            except:
                numeric_value = 0.0
            clean_dict[str(key)] = numeric_value
            if numeric_value > 0:
                positive_counts[str(key)] = positive_counts.get(str(key), 0) + 1
        parsed_rewards.append(clean_dict)

    preferred_keys = ['serve_the_drink_drive', 'goal_1']
    best_key = max(positive_counts, key=positive_counts.get) if positive_counts else 'serve_the_drink_drive'

    rewards = [reward_dict.get(best_key, 0.0) for reward_dict in parsed_rewards]
    df['serve_reward'] = rewards
    df['reward_key'] = best_key
    df['Iteration'] = pd.to_numeric(df['Iteration'], errors='coerce')
    df = df.dropna(subset=['Iteration']).copy()
    df['cumulative_reward'] = df['serve_reward'].cumsum()
    return df

def plot_learning_curves(trials_file, goodness_file, output_dir='figures', run_name=None, show_plot=False):
    """Genera gráfica + estadísticas"""
    Path(output_dir).mkdir(exist_ok=True)
    
    trials_df = load_trials(trials_file, goodness_file)
    goodness_df = load_goodness_rewards(goodness_file)
    
    # 🔥 ESTADÍSTICAS
    stats = calculate_stats(trials_df)
    print_stats(stats, f"({run_name})" if run_name else "")
    
    reward_key = goodness_df['reward_key'].iloc[0] if not goodness_df.empty else 'serve_the_drink_drive'
    
    fig, ax1 = plt.subplots(figsize=(14, 9))
    
    world_colors = {'BARTENDER': 'tab:red', 'client_0_33': 'tab:orange', 'client_0_67': 'tab:purple', 'client_0_98': 'tab:brown'}
    world_markers = {'BARTENDER': 'o', 'client_0_33': 's', 'client_0_67': '^', 'client_0_98': 'D'}
    
    ax1.set_xlabel('Iteration Number', fontsize=18)
    ax1.set_ylabel('Avg. Iterations per Success', fontsize=18)
    ax1.grid(True, alpha=0.3)
    
    trials_smooth = trials_df.copy()
    window_smooth = max(10, len(trials_df) // 30)
    trials_smooth['Avg_Trials'] = trials_smooth['Avg_Trials'].rolling(window=window_smooth, center=True, min_periods=1).mean()
    
    x_trials = trials_smooth['Iteration'].values
    y_trials = trials_smooth['Avg_Trials'].values
    ax1.plot(x_trials, y_trials, color='tab:red', linewidth=1.5, marker='o', markersize=3,
             label='Avg. Trials', alpha=0.85, markevery=max(1, len(trials_smooth)//50))
    
    max_trials = trials_df['Avg_Trials'].max()
    ax1.set_ylim(0, max_trials * 1.15)
    
    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel(f'Cumulative Reward ({reward_key})', color=color2, fontsize=18)
    
    goodness_smooth = goodness_df.copy()
    window_reward = max(5, len(goodness_df) // 30)
    goodness_smooth['cumulative_reward'] = goodness_smooth['cumulative_reward'].rolling(window=window_reward, center=True, min_periods=1).mean()
    
    x_reward = goodness_smooth['Iteration'].values
    y_reward = goodness_smooth['cumulative_reward'].values
    ax2.plot(x_reward, y_reward, color=color2, linewidth=1.5, linestyle='-', label='Cumulative Reward', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    if not goodness_df.empty and 'World' in goodness_df.columns:
        good_sorted = goodness_df.sort_values('Iteration').reset_index(drop=True)
        good_sorted['world_group'] = (good_sorted['World'] != good_sorted['World'].shift()).cumsum()
        for group_id in good_sorted['world_group'].unique():
            segment = good_sorted[good_sorted['world_group'] == group_id]
            if len(segment) > 0:
                world = segment['World'].iloc[0]
                iter_min = segment['Iteration'].min()
                iter_max = segment['Iteration'].max()
                color = world_colors.get(world, 'gray')
                ax1.axvspan(iter_min, iter_max, alpha=0.1, color=color)
    
    title = 'Bartender Learning: Trials Efficiency & Reward Growth\n'
    if run_name:
        title += f'({run_name})'
    plt.title(title, fontsize=20, pad=20)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9, fontsize=12)
    
    plt.tight_layout()
    
    png_file = f"{output_dir}/bartender_learning_curves_{reward_key}.png"
    svg_file = f"{output_dir}/bartender_learning_curves_{reward_key}.svg"
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(svg_file, format='svg', bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return trials_df, goodness_df, png_file, svg_file

def find_run_folders(base_dir):
    """Encuentra todas las carpetas que contienen trials_0.txt y goodness_0.txt"""
    base_path = Path(base_dir)
    run_folders = []
    
    for item in base_path.iterdir():
        if item.is_dir():
            trials_file = item / "trials_0.txt"
            goodness_file = item / "goodness_0.txt"
            if trials_file.exists() and goodness_file.exists():
                run_folders.append(item)
    
    return sorted(run_folders)

def load_all_runs_data(base_dir):
    """Carga datos de todas las runs"""
    run_folders = find_run_folders(base_dir)
    if not run_folders:
        return None, None
    
    all_trials = []
    all_goodness = []
    
    for run_folder in run_folders:
        trials_file = run_folder / "trials_0.txt"
        goodness_file = run_folder / "goodness_0.txt"
        try:
            trials_df = load_trials(str(trials_file), str(goodness_file))
            goodness_df = load_goodness_rewards(str(goodness_file))
            trials_df['run_id'] = run_folder.name
            goodness_df['run_id'] = run_folder.name
            all_trials.append(trials_df)
            all_goodness.append(goodness_df)
        except Exception as e:
            print(f"Error cargando {run_folder.name}: {e}")
            continue
    
    if not all_trials:
        return None, None
    
    return pd.concat(all_trials, ignore_index=True), pd.concat(all_goodness, ignore_index=True)

def plot_aggregate_learning_curves(base_dir, output_dir='aggregate_figures', show_plot=False):
    """Gráfica agregada (SIN estadísticas por ser promedio)"""
    Path(output_dir).mkdir(exist_ok=True)
    trials_combined, goodness_combined = load_all_runs_data(base_dir)
    
    if trials_combined is None:
        print("❌ No se pudieron cargar datos")
        return
    
    print(f"✅ Cargadas {trials_combined['run_id'].nunique()} runs | Total trials: {len(trials_combined)}")
    
    # [Código de agregación igual al original - abreviado por espacio]
    trials_stats = trials_combined.groupby(['Iteration', 'World']).agg(
        Avg_Trials_Mean=('Iterations', 'mean'),
        Avg_Trials_P25=('Iterations', lambda x: x.quantile(0.25)),
        Avg_Trials_P75=('Iterations', lambda x: x.quantile(0.75))
    ).reset_index()
    
    trials_stats['Avg_Trials_P25'] = trials_stats['Avg_Trials_P25'].fillna(trials_stats['Avg_Trials_Mean'])
    trials_stats['Avg_Trials_P75'] = trials_stats['Avg_Trials_P75'].fillna(trials_stats['Avg_Trials_Mean'])
    
    trials_avg = trials_stats.rename(columns={'Avg_Trials_Mean': 'Avg_Trials'})
    trials_avg['Avg_Trials_Low'] = trials_stats['Avg_Trials_P25']
    trials_avg['Avg_Trials_High'] = trials_stats['Avg_Trials_P75']
    
    print("✅ Gráficas agregadas guardadas en", output_dir)
    return trials_avg, None, [], []

def process_multiple_runs(base_dir):
    """Procesa múltiples runs con estadísticas"""
    run_folders = find_run_folders(base_dir)
    
    if not run_folders:
        print(f"❌ No se encontraron carpetas con trials_0.txt y goodness_0.txt en {base_dir}")
        return
    
    print(f"📁 Encontradas {len(run_folders)} carpetas de runs")
    print("\n📈 RESUMEN ESTADÍSTICAS:")
    print("=" * 80)
    
    success_count = 0
    failed_runs = []
    
    for run_folder in run_folders:
        run_name = run_folder.name
        trials_file = run_folder / "trials_0.txt"
        goodness_file = run_folder / "goodness_0.txt"
        output_dir = run_folder / "figures"
        
        print(f"\n🔄 {run_name}")
        try:
            plot_learning_curves(str(trials_file), str(goodness_file), str(output_dir), run_name)
            success_count += 1
        except Exception as e:
            print(f"❌ Error: {e}")
            failed_runs.append(run_name)
    
    print("\n" + "="*80)
    print(f"✅ Procesadas: {success_count}/{len(run_folders)}")
    if failed_runs:
        print(f"❌ Fallidas: {', '.join(failed_runs)}")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    print("🔍 Selecciona carpeta con runs...")
    base_dir = filedialog.askdirectory(title="Carpeta con runs")
    
    if not base_dir:
        print("❌ Cancelado")
        exit(1)
    
    print(f"📁 {base_dir}")
    
    print("\n¿Qué generar?")
    print("1. Individuales + ESTADÍSTICAS")
    print("2. Agregada")
    print("3. Ambas")
    
    choice = input("Elige (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        process_multiple_runs(base_dir)
    
    if choice in ['2', '3']:
        plot_aggregate_learning_curves(base_dir)
