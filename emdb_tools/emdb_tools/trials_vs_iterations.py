import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import Tk
from tkinter import filedialog
from tkinter import simpledialog

# Configure matplotlib font sizes
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

# Read the data
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
	title='Select trials_vs_iterations data file',
	filetypes=[('Tab-separated values', '*.txt'), ('All files', '*.*')],
)
if not file_path:
	raise SystemExit('No file selected.')

df = pd.read_csv(file_path, sep='\t')

output_dir = Path(file_path).resolve().parent

line_value = simpledialog.askfloat(
	title='Horizontal line',
	prompt='Value for horizontal line (Iteration):',
)
if line_value is None:
	raise SystemExit('No line value provided.')

# Plot 1: Iterations vs Trial
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['Trial'], df['Iterations'], c=df['Success'].map({True: 'green', False: 'red'}))
ax.set_xlabel('Trial')
ax.set_ylabel('Iterations')
ax.set_title('Iterations vs Trial')
ax.grid(True)
fig.tight_layout()
fig.savefig(output_dir / 'iterations_vs_trial.svg')

# Plot 2: Success rate moving average
fig, ax = plt.subplots(figsize=(10, 6))
window_size = 100
success_numeric = df['Success'].astype(int)
success_ma = success_numeric.rolling(window=window_size, min_periods=1).mean()
ax.plot(df['Trial'].to_numpy(), success_ma.to_numpy(), color='green')
ax.set_xlabel('Trial')
ax.set_ylabel('Success Rate (moving average)')
ax.set_title(f'Success Rate Moving Average (window={window_size})')
ax.set_ylim(0, 1.1)
ax.grid(True)
fig.tight_layout()
fig.savefig(output_dir / 'success_rate.svg')

# Plot 3: Iteration vs Trial (line plot)
fig, ax = plt.subplots(figsize=(10, 6))
#ax.plot(df['Trial'].to_numpy(), df['Iterations'].to_numpy(), linestyle='-', color='blue', alpha=0.3, label='Raw data')
iterations_ma = df['Iterations'].rolling(window=50, min_periods=1).mean()
ax.plot(df['Trial'].to_numpy(), iterations_ma.to_numpy(), linestyle='-', color='blue', linewidth=2, label=f'Average')
ax.axhline(line_value, color='red', linestyle='--', linewidth=1.2, label='Reference')
ax.set_xlabel('Trial')
ax.set_ylabel('Iterations')
ax.set_title('Iteration Progress')
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig(output_dir / 'iteration_progress.svg')

# Plot 4: Success by Trial
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['green' if x else 'red' for x in df['Success']]
ax.bar(df['Trial'], df['Iterations'], color=colors, alpha=0.7)
ax.set_xlabel('Trial')
ax.set_ylabel('Iterations')
ax.set_title('Iterations per Trial (colored by Success)')
ax.grid(True, axis='y')
fig.tight_layout()
fig.savefig(output_dir / 'iterations_per_trial.svg')

plt.show()