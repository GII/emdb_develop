import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import Tk
from tkinter import filedialog

# Read the data
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
	title='Select goodness data file',
	filetypes=[('Tab-separated values', '*.txt'), ('All files', '*.*')],
)
if not file_path:
	raise SystemExit('No file selected.')

df = pd.read_csv(file_path, sep='\t')

output_dir = Path(file_path).resolve().parent

policy_counts = df['Policy'].value_counts().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(policy_counts.index.astype(str), policy_counts.values, color='steelblue')
ax.set_xlabel('Policy')
ax.set_ylabel('Execution Count')
ax.set_title('Policy Execution Counts')
ax.grid(True, axis='y')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
fig.tight_layout()
fig.savefig(output_dir / 'policy_execution_counts.svg')

policies = df['Policy'].astype(str).to_list()
alternating_runs = []
index = 0
while index + 3 < len(policies):
	first = policies[index]
	second = policies[index + 1]
	if first == second:
		index += 1
		continue
	pos = index + 2
	expected = first
	while pos < len(policies) and policies[pos] == expected:
		expected = second if expected == first else first
		pos += 1
	run_length = pos - index
	if run_length >= 4:
		alternating_runs.append((first, second, run_length))
		index = pos - 1
	else:
		index += 1

fig, ax = plt.subplots(figsize=(8, 5))
if alternating_runs:
	pairs = [f'{a} <-> {b}' for a, b, _ in alternating_runs]
	counts = pd.Series(pairs).value_counts().sort_values(ascending=False)
	ax.bar(counts.index, counts.values, color='darkorange')
	ax.set_xlabel('Policy Pair')
	ax.set_ylabel('Loop Count')
	ax.set_title('Alternating Policy Loops (ABAB...)')
	ax.grid(True, axis='y')
	plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
else:
	ax.text(0.5, 0.5, 'No alternating loops found', ha='center', va='center')
	ax.set_axis_off()
fig.tight_layout()
fig.savefig(output_dir / 'policy_alternating_loops.svg')
plt.show()