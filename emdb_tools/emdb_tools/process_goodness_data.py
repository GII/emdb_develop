#!/usr/bin/env python3
"""
Script to process and analyze goodness iteration data from EMDB experiments.
Can be used with GUI (no arguments) or CLI (with arguments).
"""

import argparse
import ast
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Graphs will not be generated.", file=sys.stderr)


class GoodnessDataProcessor:
    """Process and analyze goodness iteration data."""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.data = []
        self.headers = []
        
    def load_data(self):
        """Load data from text file (supports tab or space-separated values)."""
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            raise ValueError("Empty file")
        
        # Detect separator (tab or multiple spaces)
        first_line = lines[0]
        if '\t' in first_line:
            separator = '\t'
        else:
            # Use regex to split by multiple spaces
            import re
            separator = re.compile(r'\s{2,}')  # 2 or more spaces
            
        # Parse headers
        if separator == '\t':
            self.headers = [h.strip() for h in first_line.split(separator)]
        else:
            self.headers = [h.strip() for h in separator.split(first_line)]
        
        # Parse data rows
        for line in lines[1:]:
            if not line.strip():
                continue
            
            if separator == '\t':
                parts = line.strip().split(separator)
            else:
                parts = [p.strip() for p in separator.split(line.strip())]
                
            if len(parts) >= 6:
                try:
                    row = {
                        'iteration': int(parts[0]),
                        'world': parts[1],
                        'goal_rewards': ast.literal_eval(parts[2]),
                        'policy': parts[3],
                        'sensorial_changes': parts[4].strip().lower() == 'true',
                        'c_nodes': int(parts[5])
                    }
                    self.data.append(row)
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Skipping malformed line: {line.strip()}", file=sys.stderr)
                    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of the dataset."""
        if not self.data:
            return {}
            
        total_iterations = len(self.data)
        unique_worlds = set(row['world'] for row in self.data)
        unique_policies = set(row['policy'] for row in self.data)
        sensorial_changes_count = sum(1 for row in self.data if row['sensorial_changes'])
        
        c_nodes_values = [row['c_nodes'] for row in self.data]
        max_c_nodes = max(c_nodes_values)
        min_c_nodes = min(c_nodes_values)
        avg_c_nodes = sum(c_nodes_values) / len(c_nodes_values)
        
        return {
            'total_iterations': total_iterations,
            'unique_worlds': len(unique_worlds),
            'worlds': sorted(unique_worlds),
            'unique_policies': len(unique_policies),
            'policies': sorted(unique_policies),
            'sensorial_changes_count': sensorial_changes_count,
            'sensorial_changes_percentage': (sensorial_changes_count / total_iterations) * 100,
            'c_nodes_max': max_c_nodes,
            'c_nodes_min': min_c_nodes,
            'c_nodes_avg': avg_c_nodes
        }
        
    def get_policy_analysis(self) -> Dict:
        """Analyze policy usage patterns."""
        policy_counter = Counter(row['policy'] for row in self.data)
        policy_with_changes = Counter(
            row['policy'] for row in self.data if row['sensorial_changes']
        )
        
        analysis = {}
        for policy, count in policy_counter.items():
            changes_count = policy_with_changes.get(policy, 0)
            analysis[policy] = {
                'total_executions': count,
                'with_sensorial_changes': changes_count,
                'change_rate': (changes_count / count) * 100 if count > 0 else 0
            }
            
        return analysis
        
    def get_goal_evolution(self) -> List[Dict]:
        """Track the evolution of goals over iterations."""
        evolution = []
        for row in self.data:
            goals = row['goal_rewards']
            non_zero_goals = {k: v for k, v in goals.items() if float(v) != 0.0}
            
            evolution.append({
                'iteration': row['iteration'],
                'world': row['world'],
                'total_goals': len(goals),
                'active_goals': len(non_zero_goals),
                'active_goal_names': list(non_zero_goals.keys()),
                'c_nodes': row['c_nodes']
            })
            
        return evolution
        
    def get_c_nodes_transitions(self) -> List[Tuple[int, int, int]]:
        """Get transitions where C-nodes value changes."""
        transitions = []
        for i in range(1, len(self.data)):
            prev_c = self.data[i-1]['c_nodes']
            curr_c = self.data[i]['c_nodes']
            if prev_c != curr_c:
                transitions.append((
                    self.data[i]['iteration'],
                    prev_c,
                    curr_c
                ))
        return transitions
        
    def find_successful_policies(self) -> List[Dict]:
        """Find policies that resulted in sensorial changes and goal achievement."""
        successful = []
        for row in self.data:
            if row['sensorial_changes']:
                non_zero_goals = {k: v for k, v in row['goal_rewards'].items() if float(v) != 0.0}
                successful.append({
                    'iteration': row['iteration'],
                    'world': row['world'],
                    'policy': row['policy'],
                    'achieved_goals': non_zero_goals,
                    'c_nodes': row['c_nodes']
                })
        return successful
        
    def get_world_statistics(self) -> Dict[str, Dict]:
        """Get statistics grouped by world."""
        world_data = defaultdict(list)
        for row in self.data:
            world_data[row['world']].append(row)
            
        stats = {}
        for world, rows in world_data.items():
            stats[world] = {
                'iterations': len(rows),
                'unique_policies': len(set(r['policy'] for r in rows)),
                'sensorial_changes': sum(1 for r in rows if r['sensorial_changes']),
                'max_c_nodes': max(r['c_nodes'] for r in rows),
                'policies_used': Counter(r['policy'] for r in rows)
            }
            
        return stats
        
    def print_report(self):
        """Print a comprehensive analysis report."""
        print("=" * 80)
        print("GOODNESS ITERATION DATA ANALYSIS")
        print("=" * 80)
        print()
        
        # Summary statistics
        print("SUMMARY STATISTICS")
        print("-" * 80)
        summary = self.get_summary_statistics()
        print(f"Total iterations: {summary['total_iterations']}")
        print(f"Unique worlds: {summary['unique_worlds']} - {summary['worlds']}")
        print(f"Unique policies: {summary['unique_policies']}")
        print(f"Sensorial changes: {summary['sensorial_changes_count']} "
              f"({summary['sensorial_changes_percentage']:.1f}%)")
        print(f"C-nodes range: {summary['c_nodes_min']} - {summary['c_nodes_max']} "
              f"(avg: {summary['c_nodes_avg']:.2f})")
        print()
        
        # Policy analysis
        print("POLICY ANALYSIS")
        print("-" * 80)
        policy_analysis = self.get_policy_analysis()
        for policy, stats in sorted(policy_analysis.items(), 
                                    key=lambda x: x[1]['total_executions'], 
                                    reverse=True):
            print(f"{policy}:")
            print(f"  Total executions: {stats['total_executions']}")
            print(f"  With changes: {stats['with_sensorial_changes']} "
                  f"({stats['change_rate']:.1f}%)")
        print()
        
        # C-nodes transitions
        print("C-NODES TRANSITIONS")
        print("-" * 80)
        transitions = self.get_c_nodes_transitions()
        if transitions:
            for iteration, prev, curr in transitions:
                print(f"Iteration {iteration}: {prev} -> {curr} "
                      f"({'increase' if curr > prev else 'decrease'})")
        else:
            print("No C-nodes transitions found")
        print()
        
        # Successful policies
        print("SUCCESSFUL POLICIES (with sensorial changes)")
        print("-" * 80)
        successful = self.find_successful_policies()
        print(f"Total: {len(successful)} policies resulted in sensorial changes")
        if successful:
            print("\nFirst 10 successful executions:")
            for entry in successful[:10]:
                print(f"Iteration {entry['iteration']}: {entry['policy']} "
                      f"(C-nodes: {entry['c_nodes']})")
                if entry['achieved_goals']:
                    print(f"  Achieved goals: {entry['achieved_goals']}")
        print()
        
        # World statistics
        print("WORLD STATISTICS")
        print("-" * 80)
        world_stats = self.get_world_statistics()
        for world, stats in world_stats.items():
            print(f"{world}:")
            print(f"  Iterations: {stats['iterations']}")
            print(f"  Unique policies: {stats['unique_policies']}")
            print(f"  Sensorial changes: {stats['sensorial_changes']}")
            print(f"  Max C-nodes: {stats['max_c_nodes']}")
            print(f"  Top 3 policies: {stats['policies_used'].most_common(3)}")
            print()
    
    def generate_graphs(self, output_dir: str = "."):
        """Generate visualization graphs."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot generate graphs.", file=sys.stderr)
            return []
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        generated_files = []
        
        # Graph 1: C-nodes evolution over iterations
        fig, ax = plt.subplots(figsize=(12, 6))
        iterations = [row['iteration'] for row in self.data]
        c_nodes = [row['c_nodes'] for row in self.data]
        sensorial = [row['sensorial_changes'] for row in self.data]
        
        # Plot C-nodes evolution
        ax.plot(iterations, c_nodes, 'b-', linewidth=2, label='C-nodes')
        
        # Mark sensorial changes with red dots
        sensorial_iters = [self.data[i]['iteration'] for i in range(len(self.data)) if self.data[i]['sensorial_changes']]
        sensorial_cnodes = [self.data[i]['c_nodes'] for i in range(len(self.data)) if self.data[i]['sensorial_changes']]
        ax.scatter(sensorial_iters, sensorial_cnodes, color='red', s=50, alpha=0.6, label='Sensorial change', zorder=5)
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('C-nodes', fontsize=12)
        ax.set_title('C-nodes Evolution Over Iterations', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        graph_file = output_path / 'c_nodes_evolution.png'
        plt.tight_layout()
        plt.savefig(graph_file, dpi=150)
        plt.close()
        generated_files.append(str(graph_file))
        
        # Graph 2: Policy success rates
        fig, ax = plt.subplots(figsize=(12, 8))
        policy_analysis = self.get_policy_analysis()
        
        policies = list(policy_analysis.keys())
        success_rates = [policy_analysis[p]['change_rate'] for p in policies]
        total_exec = [policy_analysis[p]['total_executions'] for p in policies]
        
        # Sort by success rate
        sorted_data = sorted(zip(policies, success_rates, total_exec), key=lambda x: x[1], reverse=True)
        policies, success_rates, total_exec = zip(*sorted_data) if sorted_data else ([], [], [])
        
        colors = ['green' if rate > 50 else 'orange' if rate > 20 else 'red' for rate in success_rates]
        bars = ax.barh(range(len(policies)), success_rates, color=colors, alpha=0.7)
        
        # Add execution count labels
        for i, (rate, count) in enumerate(zip(success_rates, total_exec)):
            ax.text(rate + 1, i, f'n={count}', va='center', fontsize=9)
        
        ax.set_yticks(range(len(policies)))
        ax.set_yticklabels(policies, fontsize=10)
        ax.set_xlabel('Success Rate (%)', fontsize=12)
        ax.set_title('Policy Success Rates (% with Sensorial Changes)', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        graph_file = output_path / 'policy_success_rates.png'
        plt.tight_layout()
        plt.savefig(graph_file, dpi=150)
        plt.close()
        generated_files.append(str(graph_file))
        
        # Graph 3: Sensorial changes timeline
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Cumulative sensorial changes
        cumulative_changes = []
        count = 0
        for row in self.data:
            if row['sensorial_changes']:
                count += 1
            cumulative_changes.append(count)
        
        ax1.plot(iterations, cumulative_changes, 'g-', linewidth=2)
        ax1.set_ylabel('Cumulative Changes', fontsize=12)
        ax1.set_title('Sensorial Changes Timeline', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Changes per iteration (binary)
        changes_binary = [1 if row['sensorial_changes'] else 0 for row in self.data]
        ax2.bar(iterations, changes_binary, color='orange', alpha=0.7, width=0.8)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Change Occurred', fontsize=12)
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(True, alpha=0.3)
        
        graph_file = output_path / 'sensorial_changes_timeline.png'
        plt.tight_layout()
        plt.savefig(graph_file, dpi=150)
        plt.close()
        generated_files.append(str(graph_file))
        
        # Graph 4: Active goals evolution
        fig, ax = plt.subplots(figsize=(12, 6))
        
        goal_evolution = self.get_goal_evolution()
        iters = [g['iteration'] for g in goal_evolution]
        total_goals = [g['total_goals'] for g in goal_evolution]
        active_goals = [g['active_goals'] for g in goal_evolution]
        
        ax.plot(iters, total_goals, 'b-', linewidth=2, label='Total goals', alpha=0.7)
        ax.plot(iters, active_goals, 'r-', linewidth=2, label='Active goals (non-zero)')
        ax.fill_between(iters, active_goals, alpha=0.3, color='red')
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Number of Goals', fontsize=12)
        ax.set_title('Goal Evolution Over Iterations', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        graph_file = output_path / 'goal_evolution.png'
        plt.tight_layout()
        plt.savefig(graph_file, dpi=150)
        plt.close()
        generated_files.append(str(graph_file))
        
        # Graph 5: World distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        world_stats = self.get_world_statistics()
        
        worlds = list(world_stats.keys())
        iterations_per_world = [world_stats[w]['iterations'] for w in worlds]
        changes_per_world = [world_stats[w]['sensorial_changes'] for w in worlds]
        
        x = range(len(worlds))
        width = 0.35
        
        bars1 = ax.bar([i - width/2 for i in x], iterations_per_world, width, label='Total iterations', alpha=0.8)
        bars2 = ax.bar([i + width/2 for i in x], changes_per_world, width, label='Sensorial changes', alpha=0.8)
        
        ax.set_xlabel('World', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Iterations and Sensorial Changes by World', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(worlds, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        graph_file = output_path / 'world_distribution.png'
        plt.tight_layout()
        plt.savefig(graph_file, dpi=150)
        plt.close()
        generated_files.append(str(graph_file))
        
        return generated_files


class GoodnessDataGUI:
    """GUI for processing goodness data files."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Goodness Data Processor")
        self.root.geometry("800x600")
        
        self.file_path = None
        self.processor = None
        
        self._create_widgets()
        
    def _create_widgets(self):
        """Create GUI widgets."""
        # File selection frame
        file_frame = ttk.Frame(self.root, padding="10")
        file_frame.pack(fill=tk.X)
        
        ttk.Label(file_frame, text="Data File:").pack(side=tk.LEFT)
        self.file_label = ttk.Label(file_frame, text="No file selected", 
                                    foreground="gray")
        self.file_label.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(file_frame, text="Browse...", 
                  command=self._browse_file).pack(side=tk.RIGHT)
        
        # Options frame
        options_frame = ttk.LabelFrame(self.root, text="Options", padding="10")
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.export_json_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Export as JSON", 
                       variable=self.export_json_var).pack(anchor=tk.W)
        
        self.save_report_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Save report to file", 
                       variable=self.save_report_var).pack(anchor=tk.W)
        
        self.generate_graphs_var = tk.BooleanVar(value=False)
        graph_check = ttk.Checkbutton(options_frame, text="Generate graphs (requires matplotlib)", 
                                     variable=self.generate_graphs_var)
        graph_check.pack(anchor=tk.W)
        if not MATPLOTLIB_AVAILABLE:
            graph_check.config(state='disabled')
        
        # Action buttons
        button_frame = ttk.Frame(self.root, padding="10")
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Process Data", 
                  command=self._process_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", 
                  command=self._clear_output).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", 
                  command=self.root.quit).pack(side=tk.RIGHT, padx=5)
        
        # Output text area
        output_frame = ttk.LabelFrame(self.root, text="Results", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, 
                                                     wrap=tk.WORD,
                                                     width=80, 
                                                     height=20)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                                    relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
    def _browse_file(self):
        """Open file dialog to select data file."""
        file_path = filedialog.askopenfilename(
            title="Select Goodness Data File",
            filetypes=[
                ("Text files", "*.txt"),
                ("TSV files", "*.tsv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.file_path = file_path
            self.file_label.config(text=Path(file_path).name, foreground="black")
            self.status_var.set(f"File selected: {file_path}")
            
    def _clear_output(self):
        """Clear the output text area."""
        self.output_text.delete(1.0, tk.END)
        self.status_var.set("Output cleared")
        
    def _process_data(self):
        """Process the selected data file."""
        if not self.file_path:
            messagebox.showwarning("No File", "Please select a data file first.")
            return
            
        self.status_var.set("Processing...")
        self.root.update()
        
        try:
            # Load and process data
            self.processor = GoodnessDataProcessor(self.file_path)
            self.processor.load_data()
            
            if not self.processor.data:
                messagebox.showerror("Error", "No valid data found in file")
                self.status_var.set("Error: No valid data")
                return
            
            # Generate output
            if self.export_json_var.get():
                # JSON format
                import json
                report = {
                    'summary': self.processor.get_summary_statistics(),
                    'policy_analysis': self.processor.get_policy_analysis(),
                    'c_nodes_transitions': self.processor.get_c_nodes_transitions(),
                    'successful_policies': self.processor.find_successful_policies(),
                    'world_statistics': self.processor.get_world_statistics()
                }
                output = json.dumps(report, indent=2)
                
                if self.save_report_var.get():
                    save_path = filedialog.asksaveasfilename(
                        defaultextension=".json",
                        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
                    )
                    if save_path:
                        with open(save_path, 'w') as f:
                            f.write(output)
                        self.status_var.set(f"Report saved to {save_path}")
                        messagebox.showinfo("Success", f"Report saved to:\n{save_path}")
                else:
                    self.output_text.delete(1.0, tk.END)
                    self.output_text.insert(1.0, output)
                    self.status_var.set("Processing complete")
            else:
                # Text format
                import io
                output_buffer = io.StringIO()
                original_stdout = sys.stdout
                sys.stdout = output_buffer
                
                self.processor.print_report()
                
                sys.stdout = original_stdout
                output = output_buffer.getvalue()
                
                if self.save_report_var.get():
                    save_path = filedialog.asksaveasfilename(
                        defaultextension=".txt",
                        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
                    )
                    if save_path:
                        with open(save_path, 'w') as f:
                            f.write(output)
                        self.status_var.set(f"Report saved to {save_path}")
                        messagebox.showinfo("Success", f"Report saved to:\n{save_path}")
                else:
                    self.output_text.delete(1.0, tk.END)
                    self.output_text.insert(1.0, output)
                    self.status_var.set("Processing complete")
            
            # Generate graphs if requested
            if self.generate_graphs_var.get() and MATPLOTLIB_AVAILABLE:
                graph_dir = filedialog.askdirectory(title="Select Directory to Save Graphs")
                if graph_dir:
                    self.status_var.set("Generating graphs...")
                    self.root.update()
                    generated_files = self.processor.generate_graphs(graph_dir)
                    self.status_var.set(f"Generated {len(generated_files)} graphs")
                    messagebox.showinfo("Graphs Generated", 
                                      f"Generated {len(generated_files)} graphs in:\n{graph_dir}")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Error processing file:\n{str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            
    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()


def main():
    # If no arguments provided, use GUI
    if len(sys.argv) == 1:
        try:
            gui = GoodnessDataGUI()
            gui.run()
        except Exception as e:
            print(f"Error starting GUI: {e}", file=sys.stderr)
            print("Falling back to CLI mode. Use --help for usage information.", file=sys.stderr)
            sys.exit(1)
        return
    
    # CLI mode
    parser = argparse.ArgumentParser(
        description='Process and analyze goodness iteration data from EMDB experiments'
    )
    parser.add_argument('file', type=str, help='Path to the goodness data file (txt/tsv format)')
    parser.add_argument('--output', '-o', type=str, help='Output file for the report')
    parser.add_argument('--format', '-f', choices=['text', 'json'], default='text',
                       help='Output format (default: text)')
    parser.add_argument('--graphs', '-g', type=str, metavar='DIR',
                       help='Generate graphs and save to specified directory')
    
    args = parser.parse_args()
    
    # Process data
    processor = GoodnessDataProcessor(args.file)
    
    try:
        processor.load_data()
    except Exception as e:
        print(f"Error loading file: {e}", file=sys.stderr)
        sys.exit(1)
        
    if not processor.data:
        print("No valid data found in file", file=sys.stderr)
        sys.exit(1)
    
    # Generate report
    if args.format == 'json':
        import json
        report = {
            'summary': processor.get_summary_statistics(),
            'policy_analysis': processor.get_policy_analysis(),
            'c_nodes_transitions': processor.get_c_nodes_transitions(),
            'successful_policies': processor.find_successful_policies(),
            'world_statistics': processor.get_world_statistics()
        }
        output = json.dumps(report, indent=2)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
        else:
            print(output)
    else:
        # Redirect stdout if output file specified
        if args.output:
            original_stdout = sys.stdout
            with open(args.output, 'w') as f:
                sys.stdout = f
                processor.print_report()
                sys.stdout = original_stdout
            print(f"Report saved to {args.output}")
        else:
            processor.print_report()
    
    # Generate graphs if requested
    if args.graphs:
        if MATPLOTLIB_AVAILABLE:
            print(f"\nGenerating graphs in {args.graphs}...")
            generated_files = processor.generate_graphs(args.graphs)
            print(f"Generated {len(generated_files)} graphs:")
            for f in generated_files:
                print(f"  - {f}")
        else:
            print("Error: matplotlib is required for graph generation.", file=sys.stderr)
            print("Install it with: pip install matplotlib", file=sys.stderr)


if __name__ == '__main__':
    main()
