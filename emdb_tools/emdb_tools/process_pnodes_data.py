#!/usr/bin/env python3
"""
Script to process and visualize pnodes (predictive nodes) data from EMDB experiments.
Can be used with GUI (no arguments) or CLI (with arguments).
"""

import argparse
import re
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


class PnodesDataProcessor:
    """Process and analyze pnodes data."""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.data = []
        self.headers = []
        self.feature_columns = []
        
    def load_data(self):
        """Load data from text file (tab-separated values)."""
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            raise ValueError("Empty file")
        
        # Parse headers
        self.headers = [h.strip() for h in lines[0].split('\t')]
        
        # Identify feature columns (exclude Iteration, Ident, Confidence)
        self.feature_columns = [h for h in self.headers 
                               if h not in ['Iteration', 'Ident', 'Confidence']]
        
        # Parse data rows
        for line in lines[1:]:
            if not line.strip():
                continue
                
            parts = line.strip().split('\t')
            if len(parts) == len(self.headers):
                try:
                    row = {'iteration': int(parts[0]), 'ident': parts[1]}
                    
                    # Parse feature values
                    for i, header in enumerate(self.headers[2:], start=2):
                        if header == 'Confidence':
                            row['confidence'] = float(parts[i])
                        else:
                            row[header] = float(parts[i])
                    
                    self.data.append(row)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping malformed line: {line.strip()[:100]}...", file=sys.stderr)
    
    def extract_pnode_info(self, ident: str) -> Dict[str, str]:
        """Extract client, category, and policy from pnode identifier."""
        # Example: pnode_client_0_33__goal_0__prepare_drink
        parts = ident.split('__')
        
        info = {
            'client': parts[0] if len(parts) > 0 else 'unknown',
            'category': 'unknown',
            'policy': 'unknown'
        }
        
        if len(parts) >= 3:
            info['category'] = parts[1]  # e.g., goal_0, effect_bottle_in_right_hand_data
            info['policy'] = parts[2]     # e.g., prepare_drink, pick_bottle
        
        # Extract category type (goal, effect, etc.)
        if 'goal' in info['category']:
            info['category_type'] = 'goal'
        elif 'effect' in info['category']:
            info['category_type'] = 'effect'
        else:
            info['category_type'] = 'other'
            
        return info
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of the dataset."""
        if not self.data:
            return {}
        
        iterations = sorted(set(row['iteration'] for row in self.data))
        unique_pnodes = set(row['ident'] for row in self.data)
        
        positive_conf = sum(1 for row in self.data if row['confidence'] > 0)
        negative_conf = sum(1 for row in self.data if row['confidence'] < 0)
        neutral_conf = sum(1 for row in self.data if row['confidence'] == 0)
        
        # Category analysis
        category_types = []
        for row in self.data:
            info = self.extract_pnode_info(row['ident'])
            category_types.append(info['category_type'])
        
        return {
            'total_rows': len(self.data),
            'iterations': iterations,
            'num_iterations': len(iterations),
            'unique_pnodes': len(unique_pnodes),
            'positive_confidence': positive_conf,
            'negative_confidence': negative_conf,
            'neutral_confidence': neutral_conf,
            'positive_percentage': (positive_conf / len(self.data)) * 100,
            'category_distribution': Counter(category_types)
        }
    
    def get_pnodes_per_iteration(self) -> Dict[int, int]:
        """Get number of pnodes per iteration."""
        pnodes_per_iter = defaultdict(set)
        for row in self.data:
            pnodes_per_iter[row['iteration']].add(row['ident'])
        
        return {iter_: len(pnodes) for iter_, pnodes in pnodes_per_iter.items()}
    
    def get_confidence_evolution(self) -> Dict[int, Dict]:
        """Get confidence distribution per iteration."""
        conf_per_iter = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
        
        for row in self.data:
            if row['confidence'] > 0:
                conf_per_iter[row['iteration']]['positive'] += 1
            elif row['confidence'] < 0:
                conf_per_iter[row['iteration']]['negative'] += 1
            else:
                conf_per_iter[row['iteration']]['neutral'] += 1
        
        return dict(conf_per_iter)
    
    def get_top_pnodes(self, n: int = 10) -> List[Tuple[str, int]]:
        """Get the most frequent pnodes."""
        pnode_counter = Counter(row['ident'] for row in self.data)
        return pnode_counter.most_common(n)
    
    def get_pnode_types_evolution(self) -> Dict[int, Dict]:
        """Get evolution of pnode types (goal, effect, etc.) per iteration."""
        types_per_iter = defaultdict(lambda: Counter())
        
        for row in self.data:
            info = self.extract_pnode_info(row['ident'])
            types_per_iter[row['iteration']][info['category_type']] += 1
        
        return dict(types_per_iter)
    
    def get_policy_analysis(self) -> Dict[str, Dict]:
        """Analyze policies in pnodes."""
        policy_data = defaultdict(lambda: {'count': 0, 'positive_conf': 0, 'negative_conf': 0})
        
        for row in self.data:
            info = self.extract_pnode_info(row['ident'])
            policy = info['policy']
            policy_data[policy]['count'] += 1
            
            if row['confidence'] > 0:
                policy_data[policy]['positive_conf'] += 1
            elif row['confidence'] < 0:
                policy_data[policy]['negative_conf'] += 1
        
        # Calculate confidence rates
        for policy, data in policy_data.items():
            data['positive_rate'] = (data['positive_conf'] / data['count']) * 100 if data['count'] > 0 else 0
        
        return dict(policy_data)
    
    def print_report(self):
        """Print a comprehensive analysis report."""
        print("=" * 80)
        print("PNODES DATA ANALYSIS")
        print("=" * 80)
        print()
        
        # Summary statistics
        print("SUMMARY STATISTICS")
        print("-" * 80)
        summary = self.get_summary_statistics()
        print(f"Total rows: {summary['total_rows']}")
        print(f"Iterations: {summary['num_iterations']} ({min(summary['iterations'])} - {max(summary['iterations'])})")
        print(f"Unique pnodes: {summary['unique_pnodes']}")
        print(f"Confidence distribution:")
        print(f"  Positive: {summary['positive_confidence']} ({summary['positive_percentage']:.1f}%)")
        print(f"  Negative: {summary['negative_confidence']} ({100 - summary['positive_percentage']:.1f}%)")
        if summary['neutral_confidence'] > 0:
            print(f"  Neutral: {summary['neutral_confidence']}")
        print(f"Category types: {dict(summary['category_distribution'])}")
        print()
        
        # Pnodes per iteration
        print("PNODES EVOLUTION")
        print("-" * 80)
        pnodes_per_iter = self.get_pnodes_per_iteration()
        iters = sorted(pnodes_per_iter.keys())
        print(f"Iteration range: {min(iters)} - {max(iters)}")
        print(f"Average pnodes per iteration: {sum(pnodes_per_iter.values()) / len(pnodes_per_iter):.1f}")
        print(f"Max pnodes in single iteration: {max(pnodes_per_iter.values())}")
        print(f"Min pnodes in single iteration: {min(pnodes_per_iter.values())}")
        print()
        
        # Top pnodes
        print("TOP 10 MOST FREQUENT PNODES")
        print("-" * 80)
        top_pnodes = self.get_top_pnodes(10)
        for i, (pnode, count) in enumerate(top_pnodes, 1):
            print(f"{i}. {pnode}: {count} occurrences")
        print()
        
        # Policy analysis
        print("POLICY ANALYSIS")
        print("-" * 80)
        policy_analysis = self.get_policy_analysis()
        sorted_policies = sorted(policy_analysis.items(), 
                                key=lambda x: x[1]['count'], 
                                reverse=True)
        
        print(f"Total unique policies: {len(sorted_policies)}")
        print("\nTop 10 policies by occurrence:")
        for i, (policy, data) in enumerate(sorted_policies[:10], 1):
            print(f"{i}. {policy}:")
            print(f"   Count: {data['count']}")
            print(f"   Positive confidence: {data['positive_conf']} ({data['positive_rate']:.1f}%)")
        print()
    
    def generate_graphs(self, output_dir: str = "."):
        """Generate visualization graphs."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot generate graphs.", file=sys.stderr)
            return []
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        generated_files = []
        
        # Graph 1: Pnodes evolution over iterations
        fig, ax = plt.subplots(figsize=(12, 6))
        pnodes_per_iter = self.get_pnodes_per_iteration()
        iters = sorted(pnodes_per_iter.keys())
        counts = [pnodes_per_iter[i] for i in iters]
        
        ax.plot(iters, counts, 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Number of Pnodes', fontsize=12)
        ax.set_title('Pnodes Evolution Over Iterations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        graph_file = output_path / 'pnodes_evolution.png'
        plt.tight_layout()
        plt.savefig(graph_file, dpi=150)
        plt.close()
        generated_files.append(str(graph_file))
        
        # Graph 2: Confidence distribution stacked over iterations
        fig, ax = plt.subplots(figsize=(12, 6))
        conf_evolution = self.get_confidence_evolution()
        iters = sorted(conf_evolution.keys())
        positive = [conf_evolution[i]['positive'] for i in iters]
        negative = [conf_evolution[i]['negative'] for i in iters]
        
        ax.bar(iters, positive, label='Positive confidence', color='green', alpha=0.7)
        ax.bar(iters, negative, bottom=positive, label='Negative confidence', color='red', alpha=0.7)
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Confidence Distribution Over Iterations', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        graph_file = output_path / 'confidence_distribution.png'
        plt.tight_layout()
        plt.savefig(graph_file, dpi=150)
        plt.close()
        generated_files.append(str(graph_file))
        
        # Graph 3: Pnode types evolution
        fig, ax = plt.subplots(figsize=(12, 6))
        types_evolution = self.get_pnode_types_evolution()
        iters = sorted(types_evolution.keys())
        
        # Get all category types
        all_types = set()
        for data in types_evolution.values():
            all_types.update(data.keys())
        
        # Plot each type
        colors = {'goal': 'blue', 'effect': 'orange', 'other': 'gray'}
        for cat_type in sorted(all_types):
            counts = [types_evolution[i].get(cat_type, 0) for i in iters]
            ax.plot(iters, counts, label=cat_type.capitalize(), 
                   linewidth=2, marker='o', markersize=4,
                   color=colors.get(cat_type, 'purple'))
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Pnode Types Evolution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        graph_file = output_path / 'pnode_types_evolution.png'
        plt.tight_layout()
        plt.savefig(graph_file, dpi=150)
        plt.close()
        generated_files.append(str(graph_file))
        
        # Graph 4: Top policies by positive confidence rate
        fig, ax = plt.subplots(figsize=(12, 8))
        policy_analysis = self.get_policy_analysis()
        
        # Filter policies with at least 10 occurrences
        filtered_policies = {k: v for k, v in policy_analysis.items() if v['count'] >= 10}
        
        if filtered_policies:
            sorted_policies = sorted(filtered_policies.items(), 
                                    key=lambda x: x[1]['positive_rate'], 
                                    reverse=True)[:15]
            
            policies = [p[0] for p in sorted_policies]
            rates = [p[1]['positive_rate'] for p in sorted_policies]
            counts = [p[1]['count'] for p in sorted_policies]
            
            colors_list = ['green' if rate > 50 else 'orange' if rate > 30 else 'red' for rate in rates]
            bars = ax.barh(range(len(policies)), rates, color=colors_list, alpha=0.7)
            
            # Add count labels
            for i, (rate, count) in enumerate(zip(rates, counts)):
                ax.text(rate + 1, i, f'n={count}', va='center', fontsize=9)
            
            ax.set_yticks(range(len(policies)))
            ax.set_yticklabels(policies, fontsize=10)
            ax.set_xlabel('Positive Confidence Rate (%)', fontsize=12)
            ax.set_title('Top Policies by Positive Confidence Rate (min 10 occurrences)', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)
        
        graph_file = output_path / 'policy_confidence_rates.png'
        plt.tight_layout()
        plt.savefig(graph_file, dpi=150)
        plt.close()
        generated_files.append(str(graph_file))
        
        # Graph 5: Confidence ratio over iterations
        fig, ax = plt.subplots(figsize=(12, 6))
        conf_evolution = self.get_confidence_evolution()
        iters = sorted(conf_evolution.keys())
        ratios = []
        
        for i in iters:
            total = conf_evolution[i]['positive'] + conf_evolution[i]['negative']
            ratio = (conf_evolution[i]['positive'] / total * 100) if total > 0 else 0
            ratios.append(ratio)
        
        ax.plot(iters, ratios, 'g-', linewidth=2, marker='o', markersize=4)
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Positive Confidence Ratio (%)', fontsize=12)
        ax.set_title('Positive Confidence Ratio Over Iterations', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        graph_file = output_path / 'confidence_ratio.png'
        plt.tight_layout()
        plt.savefig(graph_file, dpi=150)
        plt.close()
        generated_files.append(str(graph_file))
        
        return generated_files
    
    def generate_individual_pnode_graphs(self, output_dir: str = ".", max_pnodes: int = None):
        """Generate individual confidence evolution graphs for each pnode.
        
        Args:
            output_dir: Directory to save graphs
            max_pnodes: Maximum number of pnodes to generate graphs for (None = all)
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot generate graphs.", file=sys.stderr)
            return []
        
        output_path = Path(output_dir) / "individual_pnodes"
        output_path.mkdir(exist_ok=True, parents=True)
        generated_files = []
        
        # Group data by pnode
        pnode_data = defaultdict(list)
        for row in self.data:
            pnode_data[row['ident']].append({
                'iteration': row['iteration'],
                'confidence': row['confidence']
            })
        
        # Sort pnodes by frequency (most common first)
        pnode_counts = Counter(row['ident'] for row in self.data)
        sorted_pnodes = [pnode for pnode, _ in pnode_counts.most_common(max_pnodes)]
        
        print(f"Generating individual graphs for {len(sorted_pnodes)} pnodes...", file=sys.stderr)
        
        for idx, pnode in enumerate(sorted_pnodes, 1):
            if idx % 5 == 0:
                print(f"  Progress: {idx}/{len(sorted_pnodes)}", file=sys.stderr)
            
            data = pnode_data[pnode]
            
            # Group by iteration and calculate positive confidence percentage
            iter_confidence = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
            for d in data:
                if d['confidence'] > 0:
                    iter_confidence[d['iteration']]['positive'] += 1
                elif d['confidence'] < 0:
                    iter_confidence[d['iteration']]['negative'] += 1
                else:
                    iter_confidence[d['iteration']]['neutral'] += 1
            
            # Calculate percentage per iteration
            iterations = sorted(iter_confidence.keys())
            percentages = []
            
            for iter_num in iterations:
                counts = iter_confidence[iter_num]
                total = counts['positive'] + counts['negative'] + counts['neutral']
                percentage = (counts['positive'] / total * 100) if total > 0 else 0
                percentages.append(percentage)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot line
            ax.plot(iterations, percentages, linewidth=2, color='blue', marker='o', 
                   markersize=4, markerfacecolor='blue', markeredgewidth=0.5, 
                   markeredgecolor='darkblue')
            
            # Add reference lines
            ax.axhline(y=50, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='50% threshold')
            ax.axhline(y=80, color='green', linestyle='--', linewidth=1, alpha=0.3, label='80% threshold')
            
            # Fill area under curve
            ax.fill_between(iterations, percentages, alpha=0.3, color='blue')
            
            # Calculate overall statistics
            total_positive = sum(iter_confidence[i]['positive'] for i in iterations)
            total_negative = sum(iter_confidence[i]['negative'] for i in iterations)
            total_count = total_positive + total_negative + sum(iter_confidence[i]['neutral'] for i in iterations)
            overall_percentage = (total_positive / total_count * 100) if total_count > 0 else 0
            
            # Extract pnode info for title
            info = self.extract_pnode_info(pnode)
            
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Positive Confidence (%)', fontsize=12)
            ax.set_title(f'{info["policy"]} ({info["category_type"]})\n'
                        f'Overall: {overall_percentage:.1f}% positive ({total_positive}/{total_count} samples)',
                        fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            
            # Save with sanitized filename
            safe_filename = pnode.replace('/', '_').replace('\\', '_')
            graph_file = output_path / f'{safe_filename}.png'
            plt.tight_layout()
            plt.savefig(graph_file, dpi=150)
            plt.close()
            generated_files.append(str(graph_file))
        
        print(f"Generated {len(generated_files)} individual pnode graphs", file=sys.stderr)
        return generated_files


class PnodesDataGUI:
    """GUI for processing pnodes data files."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Pnodes Data Processor")
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
        
        self.generate_individual_var = tk.BooleanVar(value=False)
        individual_check = ttk.Checkbutton(options_frame, text="Generate individual pnode graphs (can be slow)", 
                                          variable=self.generate_individual_var)
        individual_check.pack(anchor=tk.W)
        if not MATPLOTLIB_AVAILABLE:
            individual_check.config(state='disabled')
        
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
            title="Select Pnodes Data File",
            filetypes=[
                ("Text files", "*.txt"),
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
            self.processor = PnodesDataProcessor(self.file_path)
            self.status_var.set("Loading data...")
            self.root.update()
            
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
                    'pnodes_per_iteration': self.processor.get_pnodes_per_iteration(),
                    'confidence_evolution': self.processor.get_confidence_evolution(),
                    'top_pnodes': self.processor.get_top_pnodes(20),
                    'policy_analysis': self.processor.get_policy_analysis()
                }
                # Convert sets to lists for JSON serialization
                if 'iterations' in report['summary']:
                    report['summary']['iterations'] = list(report['summary']['iterations'])
                
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
            
            # Generate individual pnode graphs if requested
            if self.generate_individual_var.get() and MATPLOTLIB_AVAILABLE:
                if not hasattr(self, 'graph_dir') or not graph_dir:
                    graph_dir = filedialog.askdirectory(title="Select Directory to Save Individual Pnode Graphs")
                
                if graph_dir:
                    self.status_var.set("Generating individual pnode graphs...")
                    self.root.update()
                    
                    # Ask for max pnodes limit
                    max_pnodes = messagebox.askquestion(
                        "Limit Number of Graphs",
                        f"Found {len(set(row['ident'] for row in self.processor.data))} unique pnodes.\n"
                        "Generate graphs for all pnodes?\n\n"
                        "(Click 'No' to limit to top 20)",
                        icon='question'
                    )
                    
                    max_limit = None if max_pnodes == 'yes' else 20
                    
                    generated_files = self.processor.generate_individual_pnode_graphs(graph_dir, max_limit)
                    self.status_var.set(f"Generated {len(generated_files)} individual graphs")
                    messagebox.showinfo("Individual Graphs Generated", 
                                      f"Generated {len(generated_files)} individual pnode graphs in:\n"
                                      f"{graph_dir}/individual_pnodes/")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Error processing file:\n{str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()


def main():
    # If no arguments provided, use GUI
    if len(sys.argv) == 1:
        try:
            gui = PnodesDataGUI()
            gui.run()
        except Exception as e:
            print(f"Error starting GUI: {e}", file=sys.stderr)
            print("Falling back to CLI mode. Use --help for usage information.", file=sys.stderr)
            sys.exit(1)
        return
    
    # CLI mode
    parser = argparse.ArgumentParser(
        description='Process and analyze pnodes data from EMDB experiments'
    )
    parser.add_argument('file', type=str, help='Path to the pnodes data file (txt format)')
    parser.add_argument('--output', '-o', type=str, help='Output file for the report')
    parser.add_argument('--format', '-f', choices=['text', 'json'], default='text',
                       help='Output format (default: text)')
    parser.add_argument('--graphs', '-g', type=str, metavar='DIR',
                       help='Generate graphs and save to specified directory')
    parser.add_argument('--individual', '-i', action='store_true',
                       help='Generate individual confidence evolution graphs for each pnode')
    parser.add_argument('--max-pnodes', '-m', type=int, default=None,
                       help='Maximum number of individual pnode graphs to generate (default: all)')
    
    args = parser.parse_args()
    
    # Process data
    processor = PnodesDataProcessor(args.file)
    
    try:
        print("Loading data...", file=sys.stderr)
        processor.load_data()
        print(f"Loaded {len(processor.data)} rows", file=sys.stderr)
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
            'pnodes_per_iteration': processor.get_pnodes_per_iteration(),
            'confidence_evolution': processor.get_confidence_evolution(),
            'top_pnodes': processor.get_top_pnodes(20),
            'policy_analysis': processor.get_policy_analysis()
        }
        # Convert sets to lists for JSON serialization
        if 'iterations' in report['summary']:
            report['summary']['iterations'] = list(report['summary']['iterations'])
        
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
            print(f"\nGenerating graphs in {args.graphs}...", file=sys.stderr)
            generated_files = processor.generate_graphs(args.graphs)
            print(f"Generated {len(generated_files)} graphs:", file=sys.stderr)
            for f in generated_files:
                print(f"  - {f}", file=sys.stderr)
        else:
            print("Error: matplotlib is required for graph generation.", file=sys.stderr)
            print("Install it with: pip install matplotlib", file=sys.stderr)
    
    # Generate individual pnode graphs if requested
    if args.individual:
        if MATPLOTLIB_AVAILABLE:
            # Use graphs directory if specified, otherwise current directory
            output_dir = args.graphs if args.graphs else '.'
            print(f"\nGenerating individual pnode graphs in {output_dir}...", file=sys.stderr)
            
            if args.max_pnodes:
                print(f"Limiting to top {args.max_pnodes} pnodes", file=sys.stderr)
            
            generated_files = processor.generate_individual_pnode_graphs(output_dir, args.max_pnodes)
            print(f"Generated {len(generated_files)} individual pnode graphs:", file=sys.stderr)
            print(f"  Location: {output_dir}/individual_pnodes/", file=sys.stderr)
        else:
            print("Error: matplotlib is required for graph generation.", file=sys.stderr)
            print("Install it with: pip install matplotlib", file=sys.stderr)


if __name__ == '__main__':
    main()
