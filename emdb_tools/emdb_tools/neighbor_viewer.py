import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from collections import defaultdict
import ast
import json
import webbrowser
from pathlib import Path

class NeighborViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Neighbor Relationships Viewer")
        self.root.geometry("800x600")
        
        # Frame para controles
        control_frame = tk.Frame(root)
        control_frame.pack(pady=10)
        
        tk.Button(control_frame, text="Load .txt File", command=self.load_file).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Show Interactive Graph", command=self.show_graph).pack(side=tk.LEFT, padx=5)
        
        # Área de texto para mostrar relaciones
        tk.Label(root, text="Relationships:").pack(anchor=tk.W, padx=10)
        self.text_area = scrolledtext.ScrolledText(root, height=20, width=80)
        self.text_area.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        self.neighbors = defaultdict(set)
        self.node_types = {}
        self.node_first_iteration = {}
        self.edge_first_iteration = {}
        self.max_iteration = 0
        self.file_path = None
    
    def load_file(self):
        self.file_path = filedialog.askopenfilename(
            filetypes=[("TXT Files", "*.txt"), ("All Files", "*.*")]
        )
        if self.file_path:
            try:
                self.parse_file(self.file_path)
                messagebox.showinfo("Success", f"File loaded: {self.file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading file: {str(e)}")
    
    def parse_file(self, filepath):
        self.neighbors.clear()
        self.node_types.clear()
        self.node_first_iteration.clear()
        self.edge_first_iteration.clear()
        self.max_iteration = 0

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '\t' in line:
                        self._parse_tabbed_line(line)
                    else:
                        self._parse_plain_line(line)

        if not self.neighbors:
            raise ValueError("No valid relations were found in the selected file")

        self.display_relationships()

    def _parse_tabbed_line(self, line):
        columns = [col.strip() for col in line.split('\t')]
        if not columns:
            return

        first_col = columns[0].lower()

        # Header rows
        if first_col in {"iteration", "goal"}:
            return

        # New format: Iteration \t Goal \t Neighbors
        if len(columns) >= 3 and columns[0].isdigit():
            iteration = int(columns[0])
            source = columns[1]
            if not source:
                return

            self.node_types.setdefault(source, "Goal")
            self._register_node_iteration(source, iteration)
            for target, node_type in self._extract_targets(columns[2]):
                self._add_relation(source, target, iteration=iteration)
                self.node_types.setdefault(target, node_type)
            return

        # Legacy format: Goal \t Neighbor1 \t Neighbor2 ...
        source = columns[0]
        if not source:
            return

        self.node_types.setdefault(source, "Goal")
        self._register_node_iteration(source, 0)
        for cell in columns[1:]:
            for target, node_type in self._extract_targets(cell):
                self._add_relation(source, target, iteration=0)
                self.node_types.setdefault(target, node_type)

    def _parse_plain_line(self, line):
        parts = line.split()
        if len(parts) < 2:
            return

        node1, node2 = parts[0], parts[1]
        self._add_relation(node1, node2, iteration=0)
        self.node_types.setdefault(node1, "Node")
        self.node_types.setdefault(node2, "Node")

    def _extract_targets(self, text):
        value = text.strip()
        if not value:
            return []

        if value.startswith('[') and value.endswith(']'):
            try:
                parsed = ast.literal_eval(value)
                targets = []
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            name = str(item.get("name", "")).strip()
                            node_type = str(item.get("node_type", "Node")).strip() or "Node"
                            if name:
                                targets.append((name, node_type))
                        elif isinstance(item, str):
                            clean_name = item.strip()
                            if clean_name:
                                targets.append((clean_name, "Node"))
                return targets
            except (SyntaxError, ValueError):
                return []

        return [(value, "Node")]

    def _register_node_iteration(self, node, iteration):
        prev = self.node_first_iteration.get(node)
        if prev is None or iteration < prev:
            self.node_first_iteration[node] = iteration
        if iteration > self.max_iteration:
            self.max_iteration = iteration

    def _add_relation(self, node1, node2, iteration=0):
        if not node1 or not node2 or node1 == node2:
            return

        self.neighbors[node1].add(node2)
        self.neighbors[node2].add(node1)
        self._register_node_iteration(node1, iteration)
        self._register_node_iteration(node2, iteration)

        edge_key = tuple(sorted((node1, node2)))
        prev_edge_iter = self.edge_first_iteration.get(edge_key)
        if prev_edge_iter is None or iteration < prev_edge_iter:
            self.edge_first_iteration[edge_key] = iteration
    
    def display_relationships(self):
        self.text_area.delete(1.0, tk.END)
        for node, related in sorted(self.neighbors.items()):
            self.text_area.insert(tk.END, f"{node}: {', '.join(sorted(related))}\n")

    def _render_interactive_html(self):
        if not self.file_path:
            raise ValueError("No input file loaded")

        nodes = []
        for node in sorted(self.node_types):
            node_type = self.node_types.get(node, "Node")
            nodes.append(
                {
                    "id": node,
                    "label": node,
                    "title": f"Type: {node_type}",
                    "group": node_type,
                    "firstIteration": int(self.node_first_iteration.get(node, 0)),
                    "hidden": True,
                }
            )

        edges = []
        seen = set()
        for source, related_nodes in self.neighbors.items():
            for target in related_nodes:
                edge_key = tuple(sorted((source, target)))
                if edge_key in seen:
                    continue
                seen.add(edge_key)
                edges.append(
                    {
                        "from": source,
                        "to": target,
                        "firstIteration": int(self.edge_first_iteration.get(edge_key, 0)),
                        "hidden": True,
                    }
                )

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Interactive Neighbor Graph</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; }}
        #toolbar {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        #network {{ width: 100vw; height: calc(100vh - 52px); }}
        #search {{ width: 320px; padding: 6px; }}
        #timeline {{ width: 260px; margin-left: 10px; vertical-align: middle; }}
        #iterLabel {{ margin-left: 8px; font-weight: bold; }}
        button {{ padding: 6px 10px; margin-left: 6px; }}
    </style>
</head>
<body>
    <div id="toolbar">
        <input id="search" type="text" placeholder="Search node name..." />
        <button id="focusBtn">Focus</button>
        <button id="resetBtn">Reset</button>
        <button id="playBtn">Play</button>
        <input id="timeline" type="range" min="0" max="{int(self.max_iteration)}" step="1" value="0" />
        <span id="iterLabel">Iteration: 0</span>
    </div>
    <div id="network"></div>

    <script>
        const nodes = new vis.DataSet({json.dumps(nodes, ensure_ascii=False)});
        const edges = new vis.DataSet({json.dumps(edges, ensure_ascii=False)});
        const container = document.getElementById('network');
        const data = {{ nodes, edges }};
        const options = {{
            interaction: {{ hover: true, navigationButtons: true, keyboard: true, multiselect: true }},
            physics: {{
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{ gravitationalConstant: -60, springLength: 130, damping: 0.5 }},
                stabilization: {{ iterations: 200 }}
            }},
            nodes: {{ shape: 'dot', size: 14, font: {{ size: 14 }} }},
            edges: {{ color: {{ color: '#777' }}, smooth: {{ type: 'dynamic' }} }}
        }};

        const network = new vis.Network(container, data, options);
        const maxIteration = {int(self.max_iteration)};
        const timeline = document.getElementById('timeline');
        const iterLabel = document.getElementById('iterLabel');
        const playBtn = document.getElementById('playBtn');
        let playTimer = null;
        const eventIterations = (() => {{
            const fromNodes = nodes.get().map(n => Number(n.firstIteration || 0));
            const uniqueSorted = [...new Set([0, ...fromNodes])].sort((a, b) => a - b);
            return uniqueSorted;
        }})();

        function applyIteration(iteration) {{
            iterLabel.textContent = `Iteration: ${{iteration}}`;

            const allNodes = nodes.get();
            const nodeUpdates = allNodes.map(node => ({{
                id: node.id,
                hidden: Number(node.firstIteration || 0) > iteration
            }}));
            nodes.update(nodeUpdates);

            const allEdges = edges.get();
            const edgeUpdates = allEdges.map(edge => ({{
                id: edge.id,
                hidden: Number(edge.firstIteration || 0) > iteration
            }}));
            edges.update(edgeUpdates);
        }}

        function stopPlayback() {{
            if (playTimer) {{
                clearInterval(playTimer);
                playTimer = null;
                playBtn.textContent = 'Play';
            }}
        }}

        function nextEventIteration(currentIteration) {{
            for (const it of eventIterations) {{
                if (it > currentIteration) return it;
            }}
            return null;
        }}

        function startPlayback() {{
            if (playTimer) return;
            playBtn.textContent = 'Pause';
            playTimer = setInterval(() => {{
                const current = Number(timeline.value);
                const nextIteration = nextEventIteration(current);
                if (nextIteration === null || current >= maxIteration) {{
                    stopPlayback();
                    return;
                }}
                timeline.value = String(nextIteration);
                applyIteration(nextIteration);
            }}, 120);
        }}

        timeline.addEventListener('input', () => {{
            applyIteration(Number(timeline.value));
        }});

        playBtn.addEventListener('click', () => {{
            if (playTimer) stopPlayback();
            else startPlayback();
        }});

        applyIteration(0);

        document.getElementById('focusBtn').addEventListener('click', () => {{
            const search = document.getElementById('search').value.trim().toLowerCase();
            if (!search) return;
            const allNodes = nodes.get();
            const match = allNodes.find(n => n.label.toLowerCase().includes(search));
            if (!match) return;

            network.selectNodes([match.id]);
            network.focus(match.id, {{ scale: 1.4, animation: true }});
        }});

        document.getElementById('resetBtn').addEventListener('click', () => {{
            network.unselectAll();
            network.fit({{ animation: true }});
        }});
    </script>
</body>
</html>'''

        source_path = Path(self.file_path)
        output_html = source_path.parent / f"{source_path.stem}_interactive_graph.html"
        output_html.write_text(html, encoding="utf-8")
        return output_html
    
    def show_graph(self):
        if not self.neighbors:
            messagebox.showwarning("Warning", "No data loaded")
            return

        html_path = self._render_interactive_html()
        webbrowser.open(f"file://{html_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = NeighborViewer(root)
    root.mainloop()