import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional

class HEPExplainer:
    """
    Explainer and Visualizer for Hypergraph-Evolved Pipelines (HEP).
    Provides tools for structural visualization and evolution analysis.
    """
    
    def __init__(self, output_dir: str = 'hep_plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Use a high-quality style
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.dpi'] = 150

    def plot_hypergraph(self, hep_node: Any, title: str = "Evolved Hypergraph Structure", save_name: str = "hypergraph.png"):
        """
        Renders a hypergraph using Euler-style blobs around nodes.
        """
        params = hep_node.parameters
        hyperedges = params.get('hyperedges', [])
        num_vertices = params.get('num_vertices', 1)
        
        if not hyperedges:
            print("No hyperedges to plot.")
            return

        # Create a graph for layout
        G = nx.Graph()
        G.add_nodes_from(range(num_vertices))
        
        # Add edges between nodes that share ANY hyperedge to guide layout
        for edge in hyperedges:
            comps = edge['components']
            for i in range(len(comps)):
                for j in range(i + 1, len(comps)):
                    G.add_edge(comps[i], comps[j])

        # Use spring layout
        pos = nx.spring_layout(G, seed=42, k=1/np.sqrt(num_vertices)*2)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw base nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightgrey', alpha=0.8, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)

        # Draw hyperedges as convex hulls or smooth blobs
        colors = plt.cm.rainbow(np.linspace(0, 1, len(hyperedges)))
        
        for i, (edge, color) in enumerate(zip(hyperedges, colors)):
            comps = edge['components']
            func = edge['func']
            
            # Extract coordinates for components
            points = np.array([pos[c] for c in comps])
            
            if len(points) >= 3:
                # Simple blob: average + radius? 
                # Better: Draw a translucent polygon or circle
                center = points.mean(axis=0)
                radius = np.max(np.linalg.norm(points - center, axis=1)) + 0.1
                circle = plt.Circle(center, radius, color=color, alpha=0.2, label=f"E{i}: {func}")
                ax.add_patch(circle)
            elif len(points) == 2:
                # Draw a thick line
                ax.plot(points[:, 0], points[:, 1], color=color, linewidth=10, alpha=0.2, solid_capstyle='round')

        ax.set_title(title, fontsize=16)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        
        path = os.path.join(self.output_dir, save_name)
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        print(f"Hypergraph plot saved to {path}")

    def plot_edge_survival(self, history: Any, save_name: str = "edge_survival.html"):
        """
        Visualizes the persistence of hyperedges across generations.
        """
        data = []
        # unique_edges stores (generation_range, edge_repr)
        # We define an edge by its components (sorted) and func
        edge_tracker = {} # Key: repr, Value: List of generations
        
        for gen_idx, generation in enumerate(history.generations):
            # Find the best individual in generation or all? 
            # For survival plot, usually we look at the whole population or the best lineages.
            # Let's look at all unique edges in the population per generation.
            current_gen_edges = set()
            for ind in generation:
                # Find HEP node
                for node in ind.graph.nodes:
                    if 'hypergraph' in str(node.name):
                        for edge in node.parameters.get('hyperedges', []):
                            edge_repr = f"{sorted(edge['components'])}_{edge['func']}"
                            current_gen_edges.add(edge_repr)
            
            for ed in current_gen_edges:
                if ed not in edge_tracker:
                    edge_tracker[ed] = []
                edge_tracker[ed].append(gen_idx)

        # Create plot data
        for i, (edge_repr, gens) in enumerate(edge_tracker.items()):
            # Find contiguous segments of generations
            if not gens: continue
            
            segments = []
            curr_seg = [gens[0]]
            for g in gens[1:]:
                if g == curr_seg[-1] + 1:
                    curr_seg.append(g)
                else:
                    segments.append(curr_seg)
                    curr_seg = [g]
            segments.append(curr_seg)
            
            for seg in segments:
                data.append(dict(
                    Edge=edge_repr,
                    Start=seg[0],
                    End=seg[-1],
                    Duration=len(seg)
                ))

        df = pd.DataFrame(data)
        if df.empty:
            print("No history data found for survival plot.")
            return

        fig = go.Figure()
        
        # Sort by duration to show important ones at the top
        df = df.sort_values('Duration', ascending=True)
        
        for idx, row in df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['Start'], row['End']],
                y=[row['Edge'], row['Edge']],
                mode='lines+markers',
                name=row['Edge'],
                line=dict(width=4),
                marker=dict(size=8),
                hovertemplate=f"Edge: {row['Edge']}<br>Gens: {row['Start']}-{row['End']}<br>Duration: {row['Duration']}"
            ))

        fig.update_layout(
            title="Hyperedge Survival Plot (Persistence across Generations)",
            xaxis_title="Generation",
            yaxis_title="Hyperedge (Formula)",
            showlegend=False,
            height=200 + 20 * len(df['Edge'].unique()),
            template="plotly_white"
        )
        
        path = os.path.join(self.output_dir, save_name)
        fig.write_html(path)
        print(f"Edge survival plot saved to {path}")

    def plot_feature_importance(self, history: Any, feature_names: List[str] = None):
        """
        Plots feature importance based on participation frequency in survived hyperedges.
        """
        participation = {}
        
        # We look at the top individuals in the final generation or throughout history
        for generation in history.generations:
            for ind in generation:
                for node in ind.graph.nodes:
                    if 'hypergraph' in str(node.name):
                        for edge in node.parameters.get('hyperedges', []):
                            for comp in edge['components']:
                                participation[comp] = participation.get(comp, 0) + 1

        if not participation:
            print("No participation data.")
            return

        df = pd.DataFrame([
            {'Feature': feature_names[idx] if feature_names and idx < len(feature_names) else f"F{idx}", 
             'Frequency': count}
            for idx, count in participation.items()
        ])
        
        df = df.sort_values('Frequency', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='Frequency', y='Feature', palette='viridis')
        plt.title("Raw Feature Interaction Frequency (Global History)", fontsize=16)
        
        path = os.path.join(self.output_dir, "feature_interaction_importance.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved to {path}")
