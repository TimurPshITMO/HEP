import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import json
import hypernetx as hnx
from typing import List, Dict, Any

class HEPVisualizer:
    """
    Визуализатор для Standalone HEP на базе HyperNetX.
    Рисует гиперграфы профессионально и собирает анимации эволюции.
    """
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Улучшаем качество графики
        plt.rcParams['figure.dpi'] = 200
        plt.rcParams['font.family'] = 'sans-serif'

    def plot_individual(self, 
                        individual_data: Any, 
                        n_features: int, 
                        title: str = "Hypergraph Structure", 
                        save_path: str = None,
                        pos: Dict = None):
        """
        Рисует гиперграф одной особи. Поддерживает как словари, так и объекты Individual.
        """
        # 1. Сбор ребер и фитнеса
        hnx_edges = {}
        actual_fitness = 0.0
        
        # Если передан объект Individual (из движка)
        if hasattr(individual_data, 'genome'):
            edges_list = list(individual_data.genome.edges.values())
            actual_fitness = getattr(individual_data, 'fitness', 0.0)
            for i, edge in enumerate(edges_list):
                label = f"E{i}: {edge.function_name}"
                hnx_edges[label] = tuple(edge.node_indices)
        # Если передан словарь (из JSON истории)
        elif isinstance(individual_data, dict):
            actual_fitness = individual_data.get('fitness', 0.0)
            for i, edge in enumerate(individual_data.get('edges', [])):
                label = f"E{i}: {edge['func']}"
                hnx_edges[label] = tuple(edge['nodes'])
        else:
            raise TypeError("individual_data must be a dict or an Individual object")
            
        all_nodes = list(range(n_features))
        
        # 2. Создание Hypergraph
        if not hnx_edges:
            # Пустой граф
            fig, ax = plt.subplots(figsize=(10, 8))
            G = nx.Graph()
            G.add_nodes_from(all_nodes)
            if pos is None: pos = nx.circular_layout(G)
            nx.draw(G, pos, ax=ax, with_labels=True, node_color='silver', node_size=600, font_size=10)
            ax.set_title(f"{title} (No features evolved yet)", fontsize=14)
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
            return pos

        H = hnx.Hypergraph(hnx_edges)
        
        # 3. Layout
        if pos is None:
            # Circular layout часто выглядит чище для небольших наборов признаков
            temp_G = nx.Graph()
            temp_G.add_nodes_from(range(n_features))
            pos = nx.circular_layout(temp_G)

        # 4. Рендеринг
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Цвета для ребер
        import matplotlib.cm as cm
        colors = cm.get_cmap('Set2')(np.linspace(0, 1, len(H.edges)))
        
        try:
            # Отключаем проблемный fill_edge_alpha и используем fill_edges=True
            # HNX сам управляет цветами, если мы не передаем их жестко
            hnx.draw(H, 
                     pos=pos,
                     ax=ax,
                     node_radius=2.0,
                     fill_edges=True,
                     with_node_labels=True,
                     with_edge_labels=True,
                     node_labels_kwargs={'fontsize': 14, 'fontweight': 'bold'},
                     edge_labels_kwargs={'fontsize': 10, 'fontstyle': 'italic'},
                     nodes_kwargs={'facecolors': 'white', 'edgecolors': 'black', 'linewidths': 1.5},
                     edges_kwargs={'linewidths': 2}
            )
        except Exception as e:
            # Fallback к NetworkX если HNX все еще ломается
            print(f"HNX Draw failed, using NetworkX fallback: {e}")
            G_nx = nx.Graph()
            G_nx.add_nodes_from(range(n_features))
            nx.draw(G_nx, pos, ax=ax, with_labels=True, node_color='skyblue', node_size=800)
            if isinstance(individual_data, dict):
                edges_to_draw = individual_data.get('edges', [])
                for edge in edges_to_draw:
                    nodes = edge['nodes']
                    for j in range(len(nodes)):
                        for k in range(j+1, len(nodes)):
                            G_nx.add_edge(nodes[j], nodes[k])
            elif hasattr(individual_data, 'genome'):
                for edge in individual_data.genome.edges.values():
                    nodes = edge.node_indices
                    for j in range(len(nodes)):
                        for k in range(j+1, len(nodes)):
                            G_nx.add_edge(nodes[j], nodes[k])
            nx.draw_networkx_edges(G_nx, pos, alpha=0.3)

        fitness = actual_fitness
        ax.set_title(f"{title}\nR2: {fitness:.4f}", fontsize=18, fontweight='bold', pad=25)
        
        # Добавляем красивый фон или рамку
        ax.set_axis_off()
        fig.patch.set_facecolor('#fdfdfd')
        
        if save_path:
            plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        return pos

    def generate_evolution_frames(self, history_file: str):
        """
        Генерирует раскадровку из истории.
        """
        if not os.path.exists(history_file):
            print(f"Error: {history_file} not found.")
            return

        with open(history_file, 'r') as f:
            history = json.load(f)
            
        frames_dir = os.path.join(self.output_dir, 'frames')
        if os.path.exists(frames_dir):
            import shutil
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir, exist_ok=True)
        
        # Определяем n_features
        all_nodes = set()
        for gen in history:
            for ind in gen['population']:
                for edge in ind['edges']:
                    all_nodes.update(edge['nodes'])
        n_features = max(all_nodes) + 1 if all_nodes else 12
        
        # Фиксированный круговой лейаут
        temp_G = nx.Graph()
        temp_G.add_nodes_from(range(n_features))
        pos = nx.circular_layout(temp_G)
        
        print(f"Starting frame generation ({len(history)} frames)...")
        for i, gen in enumerate(history):
            best_ind = max(gen['population'], key=lambda x: x['fitness'])
            save_path = os.path.join(frames_dir, f"gen_{gen['generation']:03d}.png")
            self.plot_individual(best_ind, n_features=n_features, 
                                 title=f"Evolutionary HEP Insight: Gen {gen['generation']}", 
                                 save_path=save_path,
                                 pos=pos)
            if i % 10 == 0:
                print(f"  Processed {i}/{len(history)} frames...")
        
        print(f"Success! Captured {len(history)} frames in '{frames_dir}'")
