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

    def _draw_base_canvas(self, ax: plt.Axes, pos: Dict, n_features: int, labels: List[str]):
        """Отрисовывает базовые узлы (подложку) поверх всего через NetworkX."""
        bg_G = nx.Graph()
        bg_G.add_nodes_from(range(n_features))
        
        nx.draw_networkx_nodes(bg_G, pos, ax=ax, 
                               node_color='white', edgecolors='black', 
                               node_size=800, linewidths=1.5)
                               
        labels_dict = {i: labels[i] if i < len(labels) else f"X{i}" for i in range(n_features)}
        nx.draw_networkx_labels(bg_G, pos, labels=labels_dict, ax=ax, 
                                font_color='black', font_size=12, 
                                font_weight='bold', bbox={'facecolor':'white', 'linewidth':0})
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

        ax.set_aspect('equal', adjustable='box')

    def _draw_nx_fallback(self, ax: plt.Axes, pos: Dict, individual_data: Any):
        """Резервная отрисовка компонентов гиперграфа через линии NetworkX, если HNX падает."""
        G_nx = nx.Graph()
        
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
                        
        nx.draw_networkx_edges(G_nx, pos, ax=ax, alpha=0.3)

    def plot_individual(self, 
                        individual_data: Any, 
                        n_features: int, 
                        title: str = "Hypergraph Structure", 
                        save_path: str = None,
                        pos: Dict = None,
                        labels: List[str] = None) -> Dict:
        """
        Рисует гиперграф одной особи. Поддерживает как словари, так и объекты Individual.
        """
        # 1. Сбор ребер и фитнеса
        hnx_edges = {}
        actual_fitness = 0.0
        
        if labels is None:
            labels = [str(i) for i in range(n_features)]
        
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
            
        if pos is None:
            # Получаем координаты точек напрямую из списка индексов
            pos = nx.circular_layout(range(n_features))

        # 2. Инициализация холста
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 3. Отрисовка пустого графа (нулевой кадр)
        if not hnx_edges:
            self._draw_base_canvas(ax, pos, n_features, labels)
            ax.set_title(f"{title}\nR2: {actual_fitness:.4f}", fontsize=18, fontweight='bold', pad=25)
            ax.set_axis_off()
            fig.patch.set_facecolor('#fdfdfd')
            
            if save_path:
                plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
                plt.close(fig)
            return pos

        # 4. Рендеринг гиперграфа
        H = hnx.Hypergraph(hnx_edges)

        try:
            # Отключаем проблемный fill_edge_alpha и используем fill_edges=True
            hnx.draw(H, 
                     pos=pos,
                     ax=ax,
                     fill_edges=True,
                     with_node_labels=False,  # Отключаем лейблы HNX
                     with_edge_labels=True,
                     edge_labels_kwargs={'fontsize': 12, 'fontstyle': 'italic', 'fontweight': 'bold'},
                     nodes_kwargs={'alpha': 0}, # Полностью скрываем узлы HNX
                     edges_kwargs={'linewidths': 2},
                     node_radius=2,
            )
            # Единый рендеринг ВСЕХ узлов поверх пузырей гиперграфа
            self._draw_base_canvas(ax, pos, n_features, labels)
            
        except Exception as e:
            # Fallback к NetworkX если HNX падает
            print(f"HNX Draw failed, using NetworkX fallback: {e}")
            self._draw_base_canvas(ax, pos, n_features, labels)
            self._draw_nx_fallback(ax, pos, individual_data)

        # 5. Финализация картинки
        ax.set_title(f"{title}\nR2: {actual_fitness:.4f}", fontsize=18, fontweight='bold', pad=25)
        ax.set_axis_off()
        fig.patch.set_facecolor('#fdfdfd')
        
        if save_path:
            plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
            plt.close(fig)
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
            file_data = json.load(f)
            history = file_data['history']
            labels = file_data['labels']
            
        frames_dir = os.path.join(self.output_dir, 'frames')
        if os.path.exists(frames_dir):
            import shutil
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir, exist_ok=True)
        
        n_features = len(labels)
        
        # Получаем координаты
        pos = nx.circular_layout(range(n_features))
        
        print(f"Starting frame generation ({len(history)} frames)...")
        for i, gen in enumerate(history):
            best_ind = max(gen['population'], key=lambda x: x['fitness'])
            save_path = os.path.join(frames_dir, f"gen_{gen['generation']:03d}.png")
            
            self.plot_individual(best_ind, 
                                 n_features=n_features,
                                 labels=labels,
                                 title=f"Evolutionary HEP Insight: Gen {gen['generation']}", 
                                 save_path=save_path,
                                 pos=pos)
                                 
            if i % 10 == 0:
                print(f"  Processed {i}/{len(history)} frames...")
        
        print(f"Success! Captured {len(history)} frames in '{frames_dir}'")
