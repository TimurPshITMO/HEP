import json
import os
from typing import List
from .evolution import Individual

class EvolutionTracker:
    """
    Отвечает за сохранение истории эволюции для последующей визуализации.
    """
    def __init__(self, output_dir: str = 'history'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.history = []

    def record_generation(self, gen_idx: int, individuals: List[Individual]):
        """Записывает состояние поколения."""
        gen_data = {
            'generation': gen_idx,
            'population': []
        }
        
        for ind in individuals:
            ind_data = {
                'id': ind.id,
                'fitness': float(ind.fitness),
                'edges': [
                    {'nodes': e.node_indices, 'func': e.function_name, 'sig': e.signature}
                    for e in ind.genome.edges.values()
                ],
                'parents': ind.parents
            }
            gen_data['population'].append(ind_data)
            
        self.history.append(gen_data)
        
        # Сохраняем в отдельный файл для безопасности
        filename = os.path.join(self.output_dir, f"gen_{gen_idx:03d}.json")
        with open(filename, 'w') as f:
            json.dump(gen_data, f, indent=2)

    def save_full_history(self, filename: str = "full_history.json"):
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.history, f)
        print(f"Full history saved to {path}")
