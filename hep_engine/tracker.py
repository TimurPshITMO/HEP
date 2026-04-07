import json
import os
from typing import List, Dict, Any
from .evolution import Individual

class EvolutionTracker:
    """Система логирования и сериализации процесса эволюции.
    
    Отвечает за сохранение истории изменения популяции в JSON-формате
    для последующей визуализации, анализа сходимости и анимации.
    Создает как промежуточные дампы поколений, так и итоговый полный файл.
    
    Attributes:
        output_dir (str): Директория для сохранения файлов истории.
        history (List[Dict[str, Any]]): Внутренний буфер, накапливающий дампы всех поколений.
        labels (List[str]): Список человекочитаемых имен признаков.
    """
    def __init__(self, output_dir: str = 'history'):
        """Инициализирует систему логирования.
        
        Args:
            output_dir (str, optional): Относительный или абсолютный путь 
                к папке вывода. Defaults to 'history'.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.history: List[Dict[str, Any]] = []
        self.labels: List[str] = []

    def record_generation(self, gen_idx: int, individuals: List[Individual]) -> None:
        """Делает 'снимок' текущего состояния популяции.
        
        Сериализует геномы (функции и связи) и параметры особи (фитнес, родители).
        Моментально сохраняет промежуточный файл `gen_XXX.json` для безопасности.
        
        Args:
            gen_idx (int): Индекс текущего поколения.
            individuals (List[Individual]): Список всех особей на данной итерации.
        """
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
        
        filename = os.path.join(self.output_dir, f"gen_{gen_idx:03d}.json")
        with open(filename, 'w') as f:
            json.dump(gen_data, f, indent=2)

    def save_full_history(self, filename: str = "full_history.json") -> None:
        """Экспортирует всю накопленную историю в единый файл.
        
        Включает в себя метаданные, такие как названия признаков (labels), 
        необходимые для восстановления контекста при визуализации.
        
        Args:
            filename (str, optional): Имя выходного файла. Defaults to "full_history.json".
        """
        path = os.path.join(self.output_dir, filename)
        n_features = len(self.labels) if self.labels else 0
        with open(path, 'w') as f:
            json.dump({'history': self.history, 'labels': self.labels, 'n_features': n_features}, f)
        print(f"Full history saved to {path}")

    def record_labels(self, labels: List[str]) -> None:
        """Сохраняет метаданные колонок датасета для истории.
        
        Args:
            labels (List[str]): Текстовые названия признаков.
        """
        self.labels = labels