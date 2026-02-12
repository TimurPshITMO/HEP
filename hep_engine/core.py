import numpy as np
import hashlib
from typing import List, Dict, Any, Optional

class Hyperedge:
    """
    Атомарная единица HEP — гиперребро.
    Представляет собой функцию, примененную к набору признаков.
    """
    def __init__(self, node_indices: List[int], function_name: str):
        # Храним индексы отсортированными для идентичности сигнатур
        self.node_indices = sorted(list(set(node_indices)))
        self.function_name = function_name
        self._signature = self._generate_signature()

    def _generate_signature(self) -> str:
        """Генерирует уникальный MD5 хэш для ребра."""
        s = f"{self.function_name}:{','.join(map(str, self.node_indices))}"
        return hashlib.md5(s.encode()).hexdigest()

    @property
    def signature(self) -> str:
        return self._signature

    def __repr__(self):
        return f"Edge({self.function_name}, nodes={self.node_indices})"

class Hypergraph:
    """
    Геном HEP — набор гиперребер.
    Отвечает за трансформацию данных и управление структурой.
    """
    def __init__(self, n_features: int):
        self.n_features = n_features
        self.edges: Dict[str, Hyperedge] = {}
        
        # Реестр доступных функций
        self._registry = {
            'sum': np.sum,
            'mean': np.mean,
            'max': np.max,
            'min': np.min,
            'std': np.std,
            'prod': np.prod
        }

    def add_edge(self, node_indices: List[int], function_name: str) -> str:
        """Добавляет ребро в граф. Возвращает сигнатуру."""
        if not all(0 <= i < self.n_features for i in node_indices):
            raise ValueError(f"Индексы признаков должны быть в диапазоне [0, {self.n_features})")
            
        edge = Hyperedge(node_indices, function_name)
        if edge.signature not in self.edges:
            self.edges[edge.signature] = edge
        return edge.signature

    def remove_edge(self, signature: str):
        """Удаляет ребро по сигнатуре."""
        if signature in self.edges:
            del self.edges[signature]

    def transform(self, X: np.ndarray, cache: Dict[str, np.ndarray] = None) -> np.ndarray:
        """
        Трансформирует входные данные X в новые признаки.
        X: (n_samples, n_features)
        """
        if not self.edges:
            return np.zeros((X.shape[0], 0))
            
        # Список для сбора колонок
        new_features = []
        
        for sig, edge in self.edges.items():
            # Если есть в кэше и размеры совпадают — используем
            if cache is not None and sig in cache and cache[sig].shape[0] == X.shape[0]:
                new_features.append(cache[sig])
                continue
            
            # Иначе вычисляем
            func = self._registry.get(edge.function_name, np.sum)
            X_sub = X[:, edge.node_indices]
            computed = func(X_sub, axis=1)
            
            if cache is not None:
                cache[sig] = computed
                
            new_features.append(computed)
            
        return np.column_stack(new_features)

    @property
    def signature(self) -> str:
        """Генерирует уникальный хэш для всего графа на основе всех ребер."""
        if not self.edges:
            return "empty"
        # Сортируем сигнатуры ребер для детерминированности
        sorted_edge_sigs = sorted(self.edges.keys())
        s = "|".join(sorted_edge_sigs)
        return hashlib.md5(s.encode()).hexdigest()

    def __len__(self):
        return len(self.edges)
