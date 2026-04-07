import hashlib
import numpy as np
from typing import List, Dict

class Hyperedge:
    """Атомарная единица функционального гиперграфа (HEP).
    
    Представляет собой математическую функцию (например, сумму или максимум), 
    примененную к набору индексов признаков датасета (узлам).
    
    Attributes:
        node_indices (List[int]): Уникальный отсортированный список индексов признаков, 
            на которые опирается гиперребро.
        function_name (str): Название применяемой математической функции.
        signature (str): Уникальный MD5 хэш, идентифицирующий структуру ребра.
    """

    def __init__(self, node_indices: List[int], function_name: str):
        """Инициализирует гиперребро.
        
        Args:
            node_indices (List[int]): Индексы признаков, объединяемых ребром.
            function_name (str): Название функции агрегации (например, 'sum', 'prod').
        """
        self.node_indices = sorted(list(set(node_indices)))
        self.function_name = function_name
        self._signature = self._generate_signature()

    def _generate_signature(self) -> str:
        """Генерирует MD5-хэш на основе функции и узлов.
        
        Returns:
            str: 32-символьная шестнадцатеричная строка.
        """
        s = f"{self.function_name}:{','.join(map(str, self.node_indices))}"
        return hashlib.md5(s.encode()).hexdigest()

    @property
    def signature(self) -> str:
        """str: Публичный доступ к уникальной сигнатуре ребра."""
        return self._signature

    def __repr__(self) -> str:
        return f"Hyperedge({self.function_name}, nodes={self.node_indices})"

class Hypergraph:
    """Геном особи в эволюционном алгоритме HEP (Hypergraph-Evolved Pipelines).
    
    Гиперграф управляет набором гиперребер (выученных нелинейностей) и 
    отвечает за трансформацию сырых табличных данных в многомерное 
    новое пространство признаков.
    
    Attributes:
        n_features (int): Количество базовых (сырых) признаков в данных.
        edges (Dict[str, Hyperedge]): Словарь всех гиперребер графа, 
            где ключ — это сигнатура ребра (MD5).
    """

    def __init__(self, n_features: int):
        """Создает пустой функциональный гиперграф.
        
        Args:
            n_features (int): Размерность исходного пространства данных.
        """
        self.n_features = n_features
        self.edges: Dict[str, Hyperedge] = {}
        
        self._registry = {
            'sum': np.sum,
            'mean': np.mean,
            'max': np.max,
            'min': np.min,
            'std': np.std,
            'prod': np.prod
        }

    def add_edge(self, node_indices: List[int], function_name: str) -> str:
        """Добавляет новое гиперребро в структуру графа.
        
        Если идентичное ребро (с таким же набором узлов и функцией) 
        уже существует, дубликат не добавляется.
        
        Args:
            node_indices (List[int]): Индексы опорных признаков.
            function_name (str): Имя агрегационной функции из реестра.
            
        Returns:
            str: Уникальная сигнатура добавленного (или существующего) ребра.
            
        Raises:
            ValueError: Если индекс выходит за пределы `[0, n_features)`.
            
        Examples:
            >>> hg = Hypergraph(n_features=5)
            >>> sig = hg.add_edge([0, 2, 4], 'max')
        """
        if not all(0 <= i < self.n_features for i in node_indices):
            raise ValueError(f"Индексы признаков должны быть в диапазоне [0, {self.n_features})")
            
        edge = Hyperedge(node_indices, function_name)
        if edge.signature not in self.edges:
            self.edges[edge.signature] = edge
        return edge.signature

    def remove_edge(self, signature: str) -> None:
        """Удаляет гиперребро из графа по его сигнатуре.
        
        Args:
            signature (str): MD5 хэш удаляемого ребра.
        """
        if signature in self.edges:
            del self.edges[signature]

    def transform(self, X: np.ndarray, cache: Dict[str, np.ndarray] = None) -> np.ndarray:
        """Трансформирует сырую матрицу признаков, применяя операторы гиперребер.
        
        Args:
            X (np.ndarray): Исходные данные формы `(n_samples, n_features)`.
            cache (Dict[str, np.ndarray], optional): Словарь для кэширования 
                вычисленных столбцов. Позволяет значительно ускорить оценку 
                одинаковых ребер в разных особях популяции. Defaults to None.
                
        Returns:
            np.ndarray: Новая матрица признаков формы `(n_samples, len(edges))`. 
                Если граф пуст, возвращается матрица `(n_samples, 0)`.
                
        Examples:
            >>> X = np.array([[1, 2], [3, 4]])
            >>> hg = Hypergraph(n_features=2)
            >>> hg.add_edge([0, 1], 'prod')
            >>> hg.transform(X)
            array([[ 2],
                   [12]])
        """
        if not self.edges:
            return np.zeros((X.shape[0], 0))
            
        new_features = []
        
        for sig, edge in self.edges.items():
            if cache is not None and sig in cache and cache[sig].shape[0] == X.shape[0]:
                new_features.append(cache[sig])
                continue
            
            func = self._registry.get(edge.function_name, np.sum)
            X_sub = X[:, edge.node_indices]
            computed = func(X_sub, axis=1)
            
            if cache is not None:
                cache[sig] = computed
                
            new_features.append(computed)
            
        return np.column_stack(new_features)

    @property
    def signature(self) -> str:
        """str: Генерирует детерминированный хэш всего графа.
        
        Полезен для избежания повторных вычислений фитнеса для идентичных 
        по структуре особей (например, появившихся в результате скрещивания).
        """
        if not self.edges:
            return "empty"
            
        sorted_edge_sigs = sorted(self.edges.keys())
        s = "|".join(sorted_edge_sigs)
        return hashlib.md5(s.encode()).hexdigest()

    def __len__(self) -> int:
        return len(self.edges)
