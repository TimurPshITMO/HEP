import uuid
import copy
import random
import numpy as np
from typing import List, Optional
from .core import Hypergraph

class Individual:
    """Особь популяции в генетическом алгоритме.
    
    Сочетает в себе геном (структуру функционального гиперграфа) и показатели 
    соответствия целевой метрике (фитнес). Является основным юнитом, 
    передающимся между поколениями.
    
    Attributes:
        id (str): Уникальный 8-символьный идентификатор особи.
        genome (Hypergraph): Экземпляр гиперграфа, описывающий структуру признаков.
        fitness (float): Значение функции приспособленности. Исходно равно `-1e12`.
        generation (int): Номер поколения, в котором особь впервые появилась.
        parents (List[str]): Список ID родительских особей.
    """
    def __init__(self, n_features: int, genome: Optional[Hypergraph] = None):
        """Инициализирует новую особь.
        
        Args:
            n_features (int): Количество оригинальных признаков датасета.
            genome (Optional[Hypergraph], optional): Готовый геном. 
                Если None, инициализируется пустой гиперграф. Defaults to None.
        """
        self.id = str(uuid.uuid4())[:8]
        self.genome = genome if genome is not None else Hypergraph(n_features)
        self.fitness: float = -1e12
        self.generation: int = 0
        self.parents: List[str] = []

    def clone(self) -> 'Individual':
        """Создает полную независимую (глубокую) копию особи.
        
        Используется при мутациях и элитизме для предотвращения перезаписи 
        структур родительских геномов в памяти.
        
        Returns:
            Individual: Абсолютный клон текущей особи с оригинальным fitness, 
                но независимым `Hypergraph` объектом в памяти.
        """
        new_ind = Individual(self.genome.n_features)
        new_ind.genome = copy.deepcopy(self.genome)
        new_ind.fitness = self.fitness
        new_ind.generation = self.generation
        new_ind.parents = self.parents.copy()
        return new_ind

    def __repr__(self) -> str:
        return f"Ind[{self.id}](edges={len(self.genome)}, fit={self.fitness:.4f})"

class Population:
    """Менеджер, инкапсулирующий математическую популяцию особей.
    
    Отвечает за хранение, сортировку и предоставление статистик по группе
    индивидов на текущей итерации алгоритма.
    
    Attributes:
        size (int): Максимально допустимый размер популяции.
        individuals (List[Individual]): Список всех особей текущего поколения.
    """
    def __init__(self, size: int):
        """Инициализирует контейнер популяции.
        
        Args:
            size (int): Максимальное число особей (емкость).
        """
        self.size = size
        self.individuals: List[Individual] = []

    def initialize(self, n_features: int) -> None:
        """Создает начальнуюпустую (baseline) популяцию.
        
        Args:
            n_features (int): Размерность оригинальных признаков для инициализации геномов.
        """
        self.individuals = [Individual(n_features) for _ in range(self.size)]

    def best(self) -> Individual:
        """Находит и возвращает самую приспособленную особь.
        
        Returns:
            Individual: Особь с наивысшим параметром `fitness`.
        """
        return max(self.individuals, key=lambda x: x.fitness)

    def sort(self) -> None:
        """Сортирует массив `individuals` по убыванию фитнеса (In-place)."""
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)

    def avg_fitness(self) -> float:
        """Рассчитывает средний фитнес по всей популяции.
        
        Returns:
            float: Среднее значение фитнеса (удобно для трекинга конвергенции). 
                0.0, если популяция пуста.
        """
        if not self.individuals:
            return 0.0
        return float(np.mean([ind.fitness for ind in self.individuals]))

class GeneticOperators:
    """Модуль генетических операторов, специализированных под гиперграфы.
    
    Реализует структурные мутации (манипуляции гиперребрами) и кроссовер, 
    которые соблюдают правила комбинаторной математики применяемой к моделям.
    
    Attributes:
        mutation_rate (float): Вероятность срабатывания мутации на особь (от 0 до 1).
        crossover_rate (float): Вероятность скрещивания двух особей (от 0 до 1).
    """
    def __init__(self, mutation_rate: float = 0.4, crossover_rate: float = 0.5, 
                 available_functions: Optional[List[str]] = None):
        """Инициализирует набор операторов эволюции.
        
        Args:
            mutation_rate (float, optional): Шанс мутации особи. Defaults to 0.4.
            crossover_rate (float, optional): Шанс скрещивания выбранной пары. Defaults to 0.5.
            available_functions (Optional[List[str]], optional): Реестр разрешенных 
                математических агрегаций. По умолчанию содержит 
                `['sum', 'mean', 'max', 'min', 'std', 'prod']`.
        """
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self._functions = available_functions if available_functions is not None else [
            'sum', 'mean', 'max', 'min', 'std', 'prod'
        ]

    def mutate(self, individual: Individual) -> Individual:
        """Применяет случайную структурную мутацию к гиперграфу.
        
        Возможные типы мутаций:
            - **add**: Добавление нового гиперребра со случайными функциями и узлами.
            - **remove**: Удаление одного случайного гиперребра.
            - **modify**: Замена функции или одного узла в существующем гиперребре.
            
        Args:
            individual (Individual): Входящая особь.
            
        Returns:
            Individual: Если мутация произошла — возвращается новый измененный клон. 
                Если не произошла — возвращается исходная особь (без клонирования).
        """
        if random.random() > self.mutation_rate:
            return individual
            
        new_ind = individual.clone()
        hg = new_ind.genome
        n_edges = len(hg)
        
        # Динамический выбор доступных операций
        choices = ['add']
        if n_edges > 1: 
            choices.extend(['remove', 'modify'])
        elif n_edges == 1: 
            choices.extend(['modify'])
        
        op = random.choice(choices)
        
        if op == 'add':
            # Добавление ребра, покрывающего от 2 до 4 признаков
            size = random.randint(2, min(4, hg.n_features))
            nodes = random.sample(range(hg.n_features), size)
            func = random.choice(self._functions)
            hg.add_edge(nodes, func)
            
        elif op == 'remove':
            sig = random.choice(list(hg.edges.keys()))
            hg.remove_edge(sig)
            
        elif op == 'modify':
            sig = random.choice(list(hg.edges.keys()))
            edge = hg.edges[sig]
            
            # 50/50: смена функции ИЛИ смена узла связи
            if random.random() < 0.5:
                edge.function_name = random.choice(self._functions)
                # Хэш обновился, нужна перерегистрация в словаре графа
                edge._signature = edge._generate_signature()
                new_sig = edge.signature
                if new_sig != sig:
                    hg.edges[new_sig] = hg.edges.pop(sig)
            else:
                nodes = list(edge.node_indices)
                idx_to_change = random.randrange(len(nodes))
                nodes[idx_to_change] = random.randrange(hg.n_features)
                
                # Валидация вырождения (исключение самопетель)
                nodes = sorted(list(set(nodes)))
                if len(nodes) < 2: 
                    return new_ind # Отменяем мутацию, если узел задублировался
                
                edge.node_indices = nodes
                edge._signature = edge._generate_signature()
                new_sig = edge.signature
                
                if new_sig != sig:
                    if new_sig not in hg.edges:
                        hg.edges[new_sig] = hg.edges.pop(sig)
                    else:
                        hg.remove_edge(sig) # Если клон ребра уже есть, просто удаляем оригинал
                        
        return new_ind

    def crossover(self, parent1: Individual, parent2: Individual) -> List[Individual]:
        """Скрещивание путем одноточечного обмена гиперребрами.
        
        Args:
            parent1 (Individual): Первый родитель.
            parent2 (Individual): Второй родитель.
            
        Returns:
            List[Individual]: Возвращает массив из двух полученных потомков. Если 
                вероятность скрещивания не сработала — возвращает клонов родителей.
        """
        if random.random() > self.crossover_rate:
            return [parent1.clone(), parent2.clone()]
            
        offspring1 = parent1.clone()
        offspring2 = parent2.clone()
        
        edges1 = list(offspring1.genome.edges.values())
        edges2 = list(offspring2.genome.edges.values())
        
        if not edges1 or not edges2:
            return [offspring1, offspring2]
            
        # Одноточечный разрыв (свап массивов)
        cp1 = random.randrange(len(edges1))
        cp2 = random.randrange(len(edges2))
        
        new_list1 = edges1[:cp1] + edges2[cp2:]
        new_list2 = edges2[:cp2] + edges1[cp1:]
        
        # Обновляем геномы (избыточные дубликаты уберутся благодаря Dict хэшам)
        offspring1.genome.edges = {e.signature: e for e in new_list1}
        offspring2.genome.edges = {e.signature: e for e in new_list2}
        
        return [offspring1, offspring2]
