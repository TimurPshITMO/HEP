import uuid
import copy
import random
import numpy as np
from typing import List, Optional
from .core import Hypergraph

class Individual:
    """
    Особь популяции. Сочетает геном гиперграфа и показатели его качества.
    """
    def __init__(self, n_features: int, genome: Optional[Hypergraph] = None):
        self.id = str(uuid.uuid4())[:8]
        self.genome = genome if genome is not None else Hypergraph(n_features)
        self.fitness: float = -1e12 # Отрицательная бесконечность
        self.generation: int = 0
        self.parents: List[str] = []

    def clone(self) -> 'Individual':
        """Создает полную глубокую копию особи."""
        new_ind = Individual(self.genome.n_features)
        new_ind.genome = copy.deepcopy(self.genome)
        new_ind.fitness = self.fitness
        new_ind.generation = self.generation
        new_ind.parents = self.parents.copy()
        return new_ind

    def __repr__(self):
        return f"Ind[{self.id}](edges={len(self.genome)}, fit={self.fitness:.4f})"

class Population:
    """
    Менеджер популяции особей.
    """
    def __init__(self, size: int):
        self.size = size
        self.individuals: List[Individual] = []

    def initialize(self, n_features: int):
        """Создает начальную пустую популяцию."""
        self.individuals = [Individual(n_features) for _ in range(self.size)]

    def best(self) -> Individual:
        """Возвращает лучшую особь."""
        return max(self.individuals, key=lambda x: x.fitness)

    def sort(self):
        """Сортирует популяцию по убыванию фитнеса."""
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)

    def avg_fitness(self) -> float:
        """Возвращает средний фитнес по популяции."""
        if not self.individuals:
            return 0.0
        return float(np.mean([ind.fitness for ind in self.individuals]))

class GeneticOperators:
    """
    Набор операторов для эволюции гиперграфов.
    """
    def __init__(self, mutation_rate: float = 0.4, crossover_rate: float = 0.5, available_functions: Optional[List[str]] = None):
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self._functions = available_functions if available_functions is not None else ['sum', 'mean', 'max', 'min', 'std', 'prod']

    def mutate(self, individual: Individual) -> Individual:
        """Случайная мутация структуры гиперграфа."""
        if random.random() > self.mutation_rate:
            return individual
            
        new_ind = individual.clone()
        hg = new_ind.genome
        n_edges = len(hg)
        
        # Выбор типа мутации: Добавить, Удалить, Модифицировать
        choices = ['add']
        if n_edges > 1: choices.extend(['remove', 'modify'])
        elif n_edges == 1: choices.extend(['modify'])
        
        op = random.choice(choices)
        
        if op == 'add':
            # Добавляем ребро с 2-4 компонентами
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
            # Либо меняем функцию, либо один из узлов
            if random.random() < 0.5:
                edge.function_name = random.choice(self._functions)
                # Хэш должен обновиться
                edge._signature = edge._generate_signature()
                # Переносим в словаре под новый ключ
                new_sig = edge.signature
                if new_sig != sig:
                    hg.edges[new_sig] = hg.edges.pop(sig)
            else:
                nodes = list(edge.node_indices)
                idx_to_change = random.randrange(len(nodes))
                nodes[idx_to_change] = random.randrange(hg.n_features)
                # Валидация уникальности узлов
                nodes = sorted(list(set(nodes)))
                if len(nodes) < 2: # Не даем выродиться в петлю
                    return new_ind # Отменяем мутацию узла
                
                edge.node_indices = nodes
                edge._signature = edge._generate_signature()
                new_sig = edge.signature
                if new_sig != sig:
                    if new_sig not in hg.edges:
                        hg.edges[new_sig] = hg.edges.pop(sig)
                    else:
                        hg.remove_edge(sig) # Если такое ребро уже есть
                        
        return new_ind

    def crossover(self, parent1: Individual, parent2: Individual) -> List[Individual]:
        """Скрещивание через обмен наборами гиперребер."""
        if random.random() > self.crossover_rate:
            return [parent1.clone(), parent2.clone()]
            
        offspring1 = parent1.clone()
        offspring2 = parent2.clone()
        
        edges1 = list(offspring1.genome.edges.values())
        edges2 = list(offspring2.genome.edges.values())
        
        if not edges1 or not edges2:
            return [offspring1, offspring2]
            
        # Точка разрыва для обмена списками ребер
        cp1 = random.randrange(len(edges1))
        cp2 = random.randrange(len(edges2))
        
        new_list1 = edges1[:cp1] + edges2[cp2:]
        new_list2 = edges2[:cp2] + edges1[cp1:]
        
        # Обновляем геномы (уникальность сигнатур сохранится благодаря dict)
        offspring1.genome.edges = {e.signature: e for e in new_list1}
        offspring2.genome.edges = {e.signature: e for e in new_list2}
        
        return [offspring1, offspring2]
