from __future__ import annotations
from typing import List, Optional, Dict
import time
import numpy as np
from .evolution import Individual, Population, GeneticOperators
from .evaluator import FitnessEvaluator
from .tracker import EvolutionTracker

class EvolutionaryOptimizer:
    """
    Основной управляющий класс для процесса эволюции Standalone HEP.
    """
    def __init__(self, 
                 pop_size: int = 20, 
                 mut_rate: float = 0.4, 
                 cross_rate: float = 0.5,
                 elitism_count: int = 2,
                 available_functions: Optional[List[str]] = None):
        self.pop_size = pop_size
        self.elitism_count = elitism_count
        self.operators = GeneticOperators(
            mutation_rate=mut_rate, 
            crossover_rate=cross_rate,
            available_functions=available_functions
        )
        self.tracker = EvolutionTracker()
        self._fitness_cache: Dict[str, float] = {}
        self._fitness_cache: Dict[str, float] = {}

    def run(self, 
            evaluator: FitnessEvaluator, 
            n_generations: int = 20, 
            timeout: float = 600) -> Population:
        
        start_time = time.time()
        n_features = evaluator.X.shape[1]
        
        # 1. Инициализация
        pop = Population(self.pop_size)
        pop.initialize(n_features)
        
        # Первая оценка
        for ind in pop.individuals:
            sig = ind.genome.signature
            if sig in self._fitness_cache:
                ind.fitness = self._fitness_cache[sig]
            else:
                evaluator.evaluate(ind)
                self._fitness_cache[sig] = ind.fitness
            
        for gen in range(n_generations):
            if (time.time() - start_time) > timeout:
                print("Optimization stopped by timeout.")
                break
                
            # Запись истории
            self.tracker.record_generation(gen, pop.individuals)
            
            pop.sort()
            print(f"Gen {gen:03d} | Best: {pop.individuals[0].fitness:.4f} | Avg: {pop.avg_fitness():.4f}")
            
            # 2. Создание нового поколения
            new_individuals = []
            
            # Элитизм
            new_individuals.extend([ind.clone() for ind in pop.individuals[:self.elitism_count]])
            
            # Репродукция
            while len(new_individuals) < self.pop_size:
                p1 = self._selection(pop)
                p2 = self._selection(pop)
                
                # Кроссовер
                offspring = self.operators.crossover(p1, p2)
                
                for child in offspring:
                    # Мутация
                    mutated = self.operators.mutate(child)
                    mutated.generation = gen + 1
                    mutated.parents = [p1.id, p2.id]
                    
                    if len(new_individuals) < self.pop_size:
                        new_individuals.append(mutated)
                        
            # Оценка новых (тех, кто не элиты)
            for ind in new_individuals[self.elitism_count:]:
                sig = ind.genome.signature
                if sig in self._fitness_cache:
                    ind.fitness = self._fitness_cache[sig]
                else:
                    evaluator.evaluate(ind)
                    self._fitness_cache[sig] = ind.fitness
                
            pop.individuals = new_individuals

        pop.sort()
        self.tracker.save_full_history()
        return pop

    def _selection(self, pop: Population) -> Individual:
        """Турнирная селекция."""
        import random
        k = 3
        selection = random.sample(pop.individuals, k)
        return max(selection, key=lambda x: x.fitness)
