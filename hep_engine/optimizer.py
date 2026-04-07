from __future__ import annotations
from typing import List, Optional, Dict
import time
import random
import numpy as np
from .evolution import Individual, Population, GeneticOperators
from .evaluator import FitnessEvaluator
from .tracker import EvolutionTracker

class EvolutionaryOptimizer:
    """Оркестратор процесса эволюции алгоритма Standalone HEP.
    
    Управляет жизненным циклом популяции: инициализацией геномов, 
    параллельной оценкой фитнеса, турнирной селекцией, кроссовером 
    и мутациями. Отвечает за логирование процесса.
    
    Attributes:
        pop_size (int): Размер популяции в каждом поколении.
        elitism_count (int): Количество лучших особей, переходящих в 
            следующее поколение без изменений.
        tournament_size (int): Количество особей для турнирного отбора.
        operators (GeneticOperators): Набор операторов эволюции.
        tracker (EvolutionTracker): Подсистема сохранения истории.
    """
    def __init__(self, 
                 pop_size: int = 20, 
                 mut_rate: float = 0.4, 
                 cross_rate: float = 0.5,
                 elitism_count: int = 2,
                 tournament_size: int = 3,
                 available_functions: Optional[List[str]] = None):
        """Инициализирует генетический оптимизатор.
        
        Args:
            pop_size (int, optional): Размер популяции. Defaults to 20.
            mut_rate (float, optional): Шанс мутации. Defaults to 0.4.
            cross_rate (float, optional): Шанс скрещивания. Defaults to 0.5.
            elitism_count (int, optional): Сколько элитных особей выживает 
                автоматически. Defaults to 2.
            tournament_size (int, optional): Размер турнира на селекции. Defaults to 3.
            available_functions (Optional[List[str]], optional): Функции агрегации.
        """
        self.pop_size = pop_size
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        self.operators = GeneticOperators(
            mutation_rate=mut_rate, 
            crossover_rate=cross_rate,
            available_functions=available_functions
        )
        self.tracker = EvolutionTracker()
        self._fitness_cache: Dict[str, float] = {}

    def run(self, 
            evaluator: FitnessEvaluator, 
            n_generations: int = 20, 
            timeout: float = 600,
            labels: List[str] = None,
            record_history: bool = False) -> Population:
        """Запускает основной цикл эволюции гиперграфов.
        
        Args:
            evaluator (FitnessEvaluator): Настроенный блок оценки с данными и моделью.
            n_generations (int, optional): Количество поколений эволюции. Defaults to 20.
            timeout (float, optional): Лимит времени обработки в секундах. Defaults to 600.
            labels (List[str], optional): Список исходных признаков данных для 
                трекера истории. Defaults to None.
            record_history (bool, optional): Нужно ли сохранять полные JSON дампы. Defaults to False.
            
        Returns:
            Population: Обьект итоговой популяции. Отсортирован по фитнесу (лучшие в начале).
            
        Examples:
            >>> optimizer = EvolutionaryOptimizer(pop_size=10)
            >>> evaluator = FitnessEvaluator(X, y)
            >>> output_pop = optimizer.run(evaluator, n_generations=5)
            >>> print(output_pop.best().fitness)
            0.92
        """
        start_time = time.time()
        n_features = evaluator.X.shape[1]
        
        # 1. Инициализация (Нулевое поколение)
        pop = Population(self.pop_size)
        pop.initialize(n_features)

        self.tracker.record_labels(labels if labels is not None else [f"X{i}" for i in range(n_features)])
        
        self._evaluate_population(pop.individuals, evaluator)
            
        for gen in range(n_generations):
            if (time.time() - start_time) > timeout:
                print("Optimization stopped by timeout.")
                break
                
            self.tracker.record_generation(gen, pop.individuals)
            
            pop.sort()
            print(f"Gen {gen:03d} | Best: {pop.individuals[0].fitness:.4f} | Avg: {pop.avg_fitness():.4f}")
            
            # 2. Создание нового поколения
            new_individuals = []
            
            # 2.1 Элитизм (полное копирование лучших особей)
            new_individuals.extend([ind.clone() for ind in pop.individuals[:self.elitism_count]])
            
            # 2.2 Репродукция популяции
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
                        
            # Оценка только новых потомков (элитные сохраняют фитнес)
            self._evaluate_population(new_individuals[self.elitism_count:], evaluator)
                
            pop.individuals = new_individuals

        pop.sort()
        if record_history:
            self.tracker.save_full_history()
        return pop

    def _selection(self, pop: Population) -> Individual:
        """Реализует турнирную селекцию для выбора родителей."""
        selection = random.sample(pop.individuals, self.tournament_size)
        return max(selection, key=lambda x: x.fitness)

    def _evaluate_population(self, individuals: List[Individual], evaluator: FitnessEvaluator) -> None:
        """Пакетно вычисляет фитнес, переиспользуя кэш для старых геномов."""
        for ind in individuals:
            sig = ind.genome.signature
            if sig in self._fitness_cache:
                ind.fitness = self._fitness_cache[sig]
            else:
                evaluator.evaluate(ind)
                self._fitness_cache[sig] = ind.fitness
