import unittest
import random
from hep_engine.evolution import Individual, Population, GeneticOperators

class TestEvolution(unittest.TestCase):
    """Модуль тестирования генетических операторов.
    
    Сюда входят проверки стабильности мутаций, кроссовера и 
    глубокого клонирования особей (во избежание утечек ссылок памяти
    при передаче в следующие поколения).
    """
    
    def test_individual_cloning(self):
        """Проверка свойств глубокого (deep) копирования Individual.
        
        Ожидается:
        - Клон имеет тот же фитнес и то же количество ребер графа.
        - Ссылки на объекты genome (адреса) различаются.
        - Изменение клона не протекает в оригинал.
        """
        ind = Individual(n_features=5)
        ind.genome.add_edge([0, 1], 'sum')
        ind.fitness = 0.95
        
        clone = ind.clone()
        self.assertEqual(clone.fitness, 0.95)
        self.assertEqual(len(clone.genome), 1)
        self.assertNotEqual(id(ind.genome), id(clone.genome))
        
        # Изменение клона не влияет на оригинал
        clone.genome.add_edge([2, 3], 'max')
        self.assertEqual(len(ind.genome), 1)
        self.assertEqual(len(clone.genome), 2)

    def test_mutation_stability(self):
        """Стресс-тест структурных мутаций (стабильность ссылок).
        
        Гарантирует, что при добавлении/удалении/изменении ребер 
        узлы остаются в границах валидных индексов и алгоритм не падает.
        """
        ops = GeneticOperators(mutation_rate=1.0)
        ind = Individual(n_features=10)
        ind.genome.add_edge([0, 1], 'sum')
        
        for _ in range(100):
            ind = ops.mutate(ind)
            self.assertTrue(all(0 <= i < 10 for e in ind.genome.edges.values() for i in e.node_indices))

    def test_crossover(self):
        """Проверка одноточечного скрещивания гиперграфов.
        
        Ожидается, что потомки суммарно содержат все родительские 
        уникальные признаки без потери генетического материала.
        """
        ops = GeneticOperators(crossover_rate=1.0)
        p1 = Individual(n_features=5)
        p1.genome.add_edge([0, 1], 'sum')
        p1.genome.add_edge([1, 2], 'sum')
        
        p2 = Individual(n_features=5)
        p2.genome.add_edge([3, 4], 'prod')
        
        offspring = ops.crossover(p1, p2)
        self.assertEqual(len(offspring), 2)
        # Суммарное количество уникальных ребер должно сохраниться
        total_edges = len(offspring[0].genome) + len(offspring[1].genome)
        self.assertEqual(total_edges, 3)

if __name__ == '__main__':
    unittest.main()
