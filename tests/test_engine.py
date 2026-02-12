import unittest
import numpy as np
import os
import shutil
from hep_engine.engine import FitnessEvaluator
from hep_engine.optimizer import EvolutionaryOptimizer

class TestEngine(unittest.TestCase):
    def setUp(self):
        # Подготовка простых данных для теста
        self.X = np.random.rand(50, 5)
        self.y = self.X[:, 0] * self.X[:, 1] + self.X[:, 2] # Намеренная нелинейность
        self.history_dir = 'test_history'

    def tearDown(self):
        if os.path.exists(self.history_dir):
            shutil.rmtree(self.history_dir)

    def test_optimizer_flow(self):
        evaluator = FitnessEvaluator(self.X, self.y, cv=2)
        optimizer = EvolutionaryOptimizer(pop_size=5, mut_rate=0.4, elitism_count=1)
        optimizer.tracker.output_dir = self.history_dir
        
        # Запускаем короткую эволюцию
        pop = optimizer.run(evaluator, n_generations=3)
        
        self.assertEqual(len(pop.individuals), 5)
        # Проверка, что история записывается
        self.assertTrue(os.path.exists(os.path.join(self.history_dir, 'gen_000.json')))
        self.assertTrue(os.path.exists(os.path.join(self.history_dir, 'full_history.json')))
        
        # Проверка, что фитнес посчитан
        self.assertNotEqual(pop.best().fitness, -1e12)

if __name__ == '__main__':
    unittest.main()
