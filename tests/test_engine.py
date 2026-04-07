import unittest
import numpy as np
import os
import shutil
from hep_engine.evaluator import FitnessEvaluator
from hep_engine.optimizer import EvolutionaryOptimizer

class TestEngine(unittest.TestCase):
    """Модуль интеграционного тестирования всего пайплайна (компонентов).
    
    Проверяет, что все модули (Эвалюатор, Оптимизатор, Трекер и Геномы) 
    корректно соединяются вместе при прогоне end-to-end задачи.
    """
    
    def setUp(self):
        """Подготавливает моковые данные и временную папку логов для тестов."""
        self.X = np.random.rand(50, 5)
        # Намеренная нелинейность, чтобы RF было что выучить через гиперребра
        self.y = self.X[:, 0] * self.X[:, 1] + self.X[:, 2] 
        self.history_dir = 'test_history'

    def tearDown(self):
        """Уборка: удаляет временные директории после тестов."""
        if os.path.exists(self.history_dir):
            shutil.rmtree(self.history_dir)

    def test_optimizer_flow(self):
        """Интеграционный тест: запуск 3 поколений алгоритма.
        
        Гарантирует:
        - Инициализацию популяции
        - Успешный calculate фитнеса без сбоев RF (< -1e12 не ожидаем)
        - Создание дампов .json для трекера
        """
        # Чтобы трекер не упал при логировании
        os.makedirs(self.history_dir, exist_ok=True)
        evaluator = FitnessEvaluator(self.X, self.y, cv=2)
        optimizer = EvolutionaryOptimizer(pop_size=5, mut_rate=0.4, elitism_count=1)
        optimizer.tracker.output_dir = self.history_dir
        
        # Запускаем короткую эволюцию
        pop = optimizer.run(evaluator, n_generations=3, record_history=True)
        
        self.assertEqual(len(pop.individuals), 5)
        
        # Проверка, что история генерируется (сохраняется snapshot 0-го поколения)
        self.assertTrue(os.path.exists(os.path.join(self.history_dir, 'gen_000.json')))
        # Проверка создания финального дампа
        self.assertTrue(os.path.exists(os.path.join(self.history_dir, 'full_history.json')))
        
        # Проверка, что фитнес у лучшей особи был реально вычислен
        self.assertNotEqual(pop.best().fitness, -1e12)

if __name__ == '__main__':
    unittest.main()
