import unittest
import numpy as np
from hep_engine.core import Hyperedge, Hypergraph

class TestCore(unittest.TestCase):
    def test_hyperedge_signature(self):
        # Одинаковые ребра должны иметь одинаковые сигнатуры
        e1 = Hyperedge([0, 1, 2], 'sum')
        e2 = Hyperedge([2, 1, 0], 'sum')
        self.assertEqual(e1.signature, e2.signature)
        
        # Разные функции — разные сигнатуры
        e3 = Hyperedge([0, 1, 2], 'mean')
        self.assertNotEqual(e1.signature, e3.signature)

    def test_hypergraph_management(self):
        hg = Hypergraph(n_features=10)
        sig = hg.add_edge([0, 1], 'sum')
        self.assertEqual(len(hg), 1)
        
        # Повторное добавление не должно увеличивать размер
        hg.add_edge([1, 0], 'sum')
        self.assertEqual(len(hg), 1)
        
        hg.remove_edge(sig)
        self.assertEqual(len(hg), 0)

    def test_transformation(self):
        X = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        hg = Hypergraph(n_features=3)
        hg.add_edge([0, 1, 2], 'sum') # 1+2+3=6, 4+5+6=15
        hg.add_edge([0, 2], 'prod')   # 1*3=3, 4*6=24
        
        res = hg.transform(X)
        self.assertEqual(res.shape, (2, 2))
        np.testing.assert_array_equal(res[:, 0], [6, 15])
        np.testing.assert_array_equal(res[:, 1], [3, 24])

    def test_caching(self):
        X = np.random.rand(10, 5)
        hg = Hypergraph(n_features=5)
        sig = hg.add_edge([0, 1], 'sum')
        
        cache = {}
        # Первый проход — наполнение кэша
        hg.transform(X, cache=cache)
        self.assertIn(sig, cache)
        
        # Подменяем значение в кэше, чтобы проверить, что transform берет из него
        cache[sig] = np.zeros(10)
        res = hg.transform(X, cache=cache)
        np.testing.assert_array_equal(res[:, 0], np.zeros(10))

if __name__ == '__main__':
    unittest.main()
