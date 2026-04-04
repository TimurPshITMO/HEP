import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from .evolution import Individual, Population, GeneticOperators
from typing import Optional, Dict, Any

class FitnessEvaluator:
    """
    Оценивает приспособленность особи через обучение модели на расширенных признаках.
    Поддерживает подключение любой sklearn-совместимой модели.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, problem_type: str = 'regression', cv: int = 3, model: Any = None, n_jobs: Optional[int] = None, complexity_penalty: float = 0.0):
        self.X = X
        self.y = y
        self.problem_type = problem_type
        self.cv = cv
        self.n_jobs = n_jobs
        self.cache: Dict[str, np.ndarray] = {}
        # Регуляризационный штраф за топологическую сложность (из тезиса).
        # 0.0 = выключен, например 0.005 = мягкая регуляризация.
        self.complexity_penalty = complexity_penalty
        
        # Установка модели по умолчанию, если она не передана
        if model is not None:
            self.model_template = model
        else:
            if self.problem_type == 'regression':
                self.model_template = RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)
            else:
                self.model_template = RandomForestClassifier(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)

    def evaluate(self, individual: Individual) -> float:
        """Вычисляет CV-скор модели на X_orig + X_hep."""
        # 1. Генерируем новые признаки (с использованием кэша)
        X_hep = individual.genome.transform(self.X, cache=self.cache)
        
        if X_hep.shape[1] == 0:
            X_augmented = self.X
        else:
            X_augmented = np.hstack([self.X, X_hep])
            
        # 2. Оцениваем модель (клонируем шаблон для чистоты CV)
        model = clone(self.model_template)
        scoring = 'r2' if self.problem_type == 'regression' else 'f1_macro'
            
        try:
            scores = cross_val_score(model, X_augmented, self.y, cv=self.cv, scoring=scoring, n_jobs=self.n_jobs)
            fitness = np.mean(scores) - self.complexity_penalty * len(individual.genome)
        except Exception as e:
            print(f"Evaluation error: {e}")
            fitness = -1e12
            
        individual.fitness = fitness
        return fitness
