import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from typing import Optional, Dict, Any

from .evolution import Individual

class FitnessEvaluator:
    """Оценивает приспособленность особи через обучение модели машинного обучения.
    
    Отвечает за расчет метрики качества (Fitness) особи. Признаки, 
    сгенерированные гиперграфом (X_hep), конкатенируются с оригинальными (X). 
    Затем sklearn-совместимая модель оценивается на расширенных данных 
    при помощи кросс-валидации.
    
    Attributes:
        X (np.ndarray): Исходная матрица признаков.
        y (np.ndarray): Вектор целевой переменной.
        problem_type (str): Тип задачи обучения ('regression' или 'classification').
        cv (int): Количество фолдов для кросс-валидации.
        n_jobs (Optional[int]): Количество параллельных потоков.
        cache (Dict[str, np.ndarray]): Внутренний кэш сгенерированных столбцов.
        complexity_penalty (float): Штраф за каждое лишнее ребро в гиперграфе 
            (используется для предотвращения "раздувания" графа).
        model_template (Any): Базовая sklearn-модель.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, problem_type: str = 'regression', 
                 cv: int = 3, model: Any = None, n_jobs: Optional[int] = None, 
                 complexity_penalty: float = 0.0):
        """Инициализирует оценщик фитнеса.
        
        Args:
            X (np.ndarray): Исходные признаки датасета.
            y (np.ndarray): Вектор целевых ответов.
            problem_type (str, optional): 'regression' или 'classification'. Defaults to 'regression'.
            cv (int, optional): Количество फोлдов кросс-валидации. Defaults to 3.
            model (Any, optional): Пользовательская sklearn-совместимая модель. 
                Если None, инициализируется базовый Random Forest. Defaults to None.
            n_jobs (Optional[int], optional): Потоки для вычислений. Defaults to None.
            complexity_penalty (float, optional): Сила штрафа топологической сложности. Defaults to 0.0.
        """
        self.X = X
        self.y = y
        self.problem_type = problem_type
        self.cv = cv
        self.n_jobs = n_jobs
        self.cache: Dict[str, np.ndarray] = {}
        self.complexity_penalty = complexity_penalty
        
        if model is not None:
            self.model_template = model
        else:
            if self.problem_type == 'regression':
                self.model_template = RandomForestRegressor(n_estimators=50, max_depth=5, 
                                                            n_jobs=-1, random_state=42)
            else:
                self.model_template = RandomForestClassifier(n_estimators=50, max_depth=5, 
                                                             n_jobs=-1, random_state=42)

    def evaluate(self, individual: Individual) -> float:
        """Вычисляет CV-скор модели на основе объединенных признаков (X_orig + X_hep).
        
        Note:
            Метод модифицирует переданный объект `individual`, напрямую записывая 
            результат вычисления в `individual.fitness`.
            
        Args:
            individual (Individual): Эволюционирующая особь для оценки.
            
        Returns:
            float: Скор приспособленности (R2 или F1-macro). В случае внутренней ошибки
                при кросс-валидации возвращает штрафное значение (-1e12).
                
        Examples:
            >>> evaluator = FitnessEvaluator(X, y)
            >>> ind = Individual(n_features=X.shape[1])
            >>> evaluator.evaluate(ind)
            0.875
        """
        # Генерация новых признаков с использованием кэша
        X_hep = individual.genome.transform(self.X, cache=self.cache)
        
        # Конкатенация признаков, если гиперграф сгенерировал хоть один столбец
        if X_hep.shape[1] == 0:
            X_augmented = self.X
        else:
            X_augmented = np.hstack([self.X, X_hep])
            
        model = clone(self.model_template)
        scoring = 'r2' if self.problem_type == 'regression' else 'f1_macro'
            
        try:
            scores = cross_val_score(model, X_augmented, self.y, cv=self.cv, 
                                     scoring=scoring, n_jobs=self.n_jobs)
            # Фитнес — это сырой скор минус штраф за ширину гиперграфа
            fitness = np.mean(scores) - self.complexity_penalty * len(individual.genome)
        except Exception as e:
            print(f"Evaluation error: {e}")
            fitness = -1e12
            
        individual.fitness = fitness
        return fitness
