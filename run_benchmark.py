import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score

from hep_engine.evaluator import FitnessEvaluator
from hep_engine.optimizer import EvolutionaryOptimizer
from hep_engine.visualizer import HEPVisualizer

import sys

def generate_complex_data(n_samples=1000, n_features=12):
    """
    Генерирует данные с явными сложными взаимодействиями.
    f0 * f1, f2^2, f3 * f4 * f5, sin(f6)*f7
    """
    X = np.random.uniform(-3, 3, (n_samples, n_features))
    y = (X[:, 0] * X[:, 1] + 
         X[:, 2]**2 + 
         X[:, 3] * X[:, 4] * X[:, 5] + 
         np.sin(X[:, 6]) * X[:, 7] +
         np.random.normal(0, 0.2, n_samples))
    
    feature_names = [f"feat_{i}" for i in range(n_features)]
    return X, y, feature_names

def main():
    print("=== Standalone HEP Benchmark (V5 Proper) ===")
    sys.stdout.flush()
    
    # 1. Подготовка данных
    X, y, feature_names = generate_complex_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Baseline: Random Forest на сырых данных
    rf_baseline = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    baseline_cv = cross_val_score(rf_baseline, X_train, y_train, cv=3, scoring='r2')
    print(f"Baseline RF (Raw) R2 CV: {np.mean(baseline_cv):.4f}")
    
    # 3. Эволюция HEP
    evaluator = FitnessEvaluator(X_train, y_train, cv=3)
    optimizer = EvolutionaryOptimizer(pop_size=40, mut_rate=0.4, cross_rate=0.5, elitism_count=3)
    
    print("\nStarting HEP Evolution (50 generations)...")
    final_pop = optimizer.run(evaluator, n_generations=50)
    best_ind = final_pop.best()
    
    print(f"\nBest Individual Fitness (CV): {best_ind.fitness:.4f}")
    print(f"Best Individual Edges: {len(best_ind.genome)}")
    
    # 4. Final Evaluation: Augmented Data
    X_hep_train = best_ind.genome.transform(X_train)
    X_aug_train = np.hstack([X_train, X_hep_train])
    
    X_hep_test = best_ind.genome.transform(X_test)
    X_aug_test = np.hstack([X_test, X_hep_test])
    
    # Настраиваем модель чуть сильнее для использования новых фичей
    rf_hep = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    rf_hep.fit(X_aug_train, y_train)
    y_pred = rf_hep.predict(X_aug_test)
    hep_score = r2_score(y_test, y_pred)
    
    # Baseline Test Score
    rf_baseline.fit(X_train, y_train)
    baseline_score = r2_score(y_test, rf_baseline.predict(X_test))
    
    print("\n" + "="*40)
    print(f"RESULT: Baseline R2: {baseline_score:.4f}")
    print(f"RESULT: Standalone HEP R2: {hep_score:.4f}")
    print(f"IMPROVEMENT: {((hep_score - baseline_score)/baseline_score)*100:.2f}%")
    print("="*40)
    
    # 5. Визуализация результатов
    print("\nGenerating visualizations...")
    viz = HEPVisualizer(output_dir='benchmark_results')
    
    # Финальный гиперграф
    viz.plot_individual({
        'edges': [{'nodes': e.node_indices, 'func': e.function_name} for e in best_ind.genome.edges.values()],
        'fitness': best_ind.fitness
    }, n_features=len(feature_names), title="Final Evolved Hypergraph", save_path="benchmark_results/best_genome.png")
    
    # Генерация раскадровки эволюции
    viz.generate_evolution_frames('benchmark_results/history/full_history.json')
    
    print("\nBenchmark completed. Results saved in 'benchmark_results/'.")

if __name__ == "__main__":
    main()
