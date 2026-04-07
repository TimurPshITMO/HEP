# %% [markdown]
# # Standalone HEP: Эволюция Признаков Высокого Порядка
# 
# **Hypergraph-Evolved Pipelines (HEP)** — это инновационный подход к автоматическому конструированию признаков (AutoFE), основанный на представлении генома как **динамического гиперграфа**. В отличие от классических методов, которые ограничены линейными или простыми полиномиальными комбинациями, HEP способен обнаруживать и синтезировать сложные нелинейные зависимости в данных.
# 
# ### Ключевые преимущества HEP:
# - **Интерпретируемость:** Каждое ребро гиперграфа — это конкретная математическая операция.
# - **Эффективность поиска:** Эволюционный механизм фокусно исследует наиболее перспективные комбинации признаков.
# - **Гибкость:** Возможность ограничения набора функций (max, min, std и др.) в зависимости от специфики задачи.

# %%
import sys, os, inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

# 1. Настройка путей (КРИТИЧНО для импорта hep_engine)
current_path = os.getcwd()
project_root = os.path.abspath(os.path.join(current_path, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. Настройка авто-перезагрузки
# %load_ext autoreload
# %autoreload 2

# 3. Импорт компонентов
try:
    from hep_engine.evaluator import FitnessEvaluator
    from hep_engine.optimizer import EvolutionaryOptimizer
    from hep_engine.visualizer import HEPVisualizer
    from hep_engine.core import Hypergraph
    print("✅ Библиотека hep_engine успешно загружена.")
    print(f"   Путь: {inspect.getfile(FitnessEvaluator)}")
    print(f"   Аргументы __init__: {inspect.signature(FitnessEvaluator.__init__)}")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print(f"   Текущий sys.path: {sys.path[:3]}...")

# %matplotlib inline

# %% [markdown]
# ## 2. Генерация "Невозможного" Датасета
# 
# Для демонстрации силы HEP мы создадим синтетический датасет, в котором целевая переменная $y$ зависит от базовых признаков через функции, которые сложно аппроксимировать стандартным моделям (например, случайному лесу без глубокого тюнинга):
# 
# 1. **Произведение трёх признаков:** `prod(feat_0, feat_1, feat_2)` — нелинейное 3-факторное взаимодействие.
# 2. **Стандартное отклонение:** `std(feat_3, feat_4, feat_5, feat_6)` — агрегативный признак группы.
# 3. **Максимум двух признаков:** `max(feat_0, feat_2)` — нелинейный выбор.
# 4. **Минимум трёх признаков:** `min(feat_5, feat_6, feat_7)` — обратная нелинейность.
# 5. **Простое произведение:** `prod(feat_1, feat_3)` — парное взаимодействие.

# %%
def generate_showcase_data(n_samples=500, n_features=10):
    np.random.seed(42)
    X = np.random.uniform(-2, 2, (n_samples, n_features))
    
    # 5 скрытых нелинейных зависимостей равного веса
    term1 = X[:, 0] * X[:, 1] * X[:, 2]             # prod(0,1,2)
    term2 = np.std(X[:, 3:7], axis=1)               # std(3,4,5,6)
    term3 = np.max(X[:, [0, 2]], axis=1)             # max(0,2)
    term4 = np.min(X[:, [5, 6, 7]], axis=1)          # min(5,6,7)
    term5 = X[:, 1] * X[:, 3]                        # prod(1,3)
    
    y = 2*term1 + 3*term2 + 3*term3 + 3*term4 + 2*term5 + np.random.normal(0, 0.1, n_samples)
    feature_names = [f"feat_{i}" for i in range(n_features)]
    return X, y, feature_names

X, y, feature_names = generate_showcase_data(n_samples=500, n_features=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Данные сгенерированы: {X.shape[0]} примеров, {X.shape[1]} базовых признаков.")

# %% [markdown]
# ## 3. Сравнение Baseline
# 
# Прежде чем запускать эволюцию, оценим качество стандартных подходов:
# - **Raw RF:** Обучение на исходных признаках. Модель вынуждена сама строить дерево решений, пытаясь уловить нелинейности.
# - **Polynomial Features (d=2):** Классическое расширение признакового пространства до попарных произведений. Это приводит к быстрому росту размерности данных, но не гарантирует нахождения сложных функций (например, `std`).

# %%
# Эталонная модель для всех тестов (очень быстрая)
ref_model = RandomForestRegressor(n_estimators=15, max_depth=3, random_state=42)

# 1. RF Raw
rf_raw = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42).fit(X_train, y_train)
r2_raw = r2_score(y_test, rf_raw.predict(X_test))

# 2. RF + Polynomial Features (d=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
rf_poly = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42).fit(X_poly_train, y_train)
r2_poly = r2_score(y_test, rf_poly.predict(X_poly_test))

print(f"R2 Score (RF Raw): {r2_raw:.4f}")
print(f"R2 Score (Polynomial d2): {r2_poly:.4f}")

# %% [markdown]
# ## 4. Эволюционный HEP (Честное сравнение)
# 
# Теперь применим **EvolutionaryOptimizer**. Основная идея заключается в том, что гиперграф эволюционирует, меняя связи между признаками и применяя к ним агрегационные функции. 
# 
# **Настройки эксперимента:**
# - **Размер популяции:** 10 особей.
# - **Кроссовер и Мутация:** Высокие вероятности (0.5-0.6) для активного исследования пространства.
# - **Элитизм:** Сохранение лучших 5 решений для стабильности.
# - **Набор функций:** `['max', 'min', 'std', 'prod']`.

# %%
# Создаем эвалюатор с подключаемой моделью
evaluator = FitnessEvaluator(X_train, y_train, model=ref_model, n_jobs=2, complexity_penalty=0.007)

optimizer = EvolutionaryOptimizer(
    pop_size=30,
    mut_rate=0.7, 
    cross_rate=0.7, 
    elitism_count=3,
    available_functions=['max', 'min', 'std', 'prod']
)

print("Запуск эволюции гиперграфа...")
pop = optimizer.run(evaluator, n_generations=200, timeout=1000, labels=feature_names, record_history=True)
best_ind = pop.best()

print(f"\nЛучший фитнес (CV R2): {best_ind.fitness:.4f}")

# %% [markdown]
# ## 5. Оценка и Визуализация
# 
# После завершения эволюции мы берем лучшую особь (гиперграф) и используем её как **Feature Transformer**. Новые признаки добавляются к исходным данным, и финальная модель (Random Forest) обучается на этом расширенном наборе.

# %%
# Финальная оценка на отложенной выборке
X_hep_train = best_ind.genome.transform(X_train, cache=evaluator.cache)
X_hep_test = best_ind.genome.transform(X_test)

X_aug_train = np.hstack([X_train, X_hep_train])
X_aug_test = np.hstack([X_test, X_hep_test])

final_rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42).fit(X_aug_train, y_train)
r2_hep = r2_score(y_test, final_rf.predict(X_aug_test))

print(f"Итоговый R2 (HEP + RF): {r2_hep:.4f}")

# Сводный график
plt.figure(figsize=(10, 5))
results = [r2_raw, r2_poly, r2_hep]
labels = ['Raw', 'Poly d2', 'HEP (Our)']
colors = ['#cccccc', '#8888ff', '#22cc88']
plt.bar(labels, results, color=colors)
plt.ylim(0, 1.0)
for i, v in enumerate(results): plt.text(i, v+0.02, f"{v:.4f}", ha='center', fontweight='bold')
plt.title("Сравнение эффективности AutoFE методов")
plt.savefig("comparison.png")
plt.close()

# %%
# Визуализация
viz = HEPVisualizer(output_dir='.')
print("Визуализация архитектуры найденных признаков:")
viz.plot_individual(best_ind, n_features=len(feature_names), title="Best Evolved Feature Structure", save_path="best_genome.png")

# %% [markdown]
# ## 6. Символьная декодировка

# %%
print("Математические формулы признаков:")
for i, edge in enumerate(best_ind.genome.edges.values()):
    nodes = [feature_names[idx] for idx in edge.node_indices]
    print(f"  [Edge {i}]: {edge.function_name}({', '.join(nodes)})")


