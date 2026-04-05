# HEP: Hypergraph Evolution for Pipelines

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)
![sklearn](https://img.shields.io/badge/sklearn-compatible-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Статус-Research-lightgrey?style=flat-square)
![Kaggle](https://img.shields.io/badge/Бенчмарки-Kaggle-20BEFF?style=flat-square&logo=kaggle)

**HEP** — фреймворк автоматической генерации признаков (AutoFE), использующий генетические алгоритмы на динамических гиперграфах для синтеза компактных нелинейных трансформаций признаков.

---

## Принцип работы

Задача AutoFE состоит в нахождении отображения:

$$\varphi: \mathbb{R}^d \to \mathbb{R}^{d+k}, \quad X_{\text{aug}} = \bigl[X \mid \varphi(X)\bigr]$$

Каждый компонент $\varphi$ задаётся гиперребром $e = (S,\, \sigma)$:

$$\varphi_e(X) = \sigma\!\bigl(\{X_{:,j} \mid j \in S\}\bigr), \quad \sigma \in \{\text{sum, mean, max, min, std, prod}\}$$

Геном особи — набор гиперрёбер $G = \{e_1, \ldots, e_k\}$. Функция приспособленности:

$$F(G) = \text{CV\_score}\!\bigl(f,\, X_{\text{aug}},\, y\bigr) - \lambda \cdot |E|$$

где $\lambda$ — коэффициент L0-регуляризации, $f$ — нижестоящая модель, $|E|$ — число рёбер в геноме.

---

## Ключевые результаты

Бенчмарки проводились на 5 датасетах (3 зерна, 20 поколений, `pop_size=20`):

| Метрика | Значение |
|---|---|
| Максимальный прирост качества над исходными признаками | **+0,794** |
| Средний прирост качества над исходными признаками | **+0,074** |
| Лучший метод на California Housing (R²) | **HEP — 0,714** |
| Среднее число признаков, создаваемых HEP | **3,1** |
| Среднее число признаков, создаваемых Poly2 | **159,6** |

HEP лидирует среди всех методов FE на California Housing и создаёт пространство признаков в **52 раза компактнее** полиномиального расширения при конкурентоспособном качестве предсказания.

---

## Генетический алгоритм и структура гиперграфа

### Гиперграф как геном

Каждая особь популяции хранит гиперграф $G = (V, E)$:
- **Вершины** $V = \{0, \ldots, d-1\}$ — индексы исходных признаков
- **Гиперрёбра** $E$ — набор операций над подмножествами признаков

Каждое ребро $e = (S, \sigma)$ идентифицируется MD5-хэшем пары $(S, \sigma)$. Это гарантирует уникальность рёбер внутри генома и позволяет применять кэширование: если одинаковое ребро встречается у разных особей, его трансформация вычисляется ровно один раз.

### Инициализация

Популяция инициализируется с **пустыми геномами** — без рёбер. Рёбра появляются исключительно через мутацию `add`, что обеспечивает постепенный рост сложности под давлением отбора.

### Операторы мутации

На каждой итерации особь мутирует с вероятностью `mut_rate`. Тип операции определяется числом существующих рёбер:

| Число рёбер | Доступные операции |
|---|---|
| 0 | `add` |
| 1 | `add`, `modify` |
| > 1 | `add`, `remove`, `modify` |

- **`add`** — добавляет новое ребро: случайное подмножество $S$ из 2–4 признаков и случайная функция $\sigma$
- **`remove`** — удаляет случайное ребро из генома
- **`modify`** — изменяет существующее ребро: с вероятностью 0,5 меняет функцию $\sigma$, иначе заменяет один из узлов $S$ на случайный другой

### Скрещивание

Скрещивание выполняется между двумя родителями с вероятностью `cross_rate`. Используется одноточечный обмен списками рёбер:

```
parent1: [e1, e2 | e3, e4]      offspring1: [e1, e2, e7, e8]
parent2: [e5, e6 | e7, e8]  ->  offspring2: [e5, e6, e3, e4]
```

Уникальность сигнатур рёбер сохраняется автоматически через словарь `{signature: edge}`.

### Селекция и элитизм

На каждом поколении применяется **турнирная селекция** с размером турнира $k=3$: из трёх случайных особей выбирается лучшая по фитнесу. Элитизм гарантирует, что `elitism_count` лучших особей переходят в следующее поколение без изменений.

### Двухуровневое кэширование

- **Уровень рёбер** (`FitnessEvaluator`): трансформации `X_hep` для каждого ребра кэшируются по MD5-сигнатуре. При вычислении фитнеса нового генома уже известные рёбра не пересчитываются.
- **Уровень геномов** (`EvolutionaryOptimizer`): финальный фитнес генома кэшируется по хэшу всего набора рёбер. Идентичные геномы в разных поколениях не переоцениваются.

---

## Архитектура

```
hep_engine/
    core.py          Hyperedge и Hypergraph; MD5-дедупликация рёбер
    evolution.py     Individual, Population, GeneticOperators (мутация / скрещивание)
    evaluator.py     FitnessEvaluator; CV-оценка с кэшем трансформаций
    optimizer.py     EvolutionaryOptimizer; турнирная селекция, элитизм, кэш геномов
    tracker.py       EvolutionTracker; JSON-история поколений для post-hoc анализа
    visualizer.py    HEPVisualizer; рендеринг через HyperNetX, анимация эволюции

benchmarks/
    hep_wrapper.py            sklearn-совместимый HEPTransformer (BaseEstimator)
    benchmark_1_baselines.py  Исходные признаки vs HEP-дополненные
    benchmark_2_fe_methods.py HEP vs NoFE / Poly2 / Poly2Interact / PCA_expand / FeatureTools
    benchmark_3_ablation.py   OFAT-анализ чувствительности к гиперпараметрам
    datasets.py               Загрузчики датасетов
    reporting.py              CSV, графики, кривые сходимости, тест Фридмана
    run_all.py                CLI-точка входа для полного набора бенчмарков
```

---

## Установка

```bash
# Через poetry
poetry install

# С опциональными зависимостями для бенчмарков
poetry install -E benchmarks
```

```bash
# Через pip
pip install numpy scikit-learn hypernetx
```

---

## Быстрый старт

```python
from benchmarks.hep_wrapper import HEPTransformer

hep = HEPTransformer(
    pop_size=20,
    n_generations=20,
    problem_type='regression',    # или 'classification'
    complexity_penalty=0.005,     # L0-регуляризация; 0.0 — отключить
    timeout=300,
)

hep.fit(X_train, y_train)
X_aug = hep.transform(X_test)    # (n_samples, n_original + n_hep_features)

print(f"Добавлено HEP-признаков: {hep.n_hep_features_}")
print(f"Лучший фитнес:           {hep.best_fitness_:.4f}")
```

`HEPTransformer` полностью совместим со sklearn и встраивается в `Pipeline`:

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

pipe = Pipeline([
    ('hep',   HEPTransformer(pop_size=20, n_generations=20, complexity_penalty=0.005)),
    ('model', RandomForestRegressor()),
])
pipe.fit(X_train, y_train)
```

---

## Запуск бенчмарков

```bash
# Полный набор (все 3 бенчмарка)
python benchmarks/run_all.py --benchmark all

# Быстрый smoke-тест
python benchmarks/run_all.py \
    --benchmark all --datasets synthetic_regression \
    --seeds 0 1 --hep-generations 5 --hep-pop-size 10 --hep-timeout 60

# Отдельный бенчмарк
python benchmarks/run_all.py --benchmark 2 --seeds 0 1 2
```

Результаты сохраняются в `benchmarks/results/`.

---

## Тесты

```bash
poetry run python -m pytest tests/ -v
```
