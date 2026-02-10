# Fedot.HEP: Hypergraph-Evolved Pipelines

[![Framework: FEDOT](https://img.shields.io/badge/Framework-FEDOT-blue.svg)](https://github.com/itmo-nss-team/FEDOT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)

**Fedot.HEP (Hypergraph-Evolved Pipelines)** is an extension for the [FEDOT](https://github.com/itmo-nss-team/FEDOT) AutoML framework that allows for the evolution of not only the overall ML pipeline structure but also the **interaction topology of input features** based on hypergraph theory.

## 🚀 Key Features

- **Feature Hypergraph Evolution**: The algorithm generates complex non-linear dependencies (hyperedges) combining any number of features.
- **Explainable AI (XAI)**: The `HEPExplainer` module makes the evolution process transparent, providing:
  - **Euler-style Plots**: Visualization of interaction clusters.
  - **Survival Analysis**: Formula persistence plots across generations.
  - **Feature Interaction Importance**: Contribution calculation based on participation in hyperedges.
- **Full FEDOT API Integration**: The HEP genome is fully compatible with FEDOT's mutation, crossover, and selection operators.

## 📁 Project Structure

- `fedot_hep/core/` — Mathematical core (HighPerformanceHypergraph based on CSR-matrices).
- `fedot_hep/evolution/` — Specialized mutation operators and validation rules.
- `fedot_hep/api/` — API registration and FEDOT patching logic.
- `fedot_hep/visualization/` — `HEPExplainer` module for research reporting.

## 🛠 Installation

The project uses [Poetry](https://python-poetry.org/) for dependency management:

```bash
poetry install
```

## 📖 Documentation

The project supports automatic documentation generation from docstrings.

To view locally:
```bash
poetry run mkdocs serve
```

## 📈 Examples

- `example_evolution.py` — Basic pipeline evolution with the HEP node.
- `example_visualization.py` — Geneation of research plots for interaction analysis.

## 📑 Scientific Background
The Fedot.HEP concept was developed to address anomaly detection and complex physical process analysis, where higher-order interactions are crucial.
