from typing import Optional
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.operation_parameters import OperationParameters
from fedot_hep.web.operation import HypergraphFeatureConstructor

class HypergraphStrategy(EvaluationStrategy):
    """
    Evaluation strategy for Hypergraph-Evolved Pipelines.
    Binds the HypergraphFeatureConstructor implementation to FEDOT's evaluation engine.
    """
    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation_impl = HypergraphFeatureConstructor(params)

    def fit(self, train_data):
        """Fits the hypergraph structure (or restores it from params)."""
        return self.operation_impl.fit(train_data)

    def predict(self, trained_operation, predict_data, output_mode: str = 'default'):
        """Applies hypergraph transformation."""
        return self.operation_impl.transform(predict_data)

    def _convert_to_output(self, prediction, predict_data):
        """Conversion is handled inside HypergraphFeatureConstructor.transform."""
        return prediction
