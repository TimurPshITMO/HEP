from fedot.core.repository.operation_types_repository import OperationTypesRepository, OperationMetaInfo
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot_hep.web.strategy import HypergraphStrategy

def register_hep():
    """
    Registers the HypergraphFeatureConstructor operation in FEDOT repositories.
    Allows FEDOT to recognize 'hypergraph_feature_constructor' as a valid data operation.
    """
    repo = OperationTypesRepository('data_operation')
    
    # Check if already registered
    if 'hypergraph_feature_constructor' in [op.id for op in repo._repo]:
        return

    # Define metadata
    meta = OperationMetaInfo(
        id='hypergraph_feature_constructor',
        input_types=[DataTypesEnum.table],
        output_types=[DataTypesEnum.table],
        task_type=[TaskTypesEnum.classification, TaskTypesEnum.regression],
        supported_strategies=HypergraphStrategy,
        allowed_positions=['primary'], # HEP must always be a primary node
        tags=['feature_engineering', 'hypergraph', 'hep'],
        presets=['fast_train', 'stable']
    )
    
    # Inject into repository
    # OperationTypesRepository._repo is a list of OperationMetaInfo
    repo._repo.append(meta)
    
    # Also need to register in the shared internal dict if it was already initialized
    # OperationTypesRepository.__initialized_repositories__ stores initialized repos per type
    if 'data_operation' in OperationTypesRepository.__initialized_repositories__:
         # Repositories are usually cached, we updated the instance above, 
         # but let's ensure it's propagated if needed.
         # Actually repo = OperationTypesRepository('data_operation') returns the singleton/cached instance
         pass

    print("Successfully registered 'hypergraph_feature_constructor' in FEDOT repository.")

def patch_fedot_for_hep():
    """
    Monkey-patches FEDOT to include HEP-specific rules and mutations.
    This allows the high-level Fedot API to use HEP without complex configuration.
    """
    import fedot.core.pipelines.verification as verification
    from fedot.api.api_utils.api_params_repository import ApiParamsRepository
    from fedot_hep.core.verification import has_no_multiple_hep_nodes, has_hep_node_as_primary
    from fedot_hep.evolution.mutation import hypergraph_mutation
    
    # 1. Inject validation rules
    # We add them to common_rules so they apply to all tasks
    if has_no_multiple_hep_nodes not in verification.common_rules:
        verification.common_rules.append(has_no_multiple_hep_nodes)
    if has_hep_node_as_primary not in verification.common_rules:
        verification.common_rules.append(has_hep_node_as_primary)
        
    # 2. Inject mutation
    # We wrap the original _get_default_mutations
    original_get_mutations = ApiParamsRepository._get_default_mutations
    
    @staticmethod
    def patched_get_mutations(*args, **kwargs):
        mutations = original_get_mutations(*args, **kwargs)
        # Ensure our mutation is in the list
        if hypergraph_mutation not in mutations:
            # We convert to list if it's a sequence/tuple
            mutations = list(mutations)
            mutations.append(hypergraph_mutation)
        return mutations

    ApiParamsRepository._get_default_mutations = patched_get_mutations
    print("FEDOT successfully patched with HEP rules and mutations.")

