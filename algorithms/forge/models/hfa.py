"""
Hierarchical Federated Aggregation (HFA)

Implements the federated aggregation scheme across clusters of BSs.
For simulation purposes, we assume M local agents and average their weights.
"""
import torch

class HFA:
    def __init__(self, num_rsus):
        self.num_rsus = num_rsus
        
    def aggregate_micro(self, micro_state_dicts, weights=None):
        """
        Intra-cluster aggregation of micro-policy weights.
        Averages the micro actors/critics across the local RSUs/Clusters.
        """
        if weights is None:
            weights = [1.0 / len(micro_state_dicts)] * len(micro_state_dicts)
            
        return self._federated_average(micro_state_dicts, weights)
        
    def aggregate_macro(self, macro_state_dicts, weights=None):
        """
        Inter-cluster global aggregation of macro-policy weights.
        """
        if weights is None:
            weights = [1.0 / len(macro_state_dicts)] * len(macro_state_dicts)
            
        return self._federated_average(macro_state_dicts, weights)
        
    def _federated_average(self, state_dicts, weights):
        """
        Given a list of state_dicts and mixing weights, returns the averaged state_dict.
        """
        avg_dict = {}
        for k in state_dicts[0].keys():
            # Initialize with the weighted first element
            avg_dict[k] = state_dicts[0][k].clone() * weights[0]
            
            # Accumulate the rest
            for i in range(1, len(state_dicts)):
                avg_dict[k] += state_dicts[i][k] * weights[i]
                
        return avg_dict
