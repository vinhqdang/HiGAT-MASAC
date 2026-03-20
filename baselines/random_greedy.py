import numpy as np

class RandomGreedyBaseline:
    def __init__(self, config):
        self.num_vehicles = config['env']['num_vehicles']
        self.num_subbands = config['env']['num_subbands']
        self.max_f_mec = config['env']['rsu_mec_capacity_ghz']
        
    def select_macro_action_random(self, state_graph):
        # random subband
        subband = np.zeros((self.num_vehicles, self.num_subbands))
        for i in range(self.num_vehicles):
            sb_idx = np.random.randint(0, self.num_subbands)
            subband[i, sb_idx] = 1.0
            
        # random fmec 0 to 1
        fmec = np.random.uniform(0, 1, size=(self.num_vehicles,))
        return subband, fmec
        
    def select_micro_action_random(self, state_graph):
        # p_tx [0, 1] meaning [0, pmax]
        # alpha [0, 1]
        action = np.random.uniform(0, 1, size=(2,))
        return action
        
    def select_macro_action_greedy(self, state_graph):
        # Greedy logic would require looking at state graph 
        # For simplicity in this mock, greedy assigns equal f_mec and random subbands minimizing collision
        subband = np.zeros((self.num_vehicles, self.num_subbands))
        for i in range(self.num_vehicles):
            sb_idx = i % self.num_subbands
            subband[i, sb_idx] = 1.0
            
        # give everyone max possible equal share
        fmec = np.ones((self.num_vehicles,)) * 0.5 
        return subband, fmec

    def select_micro_action_greedy(self, state_graph):
        # Full power, full offload if beneficial
        action = np.array([1.0, 1.0])
        return action
