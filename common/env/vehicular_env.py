import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .channel_model import ChannelModel
from .task_model import TaskModel
from .mobility_model import ManhattanMobilityModel

class VehicularEnv(gym.Env):
    """
    Dec-POMDP environment for HiGAT-MASAC.
    """
    def __init__(self, config):
        super(VehicularEnv, self).__init__()
        self.config = config
        
        self.num_vehicles = config['env']['num_vehicles']
        self.num_rsus = config['env']['num_rsus']
        self.K = config['env']['num_subbands']
        self.B = config['env']['subband_bandwidth_mhz'] * 1e6
        
        self.max_tx_power = 10 ** (config['env']['max_tx_power_dbm'] / 10.0) * 1e-3
        self.f_mec = config['env']['rsu_mec_capacity_ghz'] * 1e9
        
        self.w1 = config['env']['w1']
        self.w2 = config['env']['w2']
        self.w3 = config['env']['w3']
        self.w_vio = config['env']['w_vio']
        
        self.channel_model = ChannelModel(config)
        self.task_model = TaskModel(config)
        self.mobility_model = ManhattanMobilityModel(config)
        
        self.macro_steps_per_episode = int(config['env']['sim_duration_s'] / (config['rl']['macro_slot_duration_ms'] / 1000.0))
        self.micro_steps_per_macro = int(config['rl']['macro_slot_duration_ms'] / config['rl']['micro_slot_duration_ms'])
        
        # Action spaces (For reference only, actual RL agents will handle these dynamically)
        # Macro agent (RSU): Discrete logic for subband assignment, continuous for MEC allocation
        # Micro agent (Vehicle): Continuous for Tx power and offloading ratio
        
        self.reset()
        
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.mobility_model.reset()
        self.current_macro_step = 0
        self.current_micro_step = 0
        
        self.tasks = [self.task_model.generate_task() for _ in range(self.num_vehicles)]
        
        # State: graph representation
        mac_state = self._get_macro_state()
        return mac_state, {}
        
    def _cluster_vehicles(self):
        # Assign each vehicle to the nearest RSU
        _, dist_v2i = self.mobility_model.get_distances()
        clusters = np.argmin(dist_v2i, axis=1)
        return clusters, dist_v2i
        
    def _get_macro_state(self):
        # Returns graph info for each RSU
        dist_v2v, dist_v2i = self.mobility_model.get_distances()
        clusters, _ = self._cluster_vehicles()
        
        macro_states = []
        for m in range(self.num_rsus):
            # Vehicles in this cluster
            veh_in_cluster = np.where(clusters == m)[0]
            
            node_features = []
            for n in veh_in_cluster:
                feat = [
                    self.channel_model.get_channel_gain(dist_v2i[n, m], 'v2i'),
                    self.tasks[n]['data_size'],
                    self.tasks[n]['cpu_intensity'],
                    self.mobility_model.velocities[n, 0],
                    self.mobility_model.velocities[n, 1]
                ]
                node_features.append(feat)
                
            macro_states.append({
                'rsu_id': m,
                'cluster_vehicles': veh_in_cluster,
                'node_features': np.array(node_features),
                'dist_matrix': dist_v2v[np.ix_(veh_in_cluster, veh_in_cluster)]
            })
        return macro_states
        
    def get_micro_state(self, macro_actions):
        # Generates local neighborhood state for each vehicle given macro envelope
        # macro_actions contains envelope: {'subbands': X[n,k], 'f_mec_allocated': f[n,m]}
        dist_v2v, dist_v2i = self.mobility_model.get_distances()
        clusters, _ = self._cluster_vehicles()
        
        micro_states = []
        for n in range(self.num_vehicles):
            m = clusters[n]
            # Simple k-nearest neighbors (e.g., k=5)
            neighbors = np.argsort(dist_v2v[n])[:5] # Includes self
            
            env_subband = np.argmax(macro_actions['subbands'][n]) # Simple single subband max
            env_fmec = macro_actions['f_mec_allocated'][n]
            
            feat = [
                self.channel_model.get_channel_gain(dist_v2i[n, m], 'v2i'),
                self.tasks[n]['data_size'],
                env_subband,
                env_fmec
            ]
            
            neighbor_feats = []
            for nb in neighbors:
                nb_m = clusters[nb]
                neighbor_feats.append([
                    dist_v2v[n, nb],
                    self.channel_model.get_channel_gain(dist_v2i[nb, nb_m], 'v2i')
                ])
                
            micro_states.append({
                'veh_id': n,
                'local_features': np.array(feat),
                'neighbor_features': np.array(neighbor_feats)
            })
            
        return micro_states

    def step_micro(self, micro_actions, macro_actions):
        # micro_actions: Tx power p[n,k], offload ratio alpha[n]
        # Execute action, compute reward for micro slot
        self.mobility_model.step()
        self.current_micro_step += 1
        
        dist_v2v, dist_v2i = self.mobility_model.get_distances()
        clusters, _ = self._cluster_vehicles()
        
        rewards = np.zeros(self.num_vehicles)
        delays = np.zeros(self.num_vehicles)
        energies = np.zeros(self.num_vehicles)
        throughputs = np.zeros(self.num_vehicles)
        
        subbands = macro_actions['subbands'] # Shape (N, K)
        p_tx = micro_actions['p_tx'] # Shape (N, K)
        alpha = micro_actions['alpha'] # Shape (N)
        f_mec_alloc = macro_actions['f_mec_allocated'] # Shape (N)
        
        # Calculate interference and SINR
        rx_powers = np.zeros(self.num_vehicles)
        interference = np.zeros((self.num_vehicles, self.num_vehicles))
        
        for n in range(self.num_vehicles):
            m = clusters[n]
            k_n = np.argmax(subbands[n]) # assume 1 subband per veh for now
            # rx power at RSU
            rx_powers[n] = p_tx[n, k_n] * self.channel_model.get_channel_gain(dist_v2i[n, m], 'v2i')
            
            # Interference from others using the same subband
            for j in range(self.num_vehicles):
                if n != j and np.argmax(subbands[j]) == k_n:
                    m_j = clusters[j]
                    interference[n, j] = p_tx[j, k_n] * self.channel_model.get_channel_gain(dist_v2v[j, n], 'v2v')
        
        for n in range(self.num_vehicles):
            sinr = self.channel_model.compute_sinr(rx_powers[n], interference[n])
            rate = self.channel_model.compute_capacity(sinr, self.B)
            throughputs[n] = rate / 1e6 # Mbps
            
            task = self.tasks[n]
            t_total, _, _ = self.task_model.compute_latency(
                task['data_size'], task['cpu_intensity'], alpha[n], rate, f_mec_alloc[n]
            )
            e_total, _, _ = self.task_model.compute_energy(
                task['data_size'], task['cpu_intensity'], alpha[n], p_tx[n].sum(), rate
            )
            
            delays[n] = t_total
            energies[n] = e_total
            
            vio_penalty = self.w_vio if t_total > self.task_model.max_delay else 0
            
            # Reward: -w1 * T - w2 * E + w3 * throughput - vio
            rewards[n] = -self.w1 * t_total - self.w2 * e_total + self.w3 * throughputs[n] - vio_penalty
            
        next_micro_state = self.get_micro_state(macro_actions)
        return next_micro_state, rewards, False, {'delays': delays, 'energies': energies, 'throughputs': throughputs}

    def step_macro(self, macro_actions):
        # Simulate over micro_steps_per_macro
        
        macro_rewards = np.zeros(self.num_rsus)
        total_delays = np.zeros(self.num_vehicles)
        total_energies = np.zeros(self.num_vehicles)
        total_throughputs = np.zeros(self.num_vehicles)
        
        # We need a dummy micro agent for the 'step_macro' if we are just stepping the macro env directly.
        # However, normally the main training loop handles interleaving macro and micro steps.
        # We leave this to the outer loop.
        pass
        
    def render(self):
        pass
        
    def close(self):
        pass
