import numpy as np

class TaskModel:
    def __init__(self, config):
        self.config = config
        self.kappa = 1e-28 # Effective switched capacitance for energy
        self.d_min = config['env']['task_data_mbits_min'] * 1e6 # bits
        self.d_max = config['env']['task_data_mbits_max'] * 1e6
        self.c_min = config['env']['task_cpu_cycles_per_bit_min']
        self.c_max = config['env']['task_cpu_cycles_per_bit_max']
        self.f_loc = config['env']['vehicle_cpu_capacity_ghz'] * 1e9 # cycles/sec
        self.max_delay = config['env']['max_delay_ms'] / 1000.0 # sec
        
    def generate_task(self):
        d_n = np.random.uniform(self.d_min, self.d_max)
        c_n = np.random.uniform(self.c_min, self.c_max)
        return {'data_size': d_n, 'cpu_intensity': c_n}
        
    def compute_latency(self, d_n, c_n, alpha, uplink_rate, f_mec_allocated):
        # T_local = (1-α)*d_n*c_n / f_loc
        t_local = ((1 - alpha) * d_n * c_n) / self.f_loc
        
        # T_offload = T_uplink + T_mec
        # If rate is very small or zero, cap delay at 2x max simulation deadline
        t_uplink = (alpha * d_n / uplink_rate) if uplink_rate > 1e-9 else (self.max_delay * 2)
        
        # Avoid division by zero, min allocation fallback 
        f_mec_allocated = max(f_mec_allocated, 1e-9)
        t_mec = (alpha * d_n * c_n) / f_mec_allocated
        
        t_offload = t_uplink + t_mec
        
        # Total latency is max of parallel local and offload execution
        t_total = max(t_local, t_offload)
        return t_total, t_local, t_offload
        
    def compute_energy(self, d_n, c_n, alpha, transmit_power_w, uplink_rate):
        t_uplink = (alpha * d_n / uplink_rate) if uplink_rate > 1e-9 else (self.max_delay * 2)
        
        # E_comm = P_tx * T_uplink
        e_comm = transmit_power_w * t_uplink if alpha > 0 else 0.0
        
        # E_comp = κ * f_loc^2 * (1-α)*d_n*c_n
        e_comp = self.kappa * (self.f_loc ** 2) * (1 - alpha) * d_n * c_n
        
        e_total = e_comm + e_comp
        return e_total, e_comm, e_comp
