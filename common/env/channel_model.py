import numpy as np

class ChannelModel:
    def __init__(self, config):
        self.config = config
        self.noise_power = 10 ** (config['env']['noise_power_dbm'] / 10.0) * 1e-3 # linear W
        self.shadowing_std = config['env']['shadowing_std_db']
        self.fc = 5.9e9 # carrier frequency 5.9 GHz typical for V2X
        
    def calculate_pathloss_v2v(self, distance):
        # 3GPP UMi NLoS or simplified V2V model
        # using a simple log-distance path loss model: PL(d) = 32.4 + 20log10(d) + 20log10(fc)
        d = np.clip(distance, 1.0, None)
        pathloss_db = 32.4 + 20 * np.log10(d) + 20 * np.log10(self.fc / 1e9)
        return pathloss_db
        
    def calculate_pathloss_v2i(self, distance):
        # 3GPP UMi LoS model
        d = np.clip(distance, 1.0, None)
        pathloss_db = 32.4 + 21.0 * np.log10(d) + 20 * np.log10(self.fc / 1e9)
        return pathloss_db
        
    def get_channel_gain(self, distance, link_type='v2v'):
        if distance == 0:
            return 1e-3 # Max gain safeguard

        if link_type == 'v2v':
            pl_db = self.calculate_pathloss_v2v(distance)
        elif link_type == 'v2i':
            pl_db = self.calculate_pathloss_v2i(distance)
        else:
            raise ValueError("Unknown link type")
            
        shadowing_db = np.random.normal(0, self.shadowing_std)
        # Fast fading (Rayleigh)
        rayleigh_fading = np.random.exponential(1.0)
        
        total_loss_db = pl_db + shadowing_db
        total_loss_linear = 10 ** (-total_loss_db / 10.0)
        
        channel_gain = total_loss_linear * rayleigh_fading
        return channel_gain
        
    def compute_sinr(self, rx_power, interference_powers):
        # All powers in linear scale
        total_interference = np.sum(interference_powers)
        sinr = rx_power / (self.noise_power + total_interference)
        return sinr
        
    def compute_capacity(self, sinr, bandwidth=1e6):
        # Shannon capacity
        return bandwidth * np.log2(1 + sinr)
