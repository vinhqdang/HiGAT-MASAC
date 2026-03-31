import numpy as np

class ManhattanMobilityModel:
    def __init__(self, config):
        self.num_vehicles = config['env']['num_vehicles']
        self.num_rsus = config['env']['num_rsus']
        self.grid_blocks = config['env']['grid_size_blocks']
        self.block_length = config['env']['block_length_m']
        self.lane_width = config['env']['lane_width_m']
        
        self.grid_size_m = self.grid_blocks * self.block_length
        self.dt = config['rl']['micro_slot_duration_ms'] / 1000.0
        
        # Roads are on the grid lines: 0, 250, 500, 750 (for 3 blocks = 4 lines)
        self.road_coords = np.arange(self.grid_blocks + 1) * self.block_length
        
        # RSU positions (for 4 RSUs, place them at intersections)
        if self.num_rsus == 4:
            self.rsu_positions = np.array([
                [self.road_coords[1], self.road_coords[1]],
                [self.road_coords[1], self.road_coords[2]],
                [self.road_coords[2], self.road_coords[1]],
                [self.road_coords[2], self.road_coords[2]],
            ])
        else:
            # Fallback random placement
            self.rsu_positions = np.random.uniform(0, self.grid_size_m, (self.num_rsus, 2))
            
        self.reset()
        
    def reset(self):
        # Vehicle states: x, y, v_x, v_y
        self.positions = np.zeros((self.num_vehicles, 2))
        self.velocities = np.zeros((self.num_vehicles, 2))
        
        for i in range(self.num_vehicles):
            self._spawn_vehicle(i)
            
    def _spawn_vehicle(self, i):
        # Pick horizontal (0) or vertical (1) road
        is_horiz = np.random.rand() > 0.5
        road_idx = np.random.randint(0, len(self.road_coords))
        road_coord = self.road_coords[road_idx]
        
        pos = np.random.uniform(0, self.grid_size_m)
        speed = np.random.uniform(30.0, 60.0) * (1000.0 / 3600.0) # 30-60 km/h to m/s
        direction = np.random.choice([-1, 1])
        
        if is_horiz:
            self.positions[i] = [pos, road_coord]
            self.velocities[i] = [speed * direction, 0.0]
        else:
            self.positions[i] = [road_coord, pos]
            self.velocities[i] = [0.0, speed * direction]
            
    def step(self):
        self.positions += self.velocities * self.dt
        
        # Reflection at boundaries or turning at intersections
        for i in range(self.num_vehicles):
            x, y = self.positions[i]
            vx, vy = self.velocities[i]
            
            # Simple bounds check (reflection for simplicity)
            if x < 0 or x > self.grid_size_m:
                self.velocities[i, 0] *= -1
                self.positions[i, 0] = np.clip(x, 0, self.grid_size_m)
                
            if y < 0 or y > self.grid_size_m:
                self.velocities[i, 1] *= -1
                self.positions[i, 1] = np.clip(y, 0, self.grid_size_m)
                
            # Random turn at intersections (probability 0.1 at each step near intersection)
            # Find nearest intersection
            nearest_x_idx = np.argmin(np.abs(self.road_coords - x))
            nearest_y_idx = np.argmin(np.abs(self.road_coords - y))
            nearest_int_x = self.road_coords[nearest_x_idx]
            nearest_int_y = self.road_coords[nearest_y_idx]
            
            dist_to_int = np.sqrt((x - nearest_int_x)**2 + (y - nearest_int_y)**2)
            if dist_to_int < 10.0 and np.random.rand() < 0.05: # Changed from 0.1 to 0.05
                # We are near intersection, maybe turn
                speed = np.linalg.norm(self.velocities[i])
                if np.abs(vx) > 0: # moving horizontally -> turn vertically
                    self.velocities[i] = [0.0, speed * np.random.choice([-1, 1])]
                    self.positions[i, 0] = nearest_int_x # Snap to road
                else: # moving vertically -> turn horizontally
                    self.velocities[i] = [speed * np.random.choice([-1, 1]), 0.0]
                    self.positions[i, 1] = nearest_int_y # Snap to road
                    
    def get_distances(self):
        # distances[i, j] is dist between veh i and veh j
        # distances_rsu[i, m] is dist between veh i and RSU m
        diff_v2v = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        dist_v2v = np.linalg.norm(diff_v2v, axis=2)
        
        diff_v2i = self.positions[:, np.newaxis, :] - self.rsu_positions[np.newaxis, :, :]
        dist_v2i = np.linalg.norm(diff_v2i, axis=2)
        
        return dist_v2v, dist_v2i
