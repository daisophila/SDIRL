import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Replay_Buffer(object):
    def __init__(self, state_dim, action_dim, max_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
    
    def load_dataset(self, data):
        return data

    def sample(self):
        return {}


class Trajectories(Dataset):
    def __init__(self, data, top_n=None):
        if top_n:
            data = self.get_top_n_trajectories(data, top_n)

        self.states = torch.tensor(data['observations'], dtype=torch.float32)
        self.actions = torch.tensor(data['actions'], dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'states': self.states[idx],
            'actions': self.actions[idx]
        }

    def get_top_n_trajectories(self, dataset, n):
        num_samples = dataset['rewards'].shape[0]
        
        valid_keys = [k for k in dataset.keys() 
                    if isinstance(dataset[k], np.ndarray) and dataset[k].shape[0] == num_samples]
        
        trajectories = []
        current_traj = {k: [] for k in valid_keys}
        
        print(f"Filtering keys: {valid_keys}")

        for i in range(num_samples):
            for k in valid_keys:
                current_traj[k].append(dataset[k][i])
            
            if dataset['terminals'][i] or dataset['timeouts'][i]:
                traj_to_store = {k: np.array(v) for k, v in current_traj.items()}
                traj_to_store['return'] = np.sum(traj_to_store['rewards'])
                trajectories.append(traj_to_store)
                
                current_traj = {k: [] for k in valid_keys}

        trajectories.sort(key=lambda x: x['return'], reverse=True)
        top_n_list = trajectories[:n]
        
        print("-" * 30)
        print(f"Total trajectories found: {len(trajectories)}")
        for i, t in enumerate(top_n_list):
            print(f"Rank {i+1}: Return = {t['return']:.2f}, Steps = {len(t['rewards'])}")

        top_n_dict = {}
        valid_keys = top_n_list[0].keys()
        
        for k in valid_keys:
            if k == 'return':
                top_n_dict[k] = [t[k] for t in top_n_list]
            else:
                top_n_dict[k] = np.concatenate([t[k] for t in top_n_list], axis=0)
                
        return top_n_dict