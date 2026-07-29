import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class VSL400KeypointDataset3D(Dataset):
    def __init__(self, npz_path, max_len=60):
        data = np.load(npz_path, allow_pickle=True)
        self.X = data['X']
        self.y = np.array(data['y'], dtype=np.int64)
        self.max_len = max_len

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_sample = np.array(self.X[idx], dtype=np.float32)
        seq_len = x_sample.shape[0]
        
        # Determine the spatial shape, for example (76, 3) or (228,)
        spatial_shape = x_sample.shape[1:] 
        
        if seq_len >= self.max_len:
            x_padded = x_sample[:self.max_len]
        else:
            padded_shape = (self.max_len,) + spatial_shape
            x_padded = np.zeros(padded_shape, dtype=np.float32)
            x_padded[:seq_len] = x_sample

        return torch.from_numpy(x_padded), torch.tensor(self.y[idx], dtype=torch.long)

def get_dataloaders(train_path, val_path, test_path=None, batch_size=32, max_len=60, num_workers=2):
    train_dataset = VSL400KeypointDataset3D(train_path, max_len=max_len)
    val_dataset = VSL400KeypointDataset3D(val_path, max_len=max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    if test_path:
        test_dataset = VSL400KeypointDataset3D(test_path, max_len=max_len)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        return train_loader, val_loader, test_loader
        
    return train_loader, val_loader
