import os
import numpy as np
import torch
import mne #add package
from typing import Tuple
from termcolor import cprint
from torchvision import datasets, transforms, models
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        #add FFT and noise filter
        scaler = StandardScaler()
        #threshold of amp
        ac = 2
        
#       k = 0
#       for k in range(len(self.X)):
#          n_array = self.X[k].clone().numpy()
#          n_fft = np.fft.fft(n_array)
#          n_fft[n_fft < ac] = 0
#          n_ifft = np.fft.ifft(n_fft)
#          n_ifft_real = n_ifft.real
#          scaler.fit(n_ifft_real)
#          n_standard = scaler.transform(n_ifft_real)
#          self.X[k] = torch.from_numpy(n_standard.copy())
           
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
