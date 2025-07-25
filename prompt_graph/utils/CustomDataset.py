import numpy as np
import torch
from torch_geometric.data import InMemoryDataset


class CustomDataset(InMemoryDataset):
    def __init__(self, dataset):
        super().__init__()
        self.data_list = dataset

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if isinstance(idx, int):  # **如果是单个索引**
            return self.data_list[idx]
        elif isinstance(idx, list) or isinstance(idx, torch.Tensor) or isinstance(idx, np.ndarray):  # **如果是多个索引**
            return [self.data_list[i] for i in idx]
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")


