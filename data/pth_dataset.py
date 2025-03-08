import os
import torch
from torch.utils.data import Dataset


class PthDataset(Dataset):
    def __init__(self,
                 load_dir: str):   # e.g. ../data_test/PROTEINS/pth
        super().__init__()
        self.load_dir = load_dir
        self.file_list = os.listdir(load_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.load_dir, self.file_list[idx])
        data_dict = torch.load(
            open(file_path, 'rb')
            )
        return data_dict
