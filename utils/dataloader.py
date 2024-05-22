import torch
import torchvision
from torch.utils.data import Dataset
import json
import numpy as np
import pandas as pd



class MEAD(Dataset):
    def __init__(self, data_path):
        super(MEAD, self).__init__()

        with open("ultis/datalist.json","r") as f:
            self.datalist = json.load(f)

    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        row = self.datalist[idx]
        img = torchvision.io.VideoReader(row['video'], "video")
        img = torchvision.io.VideoReader(row['video'], "video")



    

