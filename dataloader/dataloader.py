import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np

class MEAD(Dataset):
    def __init__(self, data_path, duration):
        super(MEAD, self).__init__()

        with open(data_path,"r") as f:
            self.datalist = json.load(f)
        self.time = duration

    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        row = self.datalist[idx]

        waveform, sample_rate = torchaudio.load(f"dataset/audios/{row['audio']}", num_frames= 48000* self.time)
        landmark =  torch.from_numpy(np.load(f"dataset/mp_landmarks/{row['landmark']}")).permute(2,0,1)[:,:60]
        # ilm = landmark[:,0,:]
        ilm = torch.mean(landmark, dim=1)
        # waveform = torch.load(f"dataset/vec2s/{row['name']}.pt")

        return {'name': row['name'], 'audio':waveform.float(), 'target':landmark.float(), 'ilm':ilm.float(), 'label':row['emotion']}