import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import torchaudio.transforms as T


class MEAD(Dataset):
    def __init__(self, data_path, duration, batch):
        super(MEAD, self).__init__()

        with open(data_path,"r") as f:
            datalist = json.load(f)
        ln = int(len(datalist)/batch)*batch
        self.datalist = datalist[:ln]
        self.time = duration
        self.mn = 0
        self.mx = 256

    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        row = self.datalist[idx]

        waveform, sample_rate = torchaudio.load(f"dataset/duration/audios/{row['audio']}", num_frames= 48000* self.time, normalize="True")
        landmark =  torch.from_numpy(np.load(f"dataset/duration/fa_landmarks/{row['landmark']}")).permute(0,2,1)[:30*self.time]
        landmark = (landmark - self.mn) / (self.mx - self.mn)
        # ilm = torch.mean(landmark, dim=1)
        name = row["name"].split('_')[0]
        try:
            ilm = torch.from_numpy(np.load(f"dataset/duration/fa_landmarks/{name}_front_neutral_level_1_001.npy")).permute(0,2,1)[0]
        except:
            ilm = landmark[0]
        waveform = T.Resample(48000, 16000, dtype=waveform.dtype)(waveform)
        waveform = torch.mean(waveform, dim=0)

        # waveform = torch.load(f"dataset/vec2s/M003_front_{row['name']}.pt")

        return {'name': row['name'], 'audio':waveform.float(), 'target':landmark.float(), 'ilm':ilm.float(), 'label':row['emotion']}