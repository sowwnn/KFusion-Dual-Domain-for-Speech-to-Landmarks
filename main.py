from lightning import Trainer
from dataloader.dataloader import MEAD
from module.S2LM import S2LM
from torch.utils.data import DataLoader
import torch

import argparse


def main(datalist, duration, max_epochs):
    datas = MEAD(datalist, duration)
    dataloader = DataLoader(datas, batch_size=4, shuffle=True, num_workers=2)
    model = S2LM(batch=4)

    device = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(accelerator=device, max_epochs=max_epochs)
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datalist", type=str, default="dataset/datalist2.json")
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=100)

    args = parser.parse_args()
    main(args.datalist, args.duration, args.max_epochs)