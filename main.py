from lightning import Trainer
from dataloader.dataloader import MEAD
from module.S2LM import S2LM
from torch.utils.data import DataLoader
import torch
import lightning as L
import argparse


torch.set_float32_matmul_precision("high")

def main(datalist, duration, max_epochs, batch):

    datas = MEAD(datalist, duration=duration, batch=batch)
    trainsize = int(len(datas) * (90 /100) / batch) * batch
    testsize = len(datas) - trainsize
    train_dataset, test_dataset = torch.utils.data.random_split(datas, [trainsize, testsize])
    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True,  num_workers=16)
    test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False,  num_workers=16)

    model = S2LM(batch=batch, init_lr=1e-3, num_of_landmarks=68)
    torch.set_float32_matmul_precision("high")
    trainer = Trainer(max_epochs=max_epochs, default_root_dir="weights", fast_dev_run=True, check_val_every_n_epoch=4)
    trainer.fit(model, train_dataloader, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datalist", type=str, default="dataset/duration/fa_datalist.json")
    parser.add_argument("--duration", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)

    args = parser.parse_args()
    main(args.datalist, args.duration, args.max_epochs, args.batch)