from lightning import Trainer
from dataloader.dataloader import MEAD
from module.KFusionLM import KFusionLM
from torch.utils.data import DataLoader
import torch
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

torch.set_float32_matmul_precision("high")

def init_data(args):
    batch = args['batch']
    datas = MEAD(args['datalist'], duration=args['duration'], batch=batch)
    trainsize = int(len(datas) * (90 /100) / batch) * batch
    testsize = len(datas) - trainsize
    train_dataset, test_dataset = torch.utils.data.random_split(datas, [trainsize, testsize])
    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True,  num_workers=16)
    test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False,  num_workers=16)
    return train_dataloader, test_dataloader

def main(args):
    train_dataloader, test_dataloader = init_data(args)
    model = KFusionLM(batch=args['batch'], init_lr=args['init_lr'], num_of_landmarks=68)
    torch.set_float32_matmul_precision("high")
    trainer = Trainer(max_epochs=args['max_epochs'], default_root_dir=args['save_weights'], fast_dev_run=args['is_test'], check_val_every_n_epoch=args["val_epochs"])
    trainer.fit(model, train_dataloader, test_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config/exp.json')
    args = parser.parse_args()
    args = json.loads(open(args.config, 'r').read())
    print("=="*30)
    print(args)
    print("=="*30)
    main(args)
    print("=="*30)
    print("All done!!!")
    print("=="*30)