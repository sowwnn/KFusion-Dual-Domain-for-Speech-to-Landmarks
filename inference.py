from dataloader.dataloader import MEAD
from module.KFusionLM import KFusionLM
from torch.utils.data import DataLoader
import torch
import argparse
import json
import os
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import warnings
import subprocess
warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision("high")

def load_model(args):
    model = KFusionLM.load_from_checkpoint(args['weights'], batch=args['batch'], init_lr=1e-3, num_of_landmarks=68)
    model.to(args['device'])
    model.eval()

    return model

def load_data(args):
    batch = args['batch']
    datas = MEAD(args['datalist'], duration=args['duration'], batch=args['batch'], audio_path=args['audio_path'], landmark_path=args['landmark_path'])
    print(f"Len of Inference data: {len(datas)}")
    dataloader = DataLoader(datas, batch_size=batch, shuffle=False,  num_workers=args["num_workers"])
    return dataloader, datas

def plot_figure(y, y_, video, name):
    os.makedirs(f"{args['out_dir']}/figure", exist_ok=True)
    image_folder = f"{args['out_dir']}/figure/{name}"
    y = y.detach().cpu().numpy()
    y_ = y_.detach().cpu().numpy()
    for t in range(len(y[0])):
        os.makedirs(f"{image_folder}", exist_ok=True)
        plain = np.ones((256,256,3))
        plt.subplot(1,3,1)
        plt.imshow(video[t])
        plain = np.ones((256,256,3))
        plt.subplot(1,3,2)
        plt.imshow(plain)
        plt.scatter(y[0][t][0,:], y[0][t][1,:], c='b', s=0.5)
        plt.title("GT")
        plt.subplot(1,3,3)
        plt.imshow(plain)
        plt.scatter(y_[0][t][0,:], y_[0][t][1,:], c='r', s=0.5)
        plt.title("Predit")
        plt.savefig(f"{image_folder}/{t}.png")
        plt.clf()

        video_name = f'{image_folder}/{name}.mp4'

    try:
        command = ['ffmpeg', '-i', f'{image_folder}/%d.png',  '-r', '30', video_name, '-y']
        subprocess.run(command)
    except:
        print("Failed to convert images to video. May you need to install ffmpeg!")


def infer(args, model, dataloader, datas):
    os.makedirs(f"{args['out_dir']}", exist_ok=True)
    os.makedirs(f"{args['out_dir']}/prediction", exist_ok=True)
    for batch in dataloader:
        x = batch['audio'].to(args['device'])
        v = batch['ilm'].to(args['device'])
        w = batch['label'].to(args['device'])
        y = batch['target'].to(args['device'])
        # print(batch)
        name = batch['name'][0].split("_",1)
        video = torchvision.io.read_video(f"dataset/duration/vidcrops/{name[0]}_front_{name[1]}.mp4", pts_unit='sec')[0]

        y_ = model.forward(x,v,w)
        np.save(f"{args['out_dir']}/prediction/{'_'.join(name)}.npy", y_.cpu().detach().numpy())
        y = y * (datas.mx - datas.mn) + datas.mn
        y_ = y_ * (datas.mx - datas.mn) + datas.mn
        plot_figure(y, y_, video, "_".join(name))


def main(args):
    dataloader, datas = load_data(args)
    model = load_model(args)
    infer(args, model, dataloader, datas)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config/infer.json')
    args = parser.parse_args()
    args = json.loads(open(args.config, 'r').read())
    print("=="*30)
    print(args)
    print("=="*30)
    main(args)
    print("=="*30)
    print("All done!!!")
    print("=="*30)

