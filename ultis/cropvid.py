import sys
sys.settrace
import os
import torch
from tqdm import tqdm
import torchvision.io as io
from torchvision.transforms import v2
import warnings
warnings.filterwarnings("ignore")
import json
import argparse


def main(input_dir:str, output_dir:str, datajson):
    transforms = v2.Compose([v2.CenterCrop((1080,1080)), v2.Resize((256,256))])

    files = json.load(open(f"{datajson}", "r"))
    exits = os.listdir("data_crop")
    for row in tqdm(files):
        out_path = f"{row['video']}".replace("/", "_").replace("video_", "")
        if out_path not in exits:
            video = io.read_video(f"{input_dir}/{row['video']}", pts_unit="sec",output_format="TCHW")[0]
            new_clip = torch.zeros((video.shape[0], video.shape[1], 256, 256),dtype=torch.uint8)
            for idx, frame in enumerate(video):
                new_clip[idx] = transforms(frame)

            io.write_video(f"{output_dir}/{out_path}", new_clip.permute(0,2,3,1), fps=30)
            row['video'] = out_path

    with open("data_crop/datalist.json", "w") as f:
        json.dump(files, f)
    

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input_path", type=str, default="/home/sowwn/SSD/mah_space/dataset/MEAD")
    args.add_argument("--datajson", type=str, default="ultis/datalist.json")
    args.add_argument("--output_path", type=str, default="data_crop")

    # input_path = "/home/sowwn/SSD/mah_space/dataset/MEAD"
    # output_path = "data_crop"
    args = args.parse_args()
    main(args.input_path, args.output_path, args.datajson)
