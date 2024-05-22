import os
import json
from tqdm import tqdm
import argparse


def main(input_dir:str, output_dir:str, datajson):

    files = json.load(open(f"{datajson}", "r"))
    exits = os.listdir("audios")
    for row in tqdm(files):
        out_path = f"{row['audio']}".replace("/", "_").replace("audio_", "")
        if out_path.replace('.m4a', '.wav') not in exits:
            try:
                os.system(f"cp -r {input_dir}/{row['audio']} {output_dir}/{out_path}")
                os.system(f"ffmpeg -i {output_dir}/{out_path} {output_dir}/{out_path.replace('.m4a', '.wav')} -loglevel panic -y")
                os.system(f"rm -rf {output_dir}/{out_path}")
            except:
                print(f"{row['audio']}")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input_path", type=str, default="/home/sowwn/SSD/mah_space/dataset/MEAD")
    args.add_argument("--datajson", type=str, default="ultis/datalist.json")
    args.add_argument("--output_path", type=str, default="audios")

    args = args.parse_args()
    main(args.input_path, args.output_path, args.datajson)

