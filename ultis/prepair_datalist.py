import os
import json
import argparse
# W023 W021 M042

def main(root, output, level):
    fols = os.listdir(root)
    emotions = {'angry':0,'fear':1,'sad':2, 'contempt':3,'happy':4,'surprised':5, 'disgusted':6, 'neutral':7}
    datalist = []
    for fol in fols:
        try:
            for emo in emotions.keys():
                    audios = os.listdir(f"{root}/{fol}/audio/{emo}/level_1/")
                    for f in audios:
                            af = f.replace(".m4a", ".mp4")
                            row = {"audio": f"{fol}/audio/{emo}/level_1/{f}", "video": f"{fol}/video/front/{emo}/level_1/{af}", "label": emotions[emo]}
                            datalist.append(row)
        except:
            print(f"{fol}")

    with open(output, "w") as f:
        json.dump(datalist, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/home/sowwn/SSD/mah_space/dataset/MEAD")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--output", type=str, default="ultis/datalist.json")

    args = parser.parse_args()
    main(args.root, args.output, args.level)
