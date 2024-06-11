import os
import json
import argparse
# W023 W021 M042


def main(root, output):
    fols = os.listdir(root)
    emotions = {'angry': 0, 'fear': 1, 'sad': 2, 'contempt': 3,
                'happy': 4, 'surprised': 5, 'disgusted': 6, 'neutral': 7}
    datalist = []
    for fol in fols:
        if fol in ['W017']: continue
        for emo in emotions.keys():
            end = 4
            if emo=="neutral":
                end = 2
            for level in range(1, end):
                audios = os.listdir(
                    f"{root}/{fol}/audio/{emo}/level_{level}/")
                videos = os.listdir(
                    f"{root}/{fol}/video/front/{emo}/level_{level}/")
                for f in audios:
                    af = f.replace(".m4a", ".mp4")
                    if af in videos:
                        row = {"audio": f"{fol}/audio/{emo}/level_{level}/{f}",
                            "video": f"{fol}/video/front/{emo}/level_{level}/{af}", 
                            "label": emotions[emo],
                            }
                        datalist.append(row)

    with open(output, "w") as f:
        json.dump(datalist, f)

def after_extract(output):
    emotions = {'angry': 0, 'fear': 1, 'sad': 2, 'contempt': 3,
                    'happy': 4, 'surprised': 5, 'disgusted': 6, 'neutral': 7}

    audios = os.listdir("dataset/audios")
    videos = os.listdir("dataset/vidcrops")
    landmarks = os.listdir("dataset/face_alignment")

    datalist = []
    for aud in audios:
        fol, file = aud.split('_',1)
        emo,_ = file.split('_',1)
        vid = f"{fol}_front_{file.replace('wav', 'mp4')}"
        lm  = f"{fol}_front_{file.replace('wav', 'npy')}"
        name = f"{fol}_{file.replace('.wav', '')}"
        if vid in videos and lm in landmarks:
            datalist.append({'video': vid, 'audio': aud, 'landmark': lm, 'emotion': emotions[emo], "name": name})

    with open(output, "w") as f:
        json.dump(datalist, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/mnt/e/MEAD")
    parser.add_argument("--output", type=str, default="./utils/datalist.json")

    args = parser.parse_args()
    main(args.root, args.output)
