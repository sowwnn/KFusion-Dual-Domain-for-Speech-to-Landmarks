import json 
import torchvision.io as io
from tqdm import tqdm
import numpy as np
import face_alignment
import mediapipe as mp    
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import json
import argparse
import os


def fa_extract(files, exist, input_dir, output_dir):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda')

    for row in tqdm(files):
        out_path = f"{row['video']}".replace("/", "_").replace("video_", "")

        out_name = f"{out_path.replace('.mp4','')}.npy"
        if out_name in exist:
            continue

        video = io.read_video(f"{input_dir}/{out_path}", pts_unit="sec",output_format="TCHW")[0]
        new = np.zeros((video.shape[0],68,2))
        # print(f"New {new.size}")
        # print(video.shape)
        for idx, frame in enumerate(video):
            for i in range(5):
                try:
                    results = fa.get_landmarks(frame.permute(1, 2, 0).numpy())
                    new[idx] = results[0]
                    break
                except: pass
        np.save(f"{output_dir}/{out_name}", new)

def landmarks_to_numpy(results):
  """Converts MediaPipe landmarks to a NumPy matrix.

  Args:
    landmarks: A list of MediaPipe landmark objects.

  Returns:
    A NumPy matrix with dimensions (num_landmarks, dimensions_per_landmark).
  """

  # Determine number of landmarks and dimensions per landmark (usually 3 for x, y, z)
  landmarks = results.face_landmarks[0]

  num_landmarks = len(landmarks)
  dimensions = 2  # Assuming x, y, and potentially z coordinates

  landmark_matrix = np.empty((num_landmarks, dimensions), dtype=np.float32)

  for i, landmark in enumerate(landmarks):
      landmark_matrix[i, 0] = landmark.x
      landmark_matrix[i, 1] = landmark.y
  
  return landmark_matrix

def media_pipe_landmarks(files, exist, input_dir, output_dir):

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="ultis/FaceLandmarker.task"),
        running_mode=VisionRunningMode.IMAGE)

    with FaceLandmarker.create_from_options(options) as landmarker:

        for row in tqdm(files):
            out_path = f"{row['video']}".replace("/", "_").replace("video_", "")
            out_name = f"{out_path.replace('.mp4','')}.npy"
            if out_name in exist:
                continue
            try: 
                video = io.read_video(f"{input_dir}/{out_path}", pts_unit="sec",output_format="TCHW")[0]
            except:
                print(f"{input_dir}/{out_path} is not exist")
                continue
            new = np.zeros((video.shape[0],478,2))
            for idx, frame in enumerate(video):
                for i in range(5):
                    try:
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(frame.permute(1,2,0)))
                        landmark = landmarker.detect(mp_image)
                        results = landmarks_to_numpy(landmark)
                        new[idx] = results
                        break
                    except: pass
                
            np.save(f"{output_dir}/{out_name}", new)

def main(input_dir = "data_crop", datalist = "ultis/datalist.json", output_dir = "temp", mode="mediapipe"):
    files = json.load(open(datalist, "r"))
    exist = os.listdir(f"{output_dir}")
    if mode=="face_alignment":
        print("FACE ALIGNMENT EXTRACT")
        fa_extract(files, exist, input_dir, output_dir)
    else:
        print("MEDIAPIPE EXTRACT")
        media_pipe_landmarks(files, exist, input_dir, output_dir)


    


def check(args):
    file = "M029_front_angry_level_1_007"
    video = io.read_video(f"{args.input_path}/{file}.mp4", pts_unit="sec",output_format="TCHW")[0]
    mk = np.load(f"{args.output_path}/{file}.npy")

    print(f"Video: {video.shape}")
    print(f"Landmark: {np.shape(mk)}")

    print(mk)

    os.system('mkdir temp')
    plt.imshow(video[0].permute(1, 2, 0).numpy())
    plt.scatter(mk[0][:,0]*256, mk[0][:,1]*256,  c='r',s=0.5)
    plt.savefig(f"temp/2/{file}.png")



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input_path", type=str, default="dataset/data_crop")
    args.add_argument("--datajson", type=str, default="dataset/data_crop/datalist.json")
    args.add_argument("--output_path", type=str, default="dataset/mp_landmark")
    args.add_argument("--mode", type=str, default="mediapipe")


    args = args.parse_args()
    main(args.input_path, args.datajson, args.output_path, args.mode)
    # check(args)
