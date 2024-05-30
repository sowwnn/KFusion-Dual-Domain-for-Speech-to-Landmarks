import os
from tqdm import tqdm

def main():
    load_dir = '/home/sowwn/gg_space/mah_ws/dataset/MEAD'
    list_fol = os.listdir(f"{load_dir}")
    ard = ['W021']

    for item in list_fol:
        if item in ard:
            files = os.listdir(f"{load_dir}/{item}")
            save_dir = f"/mnt/e/MEAD/{item}"
            os.system(f"mkdir {save_dir}")
            print("=="*30)
            print(save_dir)

            for file in files:
                os.system(f"tar -xvf {load_dir}/{item}/{file} -C {save_dir}")
def w021():
    load_dir = '/home/sowwn/gg_space/mah_ws/dataset/MEAD'
    files = os.listdir(f"{load_dir}/W021")
    save_dir = f"/mnt/e/MEAD/W021"
    os.system(f"mkdir {save_dir}")
    print("=="*30)
    print(save_dir)

    os.system(f"tar -xvf {load_dir}/W021/audio.tar -C {save_dir}")
    os.system(f"tar -xvf {load_dir}/W021/video/1.tar -C {save_dir}")
    os.system(f"tar -xvf {load_dir}/W021/video/2.tar -C {save_dir}")

    

if __name__ == '__main__':
    # main()
    w021()

