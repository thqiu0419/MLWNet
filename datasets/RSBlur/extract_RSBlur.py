import os
import shutil

from joblib import Parallel, delayed
from tqdm import tqdm

def extract_set(root_path, ent):
    set_path = os.path.join(root_path, ent)
    with open(f'./{ent}_syn.txt') as f:
        file_list = f.readlines()
        if not os.path.exists(set_path):
            os.mkdir(set_path)
        blur_path = os.path.join(set_path, 'input')
        gt_path = os.path.join(set_path, 'target')
        if not os.path.exists(blur_path):
            os.mkdir(blur_path)
        if not os.path.exists(gt_path):
            os.mkdir(gt_path)
        Parallel(n_jobs=20, backend='multiprocessing')(
            delayed(copy_file)(line, root_path, gt_path, blur_path) for line in tqdm(file_list))

        print(f"{ent} set extracted done!")

def copy_file(line, root_path, gt_path, blur_path):
    gt, blur = line.split()
    # # copy file
    # shutil.copy2(os.path.join(root_path, gt), gt_path)
    # shutil.copy2(os.path.join(root_path, blur), blur_path)
    # rename
    n_name = f"RSBlur_{gt.split('/')[1]}_{gt.split('/')[2]}.png"
    # shutil.move(os.path.join(root_path, gt), os.path.join(gt_path, n_name))
    shutil.move(os.path.join(root_path, blur), os.path.join(blur_path, n_name))



if __name__ =='__main__':
    extract_set('/mnt/q/extra_dataset/RSBLUR_DATASET', 'train')
    # extract_set('/mnt/q/extra_dataset/RSBLUR_DATASET', 'val')
    extract_set('/mnt/q/extra_dataset/RSBLUR_DATASET', 'test')