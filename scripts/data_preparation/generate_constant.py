import shutil
from os import path as osp
import os
import sys
from tqdm import tqdm


if __name__ == '__main__':
    folder_path = '/mnt/d/realblur/'
    j_train_txt = 'RealBlur_J_train_list.txt'
    j_test_txt = 'RealBlur_J_test_list.txt'
    r_train_txt = 'RealBlur_R_train_list.txt'
    r_test_txt = 'RealBlur_R_test_list.txt'
    j_output_path = '/mnt/d/realblur/RealBlur_J/'
    r_output_path = '/mnt/d/realblur/RealBlur_R/'
    if not osp.exists(j_output_path):
        info = []
        os.makedirs(j_output_path)
        if not osp.exists(j_output_path+'train'):
            os.makedirs(j_output_path+'train')
            info.append(['train'])
        if not osp.exists(j_output_path+'train/'+'target'):
            os.makedirs(j_output_path+'train/'+'target')
            info[0].append('{target}')
        if not osp.exists(j_output_path+'train/'+'input'):
            os.makedirs(j_output_path+'train/'+'input')
            info[0].append('{input}')

        if not osp.exists(j_output_path+'test'):
            os.makedirs(j_output_path+'test')
            info.append(['test'])
        if not osp.exists(j_output_path+'test/'+'target'):
            os.makedirs(j_output_path+'test/'+'target')
            info[1].append('{target}')
        if not osp.exists(j_output_path+'test/'+'input'):
            os.makedirs(j_output_path+'test/'+'input')
            info[1].append('{input}')

        print(f'mkdir {j_output_path} {{{info}}}...')
    else:
        print(f'Folder {j_output_path} already exists. Exit.')
        sys.exit(1)
    if not osp.exists(r_output_path):
        info = []
        os.makedirs(r_output_path)

        if not osp.exists(r_output_path + 'train'):
            os.makedirs(r_output_path + 'train')
            info.append(['train'])
        if not osp.exists(r_output_path + 'train/' + 'target'):
            os.makedirs(r_output_path + 'train/' + 'target')
            info[0].append('{target}')
        if not osp.exists(r_output_path + 'train/' + 'input'):
            os.makedirs(r_output_path + 'train/' + 'input')
            info[0].append('{input}')

        if not osp.exists(r_output_path + 'test'):
            os.makedirs(r_output_path + 'test')
            info.append(['test'])
        if not osp.exists(r_output_path + 'test/' + 'target'):
            os.makedirs(r_output_path + 'test/' + 'target')
            info[1].append('{target}')
        if not osp.exists(r_output_path + 'test/' + 'input'):
            os.makedirs(r_output_path + 'test/' + 'input')
            info[1].append('{input}')
        print(f'mkdir {r_output_path} {{{info}}}...')
    else:
        print(f'Folder {r_output_path} already exists. Exit.')
        sys.exit(1)
    # process with RealBlur-J
    with open(folder_path + j_train_txt) as f:
        for gt_blur in tqdm(f.readlines()):
            gt, blur = gt_blur.split()
            # gt
            gt_info = gt.split('/')
            shutil.copyfile(folder_path+gt,
                            j_output_path+f'train/target/{gt_info[1]}_{gt_info[-1].replace("gt_", "").zfill(4)}')
            blur_info = blur.split('/')
            shutil.copyfile(folder_path+blur,
                            j_output_path+f'train/input/{blur_info[1]}_{blur_info[-1].replace("blur_", "").zfill(4)}')
    with open(folder_path + j_test_txt) as f:
        for gt_blur in tqdm(f.readlines()):
            gt, blur = gt_blur.split()
            # gt
            gt_info = gt.split('/')
            shutil.copyfile(folder_path+gt,
                            j_output_path+f'test/target/{gt_info[1]}_{gt_info[-1].replace("gt_", "").zfill(4)}')
            blur_info = blur.split('/')
            shutil.copyfile(folder_path+blur,
                            j_output_path+f'test/input/{blur_info[1]}_{blur_info[-1].replace("blur_", "").zfill(4)}')

    # process with RealBlur-R
    with open(folder_path + r_train_txt) as f:
        for gt_blur in tqdm(f.readlines()):
            gt, blur = gt_blur.split()
            # gt
            gt_info = gt.split('/')
            shutil.copyfile(folder_path+gt,
                            r_output_path+f'train/target/{gt_info[1]}_{gt_info[-1].replace("gt_", "").zfill(4)}')
            # blur
            blur_info = blur.split('/')
            shutil.copyfile(folder_path+blur,
                            r_output_path+f'train/input/{blur_info[1]}_{blur_info[-1].replace("blur_", "").zfill(4)}')
    with open(folder_path + r_test_txt) as f:
        for gt_blur in tqdm(f.readlines()):
            gt, blur = gt_blur.split()
            # blur
            gt_info = gt.split('/')
            shutil.copyfile(folder_path+gt,
                            r_output_path+f'test/target/{gt_info[1]}_{gt_info[-1].replace("gt_", "").zfill(4)}')
            blur_info = blur.split('/')
            # blur
            shutil.copyfile(folder_path+blur,
                            r_output_path+f'test/input/{blur_info[1]}_{blur_info[-1].replace("blur_", "").zfill(4)}')