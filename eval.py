#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18/01/2023 2:15 pm
# @Author  : Tianheng Qiu
# @FileName: test.py
# @Software: PyCharm

import argparse
import logging
import os
import sys
from datetime import time
from glob import glob
import cv2
import math
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from skimage import img_as_ubyte

from basicsr.data.data_util import paired_paths_from_lmdb
from basicsr.models.archs.MLWNet_arch import MLWNet_Local
from basicsr.metrics.psnr_ssim import PSNR, SSIM
import torch.nn.functional as F
import lmdb


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", "bmp", ".jpeg"])


def load_img(file_path):
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    # img = img.astype(np.uint16)
    # img = img / 255.
    return img


class dataset_val(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(dataset_val, self).__init__()
        # print(os.path.join(rgb_dir, 'blur_crops'))
        inp_files = sorted(glob(os.path.join(rgb_dir, 'input') + "/*.png"))
        tar_files = sorted(glob(os.path.join(rgb_dir, 'target') + "/*.png"))

        self.inp_filenames = [x for x in inp_files if is_image_file(x)]
        self.tar_filenames = [x for x in tar_files if is_image_file(x)]
        # self.normalize = Normalize()

        self.img_options = img_options
        self.tar_size = len(self.tar_filenames)  # get the size of target

        self.ps = None

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        index_ = index % self.tar_size
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        clean = load_img(tar_path)
        noisy = load_img(inp_path)

        # clean, noisy = self.normalize(tar_img, inp_img)

        clean = torch.from_numpy(clean.astype(np.float32) / 255.)
        noisy = torch.from_numpy(noisy.astype(np.float32) / 255.)
        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        # Validate on center crop
        if self.ps is not None:
            H = clean.shape[1]
            W = clean.shape[2]
            # r = np.random.randint(0, H - ps) if not H-ps else 0
            # c = np.random.randint(0, W - ps) if not H-ps else 0
            r = -1
            c = -1
            if H - ps <= 0:
                r = 0
            if W - ps <= 0:
                c = 0
            if r == -1:
                r = np.random.randint(0, H - ps)
            if c == -1:
                c = np.random.randint(0, W - ps)
            clean = clean[:, r:r + ps, c:c + ps]
            noisy = noisy[:, r:r + ps, c:c + ps]

        # batch = {'img256': noisy, 'img128': noisy128, 'img64': noisy64,
        #          'label256': clean, 'label128': clean128, 'label64': clean64}
        # filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        # img, label
        return noisy, clean, os.path.split(self.tar_filenames[index_])[-1]


class dataset_val_lmdb(Dataset):
    def __init__(self, folder, img_options=None, target_transform=None):
        super(dataset_val_lmdb, self).__init__()

        self.blur_folder = os.path.join(folder, 'input.lmdb')
        self.sharp_folder = os.path.join(folder, 'target.lmdb')

        # file_names
        self.names = paired_paths_from_lmdb([self.blur_folder, self.sharp_folder], ['blur', 'sharp'])

        self.generator_blur = lmdb.open(self.blur_folder, readonly=True, lock=False, readahead=False)
        self.generator_sharp = lmdb.open(self.sharp_folder, readonly=True, lock=False, readahead=False)

        self.img_options = img_options
        self.ps = None

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # get blur
        blur_path = self.names[index].get('blur_path')
        # print(blur_path)
        with self.generator_blur.begin(write=False) as txn:
            blur_bytes = txn.get(blur_path.encode("ascii"), 'blur')
            img = load_img_from_bytes(blur_bytes)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img.astype(np.float32) / 255.)

        sharp_path = self.names[index].get('sharp_path')
        with self.generator_sharp.begin(write=False) as txn:
            sharp_bytes = txn.get(sharp_path.encode("ascii"), 'sharp')
            label = load_img_from_bytes(sharp_bytes)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            label = torch.from_numpy(label.astype(np.float32) / 255.)

        img, label = img.permute(2, 0, 1), label.permute(2, 0, 1)

        if self.ps:
            H, W = img.shape[-2:]
            # r = np.random.randint(0, H - ps) if not H-ps else 0
            # c = np.random.randint(0, W - ps) if not H-ps else 0
            if H - self.ps == 0:  # equal
                r = 0
                c = 0
            else:
                r = np.random.randint(0, H - self.ps)
                c = np.random.randint(0, W - self.ps)
            img, label = img[:, r:r + self.ps, c:c + self.ps], label[:, r:r + self.ps, c:c + self.ps]

        return img, label, blur_path



def test_model(model, data, use_gpu=True, visible=True, bn=1, dataset_name='realblur-j'):
    global best_loss
    test_data = data
    running_ssim = 0.0
    running_psnr = 0.0
    # model.cuda()
    batches = tqdm(test_data)
    with torch.no_grad():
        shape_dic = {}
        h_mean = 0.
        w_mean = 0.
        for index, data in enumerate(batches):
            batches.set_description("test_process")
            img256, label256, name = data
            # if name[0] != 'scene155_11':
            #     continue
            h, w = img256.size()[-2:]
            factor = 16
            if use_gpu:
                img256 = img256.cuda()
                label256 = label256.cuda()
            # input_ = F.pad(img256, (0, 704-w, 0, 760-h), 'reflect')
            # print(input_.shape)
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(img256, (0, padw, 0, padh), 'reflect')
            # print(input_.shape)
            # label128 = F.interpolate(label256, size=(ps // 2, ps // 2))

            # img128 = F.interpolate(img256, size=(ps[0] // 2, ps[1] // 2))

            # label64 = F.interpolate(label256, size=(ps // 4, ps // 4))
            # img64 = F.interpolate(img256, size=(ps[0] // 4, ps[1] // 4))
            # print(img256.shape)
            # img256, mask = expand2regular(img256, factor=8.)
            # shape_dic[img256.shape] = shape_dic.get(img256.shape, 0) + 1
            # h_mean += img256.shape[2]
            # w_mean += img256.shape[3]
            # if shape_dic[img256.shape] == 1:
            #     print(img256.shape, name)
            # print(img256.shape)
            # continue
            # shape_dic[img256.shape]
            # print(img256.shape)
            # print(img256.shape)
            # print(input_.max(), input_.min())
            outputs = model(input_)
            if not (isinstance(outputs, list) or isinstance(outputs, tuple)):
                outputs = [outputs]
            output = outputs[0][..., :h, :w]
            #
            # output = torch.masked_select(output, mask.bool()).reshape(1, 3, label256.shape[-2], -1)
            # print(output.shape)
            psnr = PSNR(output, label256, bn, out_type=np.uint8)
            ssim = SSIM(output, label256, bn, out_type=np.uint8)
            running_psnr += psnr
            running_ssim += ssim
            batches.set_postfix(gpu_mem='%.3gG' % (
                torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0),
                                PSNR=psnr, SSIM=ssim)
            print(name[0], psnr)
            if visible:
                # out_imgs = tensor2img(torch.clamp(output, 0, 1), out_type=np.uint8, rgb2bgr=True)
                in_img = img256.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                out_img = torch.clamp(output,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                # print(name)
                # print("11111111111111111111111")
                cv2.imwrite(f'./test_dir/blur_{dataset_name}/{name[0]}',
                            cv2.cvtColor(img_as_ubyte(in_img), cv2.COLOR_RGB2BGR))
                cv2.imwrite(f'./test_dir/pred_{dataset_name}/{name[0]}',
                            cv2.cvtColor(img_as_ubyte(out_img), cv2.COLOR_RGB2BGR))
                # cv2.imwrite(f'./test_dir/pred_{dataset_name}/{name[0]}', cv2.cvtColor(img_as_ubyte(out_img), cv2.COLOR_RGB2BGR))

                # in_imgs = torch.clamp(img256,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                # enum batch
                # for index, img in enumerate(out_imgs):
                    # print(name)
                    # cv2.imwrite(f'./test_dir/blur/{name[index]}.jpg', in_imgs[index])
                    # cv2.imwrite(f'./test_dir/pred_GoPro/{name[index]}.png', img)

        # print(shape_dic)
        # print(h_mean/len(test_data))
        # print(w_mean/len(test_data))
        epoch_psnr = running_psnr / len(test_data)
        epoch_ssim = running_ssim / len(test_data)
        f.write('ori -> SSIM: {:.4f} PSNR: {:.4f}\n'.format(epoch_ssim, epoch_psnr))
        tqdm.write('SSIM: {:.4f} PSNR: {:.4f}'.format(epoch_ssim, epoch_psnr))
        batches.close()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def check_and_create_dir(path):
    p1, file = os.path.split(path)
    dir_list = p1.split('/')
    rot = '.'
    for p in dir_list:
        rot += '/' + p
        if not os.path.exists(rot):
            os.mkdir(rot)


def load_checkpoint(model, checkpoint_PATH, device, param_key='params'):
    CKPT = torch.load(checkpoint_PATH, map_location=lambda storage, loc: storage)
    model_CKPT = CKPT['params']
    # # model_CKPT = model_CKPT["state_dict"]
    # for k, v in model_CKPT.copy().items():
    #     print(k)
    #     if 'module' in k:
    #         model_CKPT[k.replace('module.', '')] = model_CKPT[k]
    #         del model_CKPT[k]
    #     if 'backbone' in k:
    #         model_CKPT[k.replace('backbone', 'encoder')] = model_CKPT[k]
    #         del model_CKPT[k]
    #     if 'neck' in k:
    #         model_CKPT[k.replace('neck', 'fusion')] = model_CKPT[k]
    #         del model_CKPT[k]
    #     if 'head' in k:
    #         model_CKPT[k.replace('head.', 'decoder.')] = model_CKPT[k]
    #         del model_CKPT[k]
    # CKPT['params'] = model_CKPT
    # torch.save(CKPT, '55000_new.pth')

    model.load_state_dict(model_CKPT, strict=True)
    print(f'loading checkpoint from {checkpoint_PATH}!')

    return model


def load_test(dir_path, batch_size=1):
    val_set = dataset_val(dir_path, None)
    # val_set = dataset_val_lmdb(dir_path, None)
    print("eval set size: ", len(val_set))
    test_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size, num_workers=6)
    return test_loader


def load_img_from_bytes(content, flag='color'):
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    return img


def pre_val(opt):
    global best_loss
    BATCH_SIZE = opt.batch_size
    torch.cuda.empty_cache()
    dir_path = opt.dir
    test_data = load_test(dir_path, batch_size=BATCH_SIZE)
    best_loss = 1E6
    base_size = int(256 * 1.5)
    model = MLWNet_Local(dim=64, base_size=base_size)  # realblur-J1.9最佳
    # model = MPRNet()
    f.write(f'{base_size}')
    # model = fftformer(dim=48, num_blocks=[6, 6, 12], num_refinement_blocks=4, ffn_expansion_factor=3, bias=False).eval().cuda()
    # model = NAFNet(img_channel=3, width=64, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]).eval().cuda()
    # model = NAFSSR(up_scale=2, num_blks=128, width=128)
    # model = DAT(upscale=4, in_chans=3, img_size=64, img_range=1., depth=[18], embed_dim=60,
    #             num_heads=[6], expansion_factor=2, resi_connection='3conv', split_size=[8, 32], upsampler='pixelshuffledirect').eval()
    use_gpu = False
    if opt.device == 'GPU':
        use_gpu = True
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)
    # print("Model: ", str(model))
    model = load_checkpoint(model, opt.weights, device)
    # traced = torch.jit.trace(model, (torch.rand(1, 3, 64, 64).cuda()))
    # torch.jit.save(traced, './DAT_light_x4.pt')
    # exit()
    # exit()
    f.write(f'{opt.weights}\n')
    test_model(model=model,
               data=test_data,
               use_gpu=use_gpu,
               bn=BATCH_SIZE,
               visible=True,
               dataset_name='RealBlur_J')

if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
    parse = argparse.ArgumentParser()
    # /mnt/q/deblur_ablation_exp/MAWNet-512p1-withloss-RealBlur_J-width32-4gpus/models/net_g_102000.pth
    # /mnt/q/deblur_ablation_exp/V100_01/experiments/MAWNet-512p3-GoPro-width64-8batch/models/net_g_7000.pth
    parse.add_argument('--weights', type=str, default="realblur_j.pth",
                       help='pretrained weights')
    parse.add_argument('--cfg', type=str, default="fptd")
    # parse.add_argument('--dir', type=str,
    #                    default='/mnt/d/NAFNet-main/datasets/GoPro/test/')
    parse.add_argument('--dir', type=str,
                       default='/mnt/d/realblur/RealBlur_J/test/')
    # parse.add_argument('--dir', type=str,
    #                    default='/mnt/q/extra_dataset/RSBLUR_DATASET/RSBlur-simple/test/')
    # parse.add_argument('--dir', type=str,
    #                    default='./noise_estimate/L5')
    # parse.add_argument('--dir', type=str,
    #                    default='/mnt/d/HIDE/test/')
    # parse.add_argument('--dir', type=str, default='./test_dir/')
    # parse.add_argument('--dir', type=str, default='./datasets/GoPro/test/')
    parse.add_argument('--batch-size', type=int, default=1)
    parse.add_argument('--device', type=str, default='GPU')
    opt = parse.parse_args()
    f = open('simple.log', 'a')
    pre_val(opt)
    f.close()