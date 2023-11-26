import math

import torch
import numpy as np
import pickle
import cv2
import torch.nn.functional as F
from gevent import os
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.autograd import Variable
import albumentations as albu
import random
from torchvision.utils import make_grid


def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])


def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])


def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])


def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict


def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)


def load_npy(filepath):
    img = np.load(filepath)
    return img


def load_img(filepath):
    img = cv2.imread(filepath)
    # img = img.astype(np.float32)
    # img = img / 255.
    return img


# def PSNR1(tar_img, prd_img):
#     imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
#     rmse = (imdff ** 2).mean().sqrt()
#     ps = 20 * torch.log10(1 / rmse)
#     return ps
#
#
# def PSNR2(tar_img, prd_img):
#     mse = np.mean((tar_img / 255. - prd_img / 255.) ** 2)
#     if mse == 0:
#         return 100
#     PIXEL_MAX = 1
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
#
#
# def batch_PSNR(img1, img2, average=True):
#     PSNR = []
#     for im1, im2 in zip(img1, img2):
#         psnr = PSNR1(im1, im2)
#         PSNR.append(psnr)
#     return sum(PSNR) / len(PSNR) if average else sum(PSNR)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


# def SSIM(img1, img2):
#     (_, channel, _, _) = img1.size()
#     window_size = 11
#     window = create_window(window_size, channel)
#
#     if img1.is_cuda:
#         window = window.cuda(img1.get_device())
#     window = window.type_as(img1)
#
#     mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
#
#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2
#
#     sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
#
#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2
#
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()


# def PSNR(output, target, max_val=1.0):
#     # output = output.clamp(0.0, 1.0)
#     # target = output.clamp()
#     mse = torch.pow(target - output, 2).mean()
#     if mse == 0:
#         return torch.Tensor([100.0])
#     return 10 * torch.log10(max_val ** 2 / mse)

def PSNR(output, target, n_batch, out_type=np.float64):
    output_arr = tensor2img(output, out_type=out_type)
    target_arr = tensor2img(target, out_type=out_type)
    psnr_sum = 0
    for i in range(len(output_arr)):
        psnr_sum += calculate_psnr(output_arr[i], target_arr[i])
    return psnr_sum / n_batch


def SSIM(output, target, n_batch, out_type=np.float64):
    output_arr = tensor2img(output, out_type=out_type)
    target_arr = tensor2img(target, out_type=out_type)
    # print(output_arr.shape)
    ssim_sum = 0
    for i in range(len(output_arr)):
        ssim_sum += calculate_ssim(output_arr[i], target_arr[i])
    return ssim_sum / n_batch


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


class Normalize:  # use mean=0.5 and std=0.5 to normalize the image to [-1, 1]
    def __init__(self, mean=0.5, std=0.5):
        self.normalize = albu.Normalize(mean=mean, std=std)
        self.normalize = albu.Compose([self.normalize], additional_targets={'target': 'image'})

    def __call__(self, a, b):
        res = self.normalize(image=a, target=b)
        return res['image'], res['target']


def img2tensor(imgs, bgr2rgb=False, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


# np.float32, [0, 1], (h, w, c)
def tensor2img(tensor: torch.tensor, rgb2bgr=False, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    # result = []
    for _tensor in tensor:
        _tensor = _tensor.float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:  # B, C, H, W
            bs_res = []
            for st in _tensor:
                # (C, H, W) -> (H, W, C)
                img_np = st.numpy().transpose(1, 2, 0)
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                bs_res.append(img_np)
            img_np = np.array(bs_res)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        # result.append(img_np)
    # if len(result) == 1:
    #     result = result[0]
    return img_np


def tensor2img_fast(tensor, rgb2bgr=True, min_max=(0, 1)):
    """This implementation is slightly faster than tensor2img.
    It now only supports torch tensor with shape (1, c, h, w).

    Args:
        tensor (Tensor): Now only support torch tensor with (1, c, h, w).
        rgb2bgr (bool): Whether to change rgb to bgr. Default: True.
        min_max (tuple[int]): min and max values for clamp.
    """
    output = tensor.squeeze(0).detach().clamp_(*min_max).permute(1, 2, 0)
    output = (output - min_max[0]) / (min_max[1] - min_max[0]) * 255
    output = output.type(torch.uint8).cpu().numpy()
    if rgb2bgr:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output


def crop_border(imgs, crop_border):
    """Crop borders of images.

    Args:
        imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
        crop_border (int): Crop border for each end of height and weight.

    Returns:
        list[ndarray]: Cropped images.
    """
    if crop_border == 0:
        return imgs
    else:
        if isinstance(imgs, list):
            return [v[crop_border:-crop_border, crop_border:-crop_border, ...] for v in imgs]
        else:
            return imgs[crop_border:-crop_border, crop_border:-crop_border, ...]


def batch_bgr2gray(images: torch.tensor):
    assert images.shape[1] == 3, "input numpy array is not a rgb pic"
    out_imgs = []
    for img in images:  # c, h, w
        b, g, r = img.chunk(3, dim=0)
        out_imgs.append(0.299*r + 0.587*g + 0.114*b)  #
    return torch.stack(out_imgs)


def _ssim_cly(img1, img2):
    assert len(img1.shape) == 2 and len(img2.shape) == 2
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    # print(kernel)
    window = np.outer(kernel, kernel.transpose())

    bt = cv2.BORDER_REPLICATE

    mu1 = cv2.filter2D(img1, -1, window, borderType=bt)
    mu2 = cv2.filter2D(img2, -1, window, borderType=bt)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window, borderType=bt) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window, borderType=bt) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=bt) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1,
                   img2,
                   crop_border=0,
                   input_order='HWC',
                   test_y_channel=False,
                   ssim3d=True):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')

    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)

    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    def _cal_ssim(img1, img2):
        if test_y_channel:
            img1 = to_y_channel(img1)
            img2 = to_y_channel(img2)
            return _ssim_cly(img1[..., 0], img2[..., 0])

        ssims = []
        # ssims_before = []

        # skimage_before = skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True)
        # print('.._skimage',
        #       skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True))
        max_value = 1 if img1.max() <= 1 else 255
        with torch.no_grad():
            final_ssim = _ssim_3d(img1, img2, max_value) if ssim3d else _ssim(img1, img2, max_value)
            ssims.append(final_ssim)

        # for i in range(img1.shape[2]):
        #     ssims_before.append(_ssim(img1, img2))

        # print('..ssim mean , new {:.4f}  and before {:.4f} .... skimage before {:.4f}'.format(np.array(ssims).mean(), np.array(ssims_before).mean(), skimage_before))
        # ssims.append(skimage.metrics.structural_similarity(img1[..., i], img2[..., i], multichannel=False))

        return np.array(ssims).mean()

    if img1.ndim == 3 and img1.shape[2] == 6:
        l1, r1 = img1[:, :, :3], img1[:, :, 3:]
        l2, r2 = img2[:, :, :3], img2[:, :, 3:]
        return (_cal_ssim(l1, l2) + _cal_ssim(r1, r2)) / 2
    else:
        return _cal_ssim(img1, img2)


def calculate_psnr(img1,
                   img2,
                   crop_border=0,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        img2 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)

    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    def _psnr(img1, img2):
        if test_y_channel:
            img1 = to_y_channel(img1)
            img2 = to_y_channel(img2)

        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        max_value = 1. if img1.max() <= 1 else 255.

        return 20. * np.log10(max_value / np.sqrt(mse))

    if img1.ndim == 3 and img1.shape[2] == 6:
        l1, r1 = img1[:, :, :3], img1[:, :, 3:]
        l2, r2 = img2[:, :, :3], img2[:, :, 3:]
        return (_psnr(l1, l2) + _psnr(r1, r2)) / 2
    else:
        # print(img1.shape, img1.dtype)
        return _psnr(img1, img2)


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def _ssim_3d(img1, img2, max_value):
    assert len(img1.shape) == 3 and len(img2.shape) == 3
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.

    Returns:
        float: ssim result.
    """
    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = _generate_3d_gaussian_kernel().cuda()

    img1 = torch.tensor(img1).float().cuda()
    img2 = torch.tensor(img2).float().cuda()

    mu1 = _3d_gaussian_calculator(img1, kernel)
    mu2 = _3d_gaussian_calculator(img2, kernel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _3d_gaussian_calculator(img1 ** 2, kernel) - mu1_sq
    sigma2_sq = _3d_gaussian_calculator(img2 ** 2, kernel) - mu2_sq
    sigma12 = _3d_gaussian_calculator(img1 * img2, kernel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def _ssim(img1, img2, max_value):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def _3d_gaussian_calculator(img, conv3d):
    out = conv3d(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    return out


def _generate_3d_gaussian_kernel():
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    kernel_3 = cv2.getGaussianKernel(11, 1.5)
    kernel = torch.tensor(np.stack([window * k for k in kernel_3], axis=0))
    conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
    conv3d.weight.requires_grad = False
    conv3d.weight[0, 0, :, :, :] = kernel
    return conv3d


def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    conversion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError(f'The img type should be np.float32 or np.uint8, but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace conversion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(f'The dst_type should be np.float32 or np.uint8, but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)
