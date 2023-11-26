import torch
import torch.nn as nn

import numbers

from basicsr.models.archs.arch_util import LayerNorm2d
from einops import rearrange
import torch.nn.functional as F

from basicsr.models.archs.local_arch import Local_Base
from basicsr.models.archs.wavelet_block import LWN


# from models.ours.wavelet_block import ResBlock_dwt




class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class WaveletBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.wavelet_block1 = LWN(c, wavelet='haar', initialize=True)
        self.conv1 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        ffn_channel = FFN_Expand * c
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.wavelet_block1(x)

        x = x * self.sca(x)

        x = self.conv1(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv2(x)
        # gate
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2
        x = self.conv3(x)
        x = self.dropout2(x)

        return y + x * self.gamma

    def get_wavelet_loss(self):
        return self.wavelet_block1.get_wavelet_loss()
            # + self.wavelet_block2.get_wavelet_loss()


# SEB
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)

        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

    def get_wavelet_loss(self):
        return 0.

class Encoder(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 dim=32,
                 num_blocks=[2, 4, 4, 6],
                 ):
        super(Encoder, self).__init__()
        self.num_blocks = num_blocks
        self.feature_embed = nn.Conv2d(in_channels=inp_channels, out_channels=dim, kernel_size=3, padding=1, stride=1,
                                       groups=1, bias=True)
        self.b1 = nn.Sequential(*[NAFBlock(dim) for _ in range(num_blocks[0])])
        self.down1 = nn.Conv2d(dim, 2 * dim, 2, 2)
        self.b2 = nn.Sequential(*[NAFBlock(dim * 2) for _ in range(num_blocks[1])])
        self.down2 = nn.Conv2d(dim * 2, dim * 2 ** 2, 2, 2)
        self.b3 = nn.Sequential(*[NAFBlock(dim * 2 ** 2) for _ in range(num_blocks[2])])
        self.down3 = nn.Conv2d(dim * 2 ** 2, dim * 2 ** 3, 2, 2)
        self.b4 = nn.Sequential(*[NAFBlock(dim * 2 ** 3) for _ in range(num_blocks[3])])

    def forward(self, x):
        x = self.feature_embed(x)  # (1, 32, 256, 256)
        x1 = self.b1(x)  # (1, 32, 256, 256)

        x = self.down1(x1)  # (1, 64, 128, 128)
        x2 = self.b2(x)  # (1, 64, 128, 128)

        x = self.down2(x2)  # (1, 128, 64, 64)
        x3 = self.b3(x)  # (1, 128, 64, 64)

        x = self.down3(x3)
        x4 = self.b4(x)

        return x4, x3, x2, x1



class Fusion(nn.Module):
    def __init__(self,
                 dim=32,
                 num_blocks=[2, 4, 4, 6],
                 ):
        super(Fusion, self).__init__()
        self.num_blocks = num_blocks
        self.up43 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 3, dim * 2 ** 4, 1, bias=False),
            nn.PixelShuffle(2)
        )
        # self.d3_conv = nn.Sequential(nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), 1),
        #                              # nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2), 1),
        #                              # nn.BatchNorm2d(dim * 2 ** 2),
        #                              # nn.GELU()
        #                              )
        self.d4 = nn.Sequential()
        self.d3 = nn.Sequential(*[WaveletBlock(dim * 2 ** 2) for _ in range(num_blocks[2])])

        self.up32 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 2, dim * 2 ** 3, 1, bias=False),
            nn.PixelShuffle(2)
        )
        # self.d2_conv = nn.Sequential(nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), 1),
        #                              # nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2), 1),
        #                              # nn.BatchNorm2d(dim * 2 ** 1),
        #                              # nn.GELU()
        #                              )
        self.d2 = nn.Sequential(*[WaveletBlock(dim * 2) for _ in range(num_blocks[1])])
        self.d1 = nn.Sequential()

    def forward(self, x4, x3, x2, x1):
        x3_b = x3.contiguous()
        x = self.up43(x4) + x3
        x3 = self.d3(x)
        # deblur head x3(min) 128
        x2_b = x2.contiguous()
        x = self.up32(x3) + x2
        x2 = self.d2(x)

        return x4, x3, x3_b, x2, x2_b, x1

    def get_wavelet_loss(self):
        wavelet_loss = 0.
        for index, _ in enumerate(self.num_blocks):
            for block in getattr(self, f'd{index+1}'):
                wavelet_loss += block.get_wavelet_loss()
        return wavelet_loss

class Deblur_head(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        self.block = nn.Sequential(
            # nn.Conv2d(num_in, num_mid, kernel_size=1),
            # nn.BatchNorm2d(num_mid),
            # nn.GELU(),
            nn.Conv2d(num_in, num_out, kernel_size=3, stride=1, padding=1),


        )

    def forward(self, x):
        x = self.block(x)
        # print(x.shape)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 dim=64,
                 out_channels=3,
                 num_blocks=[2, 4, 4, 6],
                 ):
        super().__init__()
        self.num_blocks = num_blocks
        self.head4 = Deblur_head(int(dim * 2 ** 3), int(dim * 3), 3)
        self.up43 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 3, dim * 2 ** 4, 1, bias=False),
            nn.PixelShuffle(2)
        )
        self.head3 = Deblur_head(int(dim * 2 ** 2), int(dim * 2 ** 1), out_channels)
        self.up32 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 2, dim * 2 ** 3, 1, bias=False),
            nn.PixelShuffle(2)
        )

        self.head2 = Deblur_head(int(dim * 2 ** 1), int(dim), out_channels)
        self.up21 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 1, dim * 2 ** 2, 1, bias=False),
            nn.PixelShuffle(2)
        )

        self.head1 = Deblur_head(dim, dim, out_channels)
        # self.skip4 = torch.nn.Parameter(torch.ones((1, dim * 2**3, 1, 1)), requires_grad=True)
        # self.skip3 = torch.nn.Parameter(torch.ones((1, dim * 2**2, 1, 1)), requires_grad=True)
        # self.skip2 = torch.nn.Parameter(torch.ones((1, dim * 2**1, 1, 1)), requires_grad=True)
        # self.skip1 = torch.nn.Parameter(torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.d4 = nn.Sequential(*[WaveletBlock(dim * 2 ** 3) for _ in range(num_blocks[3])])
        self.d3 = nn.Sequential(*[WaveletBlock(dim * 2 ** 2) for _ in range(num_blocks[2])])
        self.d2 = nn.Sequential(*[WaveletBlock(dim * 2) for _ in range(num_blocks[1])])
        self.d1 = nn.Sequential(*[WaveletBlock(dim) for _ in range(num_blocks[0])])


        self.alpha = nn.Parameter(torch.zeros((1, dim * 2, 1, 1)), requires_grad=True)

    def forward(self, x4, x3, x3_b, x2, x2_b, x1):
        # x = x4.contiguous()
        x = self.d4(x4)
        x4 = self.head4(x)

        x = self.up43(x) + x3
        x = self.d3(x)
        x3 = self.head3(x)

        x2_n = x2.contiguous()
        x = self.up32(x) + x2
        x = self.d2(x)
        x2 = self.head2(x)

        # x = self.d1_conv(torch.cat([self.up21(x2_n), self.up20(x)], 1))
        x = self.up21(x + x2_n * self.alpha) + x1
        # x = self.d0_conv(torch.cat([self.up20(x2_n), x], 1))
        x = self.d1(x)
        # x1_ori = x1.contiguous()
        x1 = self.head1(x)
        # x1 = x1 + x1_ori

        return x1, x2, x3, x4

    def get_wavelet_loss(self):
        wavelet_loss = 0.
        for index, _ in enumerate(self.num_blocks):
            for block in getattr(self, f'd{index+1}'):
                wavelet_loss += block.get_wavelet_loss()
        return wavelet_loss


class MLWNet(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=64,
                 ):

        super(MLWNet, self).__init__()
        # [False, True, True, False]
        # [False, False, False, False]
        self.encoder = Encoder(inp_channels=inp_channels,
                                 dim=dim,
                                 num_blocks=[1, 2, 4, 24],
                                 )
        self.fusion = Fusion(dim=dim,
                         num_blocks=[None, 2, 2, None],
                         )
        self.decoder = Decoder(dim=dim,
                         out_channels=out_channels,
                         num_blocks=[2, 2, 2, 2],
                         )

    def __repr__(self):
        return 'MLWNet'

    def forward(self, inp):
        x = self.encoder(inp)  # (1, 128, 64, 64), (1, 64, 128, 128), (1, 32, 256, 256)
        x = self.fusion(*x)  # (1, 128, 64, 64), (1, 64, 128, 128), (1, 32, 256, 256)
        x1, x2, x3, x4 = self.decoder(*x)  # ()
        return x1 + inp, x2, x3, x4

    def get_wavelet_loss(self):
        return self.fusion.get_wavelet_loss() + self.decoder.get_wavelet_loss()

class MLWNet_Local(Local_Base, MLWNet):
    def __init__(self, *args, base_size=None, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        MLWNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        if base_size is not None:
            base_size = (int(base_size), int(base_size))
        else:
            base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)



if __name__ == '__main__':
    img = torch.zeros((1, 3, 256, 256)).cuda()
    model = MLWNet().eval().cuda()
    with torch.no_grad():
        a = model(img)
    print(model.get_wavelet_loss())
    print(a[0].shape)
    exit(-1)


