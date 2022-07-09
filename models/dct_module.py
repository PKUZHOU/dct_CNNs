import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import cv2

# the DCT module converts a RGB image to DCT coefficients
# only used for training, remove it when deploying the dct models.
class DCT(nn.Module):
        def __init__(self):
            super(DCT, self).__init__()

            self.dct_conv = nn.Conv2d(in_channels= 3, out_channels= 192, kernel_size= 8, stride= 8, bias=False, groups=3)  # 3 h w -> 192 h/8 w/8
            self.dct_conv_weight = torch.cat([self.get_dct_filter(8)] * 3,
                                                  dim=0) 
            self.dct_conv.weight.data = self.dct_conv_weight  # 192 1 8 8
            self.dct_conv.weight.requires_grad = False

            self.Ycbcr = nn.Conv2d(3, 3, 1, 1, bias=False)
            trans_matrix = np.array([[0.299, 0.587, 0.114],
                                     [-0.169, -0.331, 0.5],
                                     [0.5, -0.419, -0.081]])   # a simillar version, maybe be a little wrong
            self.trans_matrix = torch.from_numpy(trans_matrix).float().unsqueeze(
                2).unsqueeze(3)
            self.Ycbcr.weight.data = self.trans_matrix
            self.Ycbcr.weight.requires_grad = False

            self.reYcbcr = nn.Conv2d(3, 3, 1, 1, bias=False)
            re_matrix = np.linalg.pinv(np.array([[0.299, 0.587, 0.114],
                                                 [-0.169, -0.331, 0.5],
                                                 [0.5, -0.419, -0.081]]))
            re_matrix = torch.from_numpy(re_matrix).float().unsqueeze(
                2).unsqueeze(3)
            self.reYcbcr.weight.data = re_matrix

            self.cuda()
            
        def cuda(self):
            self.trans_matrix = self.trans_matrix.cuda()
            self.dct_conv_weight = self.dct_conv_weight.cuda()

        def forward(self, x):
            # jpg = (jpg * self.std) + self.mean # 0-1
            ycbcr = F.conv2d(x, self.Ycbcr.weight.data)
            dct = F.conv2d(ycbcr, self.dct_conv.weight.data, stride = 8, groups=3)
            return dct

        def reverse(self, x):
            ycbcr = F.conv_transpose2d(x, torch.cat([self.weight] * 3, 0),
                                     bias=None, stride=8, groups=3)
            rgb = self.reYcbcr(ycbcr)
            return rgb, ycbcr

        def build_filter(self, pos, freq, POS):
            # freq : u,v
            # pos : i,j
            # POS : N
            result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
            if freq == 0:
                return result
            else:
                return result * math.sqrt(2)
        
        def get_dct_filter(self, N):
            dct_filter = torch.zeros(N*N, 1, N, N)

            for u in range(N):
                for v in range(N):
                    for i in range(N):
                        for j in range(N):
                            dct_filter[u * N + v, 0, i, j] = self.build_filter(i, u, N) * self.build_filter(j, v, N)
                            
            return dct_filter


if __name__ == '__main__':
    dct_moudle = DCT()
    x = torch.ones((1,3,224,224))
    dct = dct_moudle(x)
    print(dct)
