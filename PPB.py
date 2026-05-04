import torch.nn as nn 
import torch.nn.functional as F
import torch


class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, channels=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        kernel = self.gauss_kernel(channels=channels)
        self.register_buffer('kernel', kernel)

    def gauss_kernel(self, channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        # kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def gauss_decom(self, img):
        current = img
        pyr = [img]
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            pyr.append(down)
            current = down
        return pyr

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            
            up = self.upsample(down)
            
            if up.shape[2:] != current.shape[2:]:
                up = F.interpolate(up, size=current.shape[2:], mode='bilinear', align_corners=False)
                
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr