import torch
import torch.nn as nn
import yaml

from Encoder import Encoder 


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class LFINet(nn.Module):
    def __init__(self, config):
        super(LFINet, self).__init__()
        if 'model' in config:
            model_cfg = config['model']
        else:
            model_cfg = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        llf_cfg = {
            'gpu_ids': None,
            'attn_dw_dilation':model_cfg['attn_dw_dilation'],
            'attn_channel_split':model_cfg['attn_channel_split'],
            'transformer': {
                'input_channel': 3,
                'embed_dim': model_cfg['llf_embed_dim'],
                'output_channel': model_cfg['llf_embed_dim'],
                'depths': model_cfg['llf_depths'],
                'num_heads': model_cfg['llf_num_heads'],
                'window_sizes': model_cfg['llf_window_sizes'],
                'back_RBs': 0,
            }
        }
        self.llf_encoder = Encoder(llf_cfg)

        base_c = model_cfg['base_channels']

        self.proj_llf_256 = nn.Conv2d(32, base_c, 1)
        self.proj_llf_128 = nn.Conv2d(32, base_c * 2, 1)
        self.proj_llf_64  = nn.Conv2d(32, base_c * 4, 1)
        self.proj_llf_32  = nn.Conv2d(model_cfg['llf_embed_dim'], base_c * 8, 1)

        self.up4 = self.upsample(base_c * 16, base_c * 8)
        self.conv4 = self.conv_stage(base_c * (8 + 8), base_c * 8)
        self.up3 = self.upsample(base_c * 8, base_c * 4)
        self.conv3 = self.conv_stage(base_c * (4 + 4), base_c * 4)
        self.up2 = self.upsample(base_c * 4, base_c * 2)
        self.conv2 = self.conv_stage(base_c * (2 + 2), base_c * 2)
        self.up1 = self.upsample(base_c * 2, base_c)
        self.conv1 = self.conv_stage(base_c * (1 + 1), base_c)
        self.final = nn.Sequential(nn.Conv2d(base_c, 1, 3, 1, 1),nn.Sigmoid())

        self.max_pool = nn.MaxPool2d(2)
        self.center_proj = nn.Conv2d(base_c * 8, base_c * 16, 1)

        # 模型移到设备
        self.to(self.device)

    def upsample(self, c_in, c_out):
        return nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True)
        )

    def conv_stage(self, c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 1, 1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, 1, 1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.to(next(self.parameters()).device)

        llf_feats = self.llf_encoder(x)      

        llf_256 = self.proj_llf_256(llf_feats[0])
        llf_128 = self.proj_llf_128(llf_feats[1])
        llf_64  = self.proj_llf_64(llf_feats[2])
        llf_32  = self.proj_llf_32(llf_feats[3])

        # center
        center = self.max_pool(llf_32)
        center = self.center_proj(center)

        # 解码
        out = self.up4(center)
        out = self.conv4(torch.cat([out, llf_32], dim=1))
        out = self.up3(out)
        out = self.conv3(torch.cat([out, llf_64], dim=1))
        out = self.up2(out)
        out = self.conv2(torch.cat([out, llf_128], dim=1))
        out = self.up1(out)
        out = self.conv1(torch.cat([out, llf_256], dim=1))
        out = self.final(out)

        return out