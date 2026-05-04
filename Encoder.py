import torch
import torch.nn as nn
import torch.nn.functional as F
from Spatial_Transformer import Spatial_Transformer
from PPB import Lap_Pyramid_Conv


# ==========  频率自适应门控调制模块 (FGM) ==========
class FrequencyGatedModulation(nn.Module):
    def __init__(self, high_channels=32, low_channels=96):
        super().__init__()
        
        self.low_to_gate = nn.Sequential(
            nn.Conv2d(low_channels, high_channels, 1), 
            nn.BatchNorm2d(high_channels),
            nn.GELU(), 
            nn.Conv2d(high_channels, high_channels, 3, padding=1, groups=high_channels),  # DWConv
            nn.Sigmoid()  
        )
        
        self.high_enhance = nn.Sequential(
            nn.Conv2d(high_channels, high_channels, 3, padding=1, groups=high_channels),
            nn.BatchNorm2d(high_channels),
            nn.GELU()
        )
        
        self.freq_diff = nn.Sequential(
        nn.Conv2d(high_channels + low_channels, high_channels, 1),
        nn.BatchNorm2d(high_channels),
        nn.GELU()
    )

        self.fusion_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, 3, 1),  # 3个分支
            nn.Softmax(dim=1)
        )
        
        self.residual_proj = nn.Conv2d(high_channels, high_channels, 1)
        
    def forward(self, high_feat, low_feat):

        low_upsampled = F.interpolate(low_feat, size=high_feat.shape[2:], 
                                      mode='bilinear', align_corners=False)
        

        gate = self.low_to_gate(low_upsampled)
        high_enhanced = self.high_enhance(high_feat)
        gated_feat = gate * high_enhanced  
        
        low_aligned = F.adaptive_avg_pool2d(low_upsampled, 1)  
        low_aligned = F.interpolate(low_aligned, size=high_feat.shape[2:], mode='nearest')
        high_global = F.adaptive_avg_pool2d(high_feat, 1)
        high_global = F.interpolate(high_global, size=high_feat.shape[2:], mode='nearest')
        diff_feat = self.freq_diff(torch.cat([high_feat, low_upsampled], dim=1))
        
        original_feat = self.residual_proj(high_feat)
        
        fusion_input = gated_feat + diff_feat + original_feat
        weights = self.fusion_weight(fusion_input)  # (B, 3, 1, 1)
        
        w1, w2, w3 = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        modulated_feat = w1 * gated_feat + w2 * diff_feat + w3 * original_feat
        
        return modulated_feat


# ========== 集成到LLF_Encoder ==========
class MultiOrderDWConv(nn.Module):
    def __init__(self, embed_dims, dw_dilation=[1, 2, 3], channel_split=[1, 3, 4]):
        super().__init__()
        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        
        self.DW_conv0 = nn.Conv2d(embed_dims, embed_dims, 5,
                                  padding=(1 + 4 * dw_dilation[0]) // 2,
                                  groups=embed_dims, dilation=dw_dilation[0])
        self.DW_conv1 = nn.Conv2d(self.embed_dims_1, self.embed_dims_1, 5,
                                  padding=(1 + 4 * dw_dilation[1]) // 2,
                                  groups=self.embed_dims_1, dilation=dw_dilation[1])
        self.DW_conv2 = nn.Conv2d(self.embed_dims_2, self.embed_dims_2, 7,
                                  padding=(1 + 6 * dw_dilation[2]) // 2,
                                  groups=self.embed_dims_2, dilation=dw_dilation[2])
        self.PW_conv = nn.Conv2d(embed_dims, embed_dims, 1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(x_0[:, self.embed_dims_0:self.embed_dims_0+self.embed_dims_1])
        x_2 = self.DW_conv2(x_0[:, self.embed_dims_0+self.embed_dims_1:])
        x = torch.cat([x_0[:, :self.embed_dims_0], x_1, x_2], dim=1)
        return self.PW_conv(x)


class HighFrequencyBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, 
                 dw_dilation=[1, 2, 3], channel_split=[1, 3, 4]):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.spatial_agg = MultiOrderDWConv(out_channels, dw_dilation, channel_split)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.proj(x)
        spatial_feat = self.spatial_agg(x)
        attn = self.channel_attn(spatial_feat)
        return spatial_feat * attn + x


class Encoder(nn.Module):
    def __init__(self, config, use_fagm=True, fagm_lite=False):
        super().__init__()
        self.transformer_config = config['transformer']
        self.use_fagm = use_fagm
        
        input_c = self.transformer_config['input_channel']
        embed_dim = self.transformer_config['embed_dim']
        
        self.pyramid = Lap_Pyramid_Conv(num_high=3, channels=input_c)
        
        self.low_freq_encoder = Spatial_Transformer(
            in_chans=input_c,
            embed_dim=embed_dim,
            depths=self.transformer_config['depths'],
            num_heads=self.transformer_config['num_heads'],
            window_sizes=self.transformer_config['window_sizes'],
            back_RBs=self.transformer_config['back_RBs']
        )
        
        dw_dilation = config.get('attn_dw_dilation', [1, 2, 3])
        channel_split = config.get('attn_channel_split', [1, 3, 4])
        
        self.high_freq_encoders = nn.ModuleList([
            HighFrequencyBlock(input_c, 32, dw_dilation, channel_split),
            HighFrequencyBlock(input_c, 32, dw_dilation, channel_split),
            HighFrequencyBlock(input_c, 32, dw_dilation, channel_split),
        ])

        if self.use_fagm:
            self.fagm_modules = nn.ModuleList([
                FrequencyGatedModulation(high_channels=32, low_channels=embed_dim)
                for _ in range(3)
            ])
        else:
            self.fagm_modules = None

    def forward(self, input_image):
        pyr_feats = self.pyramid.pyramid_decom(input_image)
        
        low_feat = self.low_freq_encoder(pyr_feats[-1])
        if isinstance(low_feat, list):
            low_feat = low_feat[-1]  
        
        high_feats = []
        for i, encoder in enumerate(self.high_freq_encoders):
            high_f = encoder(pyr_feats[i])  
            
            if self.use_fagm and self.fagm_modules is not None:
                high_f = self.fagm_modules[i](high_f, low_feat)
            
            high_feats.append(high_f)
        
        return high_feats + [low_feat]