"""UKAN-CBAM main architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import KANLayer, ConvLayer, PatchEmbed

class UKAN_CBAM(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 img_size=224, patch_size=16, in_chans=3, embed_dims=[256, 320, 512], 
                 no_kan=False, drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, 
                 depths=[1, 1, 1], add_cbam=False):
        super().__init__()

        kan_input_dim = embed_dims[0]

        self.encoder1 = ConvLayer(3, kan_input_dim // 8)  
        self.encoder2 = ConvLayer(kan_input_dim // 8, kan_input_dim // 4)  
        self.encoder3 = ConvLayer(kan_input_dim // 4, kan_input_dim)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([KANLayer(
            in_features=embed_dims[1], 
            hidden_features=embed_dims[1], 
            drop=drop_rate, add_cbam=add_cbam, no_kan=no_kan)])

        self.block2 = nn.ModuleList([KANLayer(
            in_features=embed_dims[2],
            hidden_features=embed_dims[2], 
            drop=drop_rate, add_cbam=add_cbam, no_kan=no_kan)])

        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, 
                                       in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, 
                                       in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.decoder1 = ConvLayer(embed_dims[2], embed_dims[1])  
        self.decoder2 = ConvLayer(embed_dims[1], embed_dims[0])  
        self.decoder3 = ConvLayer(embed_dims[0], embed_dims[0] // 4) 
        self.decoder4 = ConvLayer(embed_dims[0] // 4, embed_dims[0] // 8)
        self.decoder5 = ConvLayer(embed_dims[0] // 8, embed_dims[0] // 8)

        self.final = nn.Conv2d(embed_dims[0] // 8, num_classes, kernel_size=1)
        self.soft = nn.Sigmoid()

    def forward(self, x):
        B = x.shape[0]

        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out

        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out

        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out

        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t4)
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t3)
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        return self.final(out)
