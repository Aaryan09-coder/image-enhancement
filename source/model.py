import torch
import torch.nn as nn
import torch.nn.functional as F


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(3 * x + 3.0, inplace=self.inplace) / 6.0


class SEModule(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se="SE", nl="RE"):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup
        
        conv_layer = nn.Conv2d
        if nl == "RE":
            nlin_layer = nn.ReLU
        elif nl == "HS":
            nlin_layer = Hswish
        elif nl == "LeRE":
            nlin_layer = nn.LeakyReLU
        elif nl == "HSig":
            nlin_layer = Hsigmoid
        else:
            raise NotImplementedError
            
        if se == "SE":
            SELayer = SEModule
        else:
            SELayer = lambda x: nn.Identity()
        
        if exp != oup:
            self.conv = nn.Sequential(
                conv_layer(inp, exp, 1, 1, 0, bias=True, padding_mode="reflect"),
                nlin_layer(inplace=True),
                conv_layer(exp, exp, kernel, stride=stride, padding=padding, 
                          groups=exp, bias=True, padding_mode="reflect"),
                SELayer(exp),
                nlin_layer(inplace=True),
                conv_layer(exp, oup, 1, 1, 0, bias=True, padding_mode="reflect"),
            )
        else:
            self.conv = nn.Sequential(
                conv_layer(inp, exp, 1, 1, 0, bias=True),
                nlin_layer(inplace=False),
                conv_layer(exp, oup, 1, 1, 0, bias=True),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class AttentionUnetTMO(nn.Module):
    """
    Enhanced U-Net with CBAM attention mechanisms
    Provides better feature extraction and enhancement
    """
    def __init__(self):
        super().__init__()
        base_number = 16
        
        # Encoder
        self.first_conv = MobileBottleneck(3, 3, 3, 1, 6, nl="LeRE")
        self.conv1 = MobileBottleneck(3, base_number, 3, 2, int(base_number * 1.5), False, "LeRE")
        self.attention1 = CBAM(base_number, reduction=8)
        
        self.conv2 = MobileBottleneck(base_number, base_number, 3, 1, int(base_number * 1.5), False, "LeRE")
        self.attention2 = CBAM(base_number, reduction=8)
        
        self.conv3 = MobileBottleneck(base_number, base_number * 2, 3, 2, base_number * 3, False, "LeRE")
        self.attention3 = CBAM(base_number * 2, reduction=8)
        
        # Bottleneck
        self.conv5 = MobileBottleneck(base_number * 2, base_number * 2, 3, 1, base_number * 3, False, "LeRE")
        self.attention_bottleneck = CBAM(base_number * 2, reduction=8)
        
        # Decoder
        self.conv6 = MobileBottleneck(base_number * 2, base_number, 3, 1, base_number * 3, False, "LeRE")
        self.attention6 = CBAM(base_number, reduction=8)
        
        self.conv7 = MobileBottleneck(base_number * 2, base_number, 3, 1, base_number * 3, False, "LeRE")
        self.attention7 = CBAM(base_number, reduction=8)
        
        self.conv8 = MobileBottleneck(base_number, 3, 3, 1, int(base_number * 1.5), False, "LeRE")
        self.last_conv = MobileBottleneck(6, 3, 3, 1, 9, nl="LeRE")
        
        # Final attention for output refinement
        self.output_attention = CBAM(3, reduction=2)

    def forward(self, x):
        x_down = x
        
        # Encoder with attention
        x_1 = self.first_conv(x)
        r = self.conv1(x_1)
        r = self.attention1(r)
        r = self.conv2(r)
        r = self.attention2(r)
        r_d2 = r
        
        r = self.conv3(r)
        r = self.attention3(r)
        
        # Bottleneck with attention
        r = self.conv5(r)
        r = self.attention_bottleneck(r)
        
        # Decoder with attention
        r = self.conv6(r)
        r = self.attention6(r)
        r = F.interpolate(r, (r_d2.shape[2], r_d2.shape[3]), mode="bilinear", align_corners=True)
        
        r = self.conv7(torch.cat([r_d2, r], dim=1))
        r = self.attention7(r)
        
        r = self.conv8(r)
        r = F.interpolate(r, (x_down.shape[2], x_down.shape[3]), mode="bilinear", align_corners=True)
        
        r = self.last_conv(torch.cat([x_1, r], dim=1))
        r = torch.abs(r + 1)
        
        # Apply output attention for refinement
        x_enhanced = 1 - (1 - x) ** r
        x_enhanced = self.output_attention(x_enhanced)
        
        return x_enhanced, r


class UnetTMO(nn.Module):
    """Original U-Net architecture (kept for compatibility)"""
    def __init__(self):
        super().__init__()
        self.first_conv = MobileBottleneck(3, 3, 3, 1, 6, nl="LeRE")
        base_number = 16
        self.conv1 = MobileBottleneck(3, base_number, 3, 2, int(base_number * 1.5), False, "LeRE")
        self.conv2 = MobileBottleneck(base_number, base_number, 3, 1, int(base_number * 1.5), False, "LeRE")
        self.conv3 = MobileBottleneck(base_number, base_number * 2, 3, 2, base_number * 3, False, "LeRE")
        self.conv5 = MobileBottleneck(base_number * 2, base_number * 2, 3, 1, base_number * 3, False, "LeRE")
        self.conv6 = MobileBottleneck(base_number * 2, base_number, 3, 1, base_number * 3, False, "LeRE")
        self.conv7 = MobileBottleneck(base_number * 2, base_number, 3, 1, base_number * 3, False, "LeRE")
        self.conv8 = MobileBottleneck(base_number, 3, 3, 1, int(base_number * 1.5), False, "LeRE")
        self.last_conv = MobileBottleneck(6, 3, 3, 1, 9, nl="LeRE")

    def forward(self, x):
        x_down = x
        x_1 = self.first_conv(x)
        r = self.conv1(x_1)
        r = self.conv2(r)
        r_d2 = r
        r = self.conv3(r)
        r = self.conv5(r)
        r = self.conv6(r)
        r = F.interpolate(r, (r_d2.shape[2], r_d2.shape[3]), mode="bilinear", align_corners=True)
        r = self.conv7(torch.cat([r_d2, r], dim=1))
        r = self.conv8(r)
        r = F.interpolate(r, (x_down.shape[2], x_down.shape[3]), mode="bilinear", align_corners=True)
        r = self.last_conv(torch.cat([x_1, r], dim=1))
        r = torch.abs(r + 1)
        x = 1 - (1 - x) ** r
        return x, r