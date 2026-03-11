import torch.nn as nn
import torch
import segmentation_models_pytorch as smp

class LocalMambaBlock(nn.Module):
    """
    Efficiently mixes tokens with linear complexity.
    """
    def __init__(self, dim, kernel_size=5, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Depthwise conv mixes spatial information locally
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Tokens, Dim)
        shortcut = x
        x = self.norm(x)
        # Gating mechanism
        g = torch.sigmoid(self.gate(x))
        x = x * g
        # Spatial mixing via 1D Conv (requires transpose)
        x = x.transpose(1, 2)  # -> (B, D, N)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # -> (B, N, D)
        # Projection
        x = self.proj(x)
        x = self.drop(x)
        return shortcut + x

class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=1,pretrained=True, in_channels=4):
        """
        Initializes the EfficientNetV2 model for segmentation.

        Args:
            num_classes (int): Number of output classes.
            encoder_name (str): Name of the encoder backbone.
            pretrained (bool): Whether to use pretrained weights.
            in_channels (int): Number of input channels.
        """
        super().__init__()
        self.smp_model = smp.Unet(
            encoder_name='timm-efficientnet-b2',
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
        )

    def forward(self, x):
        return self.smp_model(x)

class MiT(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, in_channels=4):
        super().__init__()
        self.model = smp.Unet(
            encoder_name='mit_b3',
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
        )
    def forward(self, x):
        return self.smp_model(x)

class ConvNext(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, in_channels=4):
        super().__init__()
        self.model = smp.Unet(
            encoder_name='timm-convnext-small',
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
        )
    def forward(self, x):
        return self.model(x)