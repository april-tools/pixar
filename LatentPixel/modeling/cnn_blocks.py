import torch
from torch import nn

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def Normalize(in_channels: int, num_groups: int=32) -> nn.GroupNorm:
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )

    def forward(self, x: torch.Tensor):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=0
            )

    def forward(self, x: torch.Tensor):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int | None=None, conv_shortcut: bool=False, dropout: float, norm_groups: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_groups)
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.norm2 = Normalize(out_channels, norm_groups)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )

    def forward(self, x: torch.Tensor):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h
    

class CNNEncoder(nn.Module):
    
    def __init__(
            self, 
            *, 
            in_channels: int, 
            hidden_channels: int, 
            num_downsample: int, 
            num_res: int,
            hidden_dim: int,
            dropout: float,
            norm_groups: int
        ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_downsample = num_downsample
        self.num_res = num_res
        self.hidden_dim = hidden_dim
        self.dropout = dropout 
        
        self.in_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        blocks = []
        for _ in range(self.num_downsample):
            block = nn.Sequential(
                *[ResnetBlock(in_channels=self.hidden_channels, dropout=self.dropout, norm_groups=norm_groups) for _ in range(self.num_res)]
            )
            blocks.append(block)
            blocks.append(Downsample(in_channels=self.hidden_channels, with_conv=True))
        
        self.blocks = nn.Sequential(*blocks)
        
        self.norm_out = Normalize(self.hidden_channels, norm_groups)
        self.out_conv = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        x = self.blocks(x)
        x = self.norm_out(x)
        return self.out_conv(x)


class CNNDecoder(nn.Module):
    
    def __init__(
            self, 
            *, 
            target_channels: int, 
            hidden_channels: int, 
            num_upsample: int, 
            num_res: int,
            hidden_dim: int,
            dropout: float,
            norm_groups: int
        ) -> None:
        super().__init__()
        
        self.target_channels = target_channels
        self.hidden_channels = hidden_channels
        self.num_upsample = num_upsample
        self.num_res = num_res
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        self.in_conv = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        blocks = []
        for _ in range(self.num_upsample):
            block = nn.Sequential(
                *[ResnetBlock(in_channels=self.hidden_channels, dropout=self.dropout, norm_groups=norm_groups) for _ in range(self.num_res)]
            )
            blocks.append(block)
            blocks.append(Upsample(in_channels=self.hidden_channels, with_conv=True))
        
        self.blocks = nn.Sequential(*blocks)
        
        # self.norm_out = Normalize(self.hidden_channels, norm_groups)
        self.out_conv = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.target_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        x = self.blocks(x)
        # x = self.norm_out(x)
        return self.out_conv(x)
    

if __name__ == '__main__' :
    img = torch.randn([32, 128, 16, 1600])
    resblock = ResnetBlock(in_channels=128, out_channels=128, dropout=0.1)
    resblockout = resblock(img)
    print('shape of resblock out:', resblockout.shape)
    downblock = Downsample(128, True)
    downout = downblock(resblockout)
    print('shape of the downsample block out:', downout.shape)
    
    upblock = Upsample(128, True)
    upout = upblock(downout)
    print('shape of the upsample block output', upout.shape)
    