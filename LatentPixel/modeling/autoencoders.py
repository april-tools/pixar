import torch
from torch import nn
from .cnn_blocks import (
    Downsample,
    Upsample,
    ResnetBlock,
    Normalize
)

class CNNEncoder(nn.Module):
    
    def __init__(
            self, 
            *, 
            in_channels: int, 
            hidden_channels: int, 
            num_downsample: int, 
            num_res: int,
            hidden_dim: int,
            dropout: float
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
                *[ResnetBlock(in_channels=self.hidden_channels, dropout=self.dropout) for _ in range(self.num_res)]
            )
            blocks.append(block)
            blocks.append(Downsample(in_channels=self.hidden_channels, with_conv=True))
        
        self.blocks = nn.Sequential(*blocks)
        
        self.norm_out = Normalize(self.hidden_channels)
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
            dropout: float
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
                *[ResnetBlock(in_channels=self.hidden_channels, dropout=self.dropout) for _ in range(self.num_res)]
            )
            blocks.append(block)
            blocks.append(Upsample(in_channels=self.hidden_channels, with_conv=True))
        
        self.blocks = nn.Sequential(*blocks)
        
        self.norm_out = Normalize(self.hidden_channels)
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
        x = self.norm_out(x)
        return self.out_conv(x)
