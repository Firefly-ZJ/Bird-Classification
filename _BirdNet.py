#####     Bird Network     #####
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """Residual Block"""
    def __init__(self, dim:int, scale_init=1e-6):
        super().__init__()
        self.scale_init = scale_init
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim*4) # 1x1 conv (by linear)
        self.act = nn.ReLU()
        self.pwconv2 = nn.Linear(dim*4, dim) # 1x1 conv (by linear)
        self.gamma = nn.Parameter(self.scale_init * torch.ones((dim)), requires_grad=True) # layer scale
        
    def forward(self, x):
        input = x

        res = self.dwconv(x)
        res = res.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        res = self.norm(res)
        res = self.pwconv1(res)
        res = self.pwconv2(self.act(res))
        res = self.gamma * res
        res = res.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        return res + input

class BirdNet(nn.Module):
    """Network for Bird Classification
    Args:
        species_num (int): num of output classes (> real number)
        dims: channels of each stage (default: 64 → 512))
    """
    def __init__(self, species_num:int=360, dims=(64, 128, 256, 512)):
        super().__init__()
        self.speciesNum = species_num # number of species
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm((dims[0], 56, 56), eps=1e-6))
        self.conv1 = nn.Sequential(
            *[ResBlock(dims[0]) for _ in range(3)])

        self.down2 = nn.Sequential(
            nn.LayerNorm((dims[0], 56, 56), eps=1e-6),
            nn.Conv2d(dims[0],dims[1], kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
            *[ResBlock(dims[1]) for _ in range(3)])
        
        self.down3 = nn.Sequential(
            nn.LayerNorm((dims[1], 28, 28), eps=1e-6),
            nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2))
        self.conv3 = nn.Sequential(
            *[ResBlock(dims[2]) for _ in range(6)])
        
        self.down4 = nn.Sequential(
            nn.LayerNorm((dims[2], 14, 14), eps=1e-6),
            nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2))
        self.conv4 = nn.Sequential(
            *[ResBlock(dims[3]) for _ in range(3)])
        
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], self.speciesNum)

    def forward(self, x): # Input: [B, 3, 224, 224]
        x = self.conv1(self.stem(x)) # [B, 64, 56, 56]
        x = self.conv2(self.down2(x)) # [B, 128, 28, 28]
        x = self.conv3(self.down3(x)) # [B, 256, 14, 14]
        x = self.conv4(self.down4(x)) # [B, 512, 7, 7]
        
        Sp = self.norm(x.mean([-2, -1])) # global average pooling -> (B, 512)
        Sp = self.head(Sp)  # [B, SpNum]
        return Sp

if __name__ == "__main__":
    from torchinfo import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BirdNet().to(device)
    testSize = (32, 3, 224, 224)
    Sp = net(torch.randn(*testSize).float().to(device))
    print(Sp.shape)
    print(summary(BirdNet(), testSize))