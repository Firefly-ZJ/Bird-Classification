#####     Bird Network     #####
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, dim:int, scale_init=1e-6):
        """Residual Block (7x7 dwConv → LN → 1x1 Conv → Act → 1x1 Conv → Scale)"""
        super().__init__()
        self.scale_init = scale_init
        self.skip = nn.Identity() # skip connection

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding="same", groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim*4) # 1x1 conv (by linear)
        self.act = nn.ReLU()
        self.pwconv2 = nn.Linear(dim*4, dim) # 1x1 conv (by linear)
        self.gamma = nn.Parameter(self.scale_init * torch.ones((dim)), requires_grad=True) # layer scale
        
    def forward(self, x):
        skip = self.skip(x)
        res = self.dwconv(x)
        res = res.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        res = self.norm(res)
        res = self.act(self.pwconv1(res))
        res = self.gamma * (self.pwconv2(res))
        res = res.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return skip + res

class DownSample(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, height:int, width:int=0):
        """Downsample Block (with MaxPool)"""
        super().__init__()
        if not width: width = height
        self.norm = nn.LayerNorm((in_dim, height, width), eps=1e-6)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=1)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.pool(x)
        x = self.conv(x)
        return x

class BirdNet(nn.Module):
    def __init__(self, dims=(64, 128, 256, 512), layers=(3, 4, 6, 3), class_num:int=384):
        """Network for Bird Classification
        Args:
            dims (tuple): channels of each stage
            layers (tuple): blocks of each stage
            class_num (int): num of output classes
        """
        if not (len(dims) == len(layers) == 4):
            raise ValueError("Dims and layers must be 4 stages.")
        super().__init__()
        self.classNum = class_num
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm((dims[0], 56, 56), eps=1e-6))
        self.conv1 = nn.Sequential(*[ResBlock(dims[0]) for _ in range(layers[0])]) # 56*56

        self.down2 = DownSample(dims[0], dims[1], 56, 56)
        self.conv2 = nn.Sequential(*[ResBlock(dims[1]) for _ in range(layers[1])]) # 28*28
        
        self.down3 = DownSample(dims[1], dims[2], 28, 28)
        self.conv3 = nn.Sequential(*[ResBlock(dims[2]) for _ in range(layers[2])]) # 14*14
        
        self.down4 = DownSample(dims[2], dims[3], 14, 14)
        self.conv4 = nn.Sequential(*[ResBlock(dims[3]) for _ in range(layers[3])]) # 7*7
        
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], self.classNum)

    def forward(self, x): # Input: [B, 3, 224, 224]
        x = self.conv1(self.stem(x)) # [B, 64, 56, 56]
        x = self.conv2(self.down2(x)) # [B, 128, 28, 28]
        x = self.conv3(self.down3(x)) # [B, 256, 14, 14]
        x = self.conv4(self.down4(x)) # [B, 512, 7, 7]
        
        Sp = self.norm(x.mean([-2, -1])) # global average pooling -> (B, 512)
        Sp = self.head(Sp)  # [B, SpNum]
        return Sp
    
    def getClassNum(self) -> int:
        return self.classNum

def getModel(version:str, load_weight=False) -> BirdNet:
    """BirdNet model loader.
    Args:
        version (str): model version
            - "v1base": Version 1 (base size)
            - "v1large": Version 1 (large size)
        load_weight (bool, str): load pretrained weights (bool); Path/To/Your/Weights/File (str)
    """
    models = {"v1base": "./trained/model_v1base.pth",
              "v1large": "./trained/model_v1large.pth"}
    if version == "v1base":
        net = BirdNet(dims=(64, 128, 256, 512), layers=(3, 4, 6, 3))
    elif version == "v1large":
        net = BirdNet(dims=(96, 192, 384, 768), layers=(3, 6, 12, 3))
    else:
        raise ValueError(f"Unsupported version: {version}")
    if isinstance(load_weight, str):
        net.load_state_dict(torch.load(load_weight, map_location="cpu", weights_only=True))
    elif load_weight:
        net.load_state_dict(torch.load(models[version], map_location="cpu", weights_only=True))
    return net

if __name__ == "__main__":
    from torchinfo import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BirdNet(dims=(96, 192, 384, 768), layers=(3, 6, 12, 3)).to(device)
    Sp = net(torch.randn(*(16, 3, 224, 224)).float().to(device))
    print(Sp.shape)
    print(summary(net, (256, 3, 224, 224)))

# ----- V1 base, B=512 ----- #
# Total params: 11,756,736
# Estimated Total Size (MB): 43723.87

# ----- V1 large, B=256 ----- #
# Total params: 33,132,384
# Estimated Total Size (MB): 43602.53