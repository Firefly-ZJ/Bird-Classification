#####     Bird Classification     #####
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np

from _BirdNet import BirdNet

### ----- 数据预处理配置 -----
Bilinear = transforms.InterpolationMode.BILINEAR
addNoise = transforms.RandomApply(
    [transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01)],
    p=0.5)

train_transform = transforms.Compose([
    transforms.Resize(256, Bilinear),           # 短边缩放至256
    transforms.RandomRotation(10, Bilinear),    # 随机旋转
    transforms.RandomCrop(224),                 # 中心裁剪224x224
    transforms.RandomHorizontalFlip(),          # 随机水平翻转
    transforms.ToTensor(),
    addNoise,                                   # 添加高斯噪声  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

### ----- 鸟类图像数据集 -----
class BirdDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)
        self.classes = self.dataset.classes
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

### ----- 交叉熵损失 (标签平滑) -----
class CEloss_smooth(nn.Module):
    """Cross entropy loss with label smoothing"""
    def __init__(self, num_classes:int, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        log_prob = torch.log_softmax(pred, dim=-1)
        with torch.no_grad():
            smooth_labels = torch.full_like(log_prob, self.smoothing / (self.num_classes-1))
            smooth_labels.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(-torch.sum(smooth_labels * log_prob, dim=-1))

### ----- 学习率调度器 -----
def create_scheduler(optimizer, num_epochs:int, warmup_epochs:int=5):
    """Create a learning rate scheduler with warmup and cosine annealing"""
    if not num_epochs > warmup_epochs: raise ValueError("num_epochs too small")

    def lr_lambda(current:int):
        if current < warmup_epochs: # Warmup阶段
            return (current + 1) / warmup_epochs
        else: # 余弦退火阶段
            progress = (current-warmup_epochs) / (num_epochs-warmup_epochs)
            return max(0.5 * (1 + np.cos(np.pi * progress)), 0.01)
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

### ----- 训练 -----
def TRAIN(epochs:int=200):
    print("Training...")
    num_classes = 350
    batch_size, accu = 640, 4 # 梯度累积
    init_lr = 2e-3

    # 训练集加载
    train_dataset = BirdDataset(rootPath+"birdData/train", transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    print(f"Data Size: {len(train_dataset)},  Batch Num: {len(train_loader)}")
    
    # 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BirdNet(num_classes).to(device)
    
    # 损失函数与优化器
    criterion = CEloss_smooth(num_classes, smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=0.01)
    scheduler = create_scheduler(optimizer, num_epochs=epochs)
    
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        
        for step, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item() * images.size(0)
            loss = loss / accu # 梯度累积需平均损失
            loss.backward()
            if (step+1) % accu == 0 or (step+1) == len(train_loader):
                optimizer.step() # 更新参数
                optimizer.zero_grad()
        
        scheduler.step()
        epoch_loss = epoch_loss / len(train_dataset)
        print(f"Epoch:{epoch}/{epochs},  Loss={epoch_loss:.4f}")
        if epoch % 50 == 0:
            torch.save(model.state_dict(), rootPath+f"trained/model_{epoch}.pth")
            print("Model saved\n")
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("Training completed")

if __name__ == "__main__":
    rootPath = "./"
    TRAIN()