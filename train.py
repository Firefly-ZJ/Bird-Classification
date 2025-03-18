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
    p=0.25)

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

val_transform = transforms.Compose([
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
    """Cross Entropy Loss with Label Smoothing"""
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        
    def forward(self, pred, target):
        log_prob = torch.log_softmax(pred, dim=-1)
        with torch.no_grad():
            smooth_labels = torch.full_like(log_prob, self.smoothing / (self.num_classes - 1))
            smooth_labels.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(-torch.sum(smooth_labels * log_prob, dim=-1))

### ----- 学习率调度器 -----
def create_scheduler(optimizer, num_epochs, warmup_epochs=10):
    """Create a learning rate scheduler with warmup and cosine annealing"""
    if not num_epochs > warmup_epochs: raise ValueError("num_epochs too small")

    def lr_lambda(current_epoch:int):
        if current_epoch < warmup_epochs:  # Warmup阶段
            return (current_epoch + 1) / warmup_epochs
        else:  # 余弦退火阶段
            progress = (current_epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return max(0.5 * (1 + np.cos(np.pi * progress)), 0.01)
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

### ----- 训练 -----
def main():
    species_num = 360
    epochs, batch_size = 200, 32

    # 数据集加载
    train_dataset = BirdDataset("./birdData/train", transform=train_transform)
    val_dataset = BirdDataset("./birdData/val", transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size,
                              shuffle=True, num_workers=8)
    #val_loader = DataLoader(val_dataset, batch_size*4, num_workers=4)
    
    # 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BirdNet(species_num).to(device)
    
    # 损失函数与优化器
    criterion = CEloss_smooth(species_num, smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    scheduler = create_scheduler(optimizer, num_epochs=epochs, warmup_epochs=10)
    
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
        
        scheduler.step()
        epoch_loss = epoch_loss / len(train_dataset)
        print(f"Epoch:{epoch}/{epochs},  Loss={epoch_loss:.4f}")
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"./trained/model_{epoch}.pth")
            print("Model saved\n")
            print()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == "__main__":
    main()