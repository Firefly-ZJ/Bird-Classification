#####     Bird Classification     #####
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
from time import time

from _BirdNet import getModel

device = torch.device("cuda" if torch.cuda.is_available() else
                      "xpu" if torch.xpu.is_available() else "cpu")

### ---------- 数据预处理配置 ----------
bilinear = transforms.InterpolationMode.BILINEAR
addNoise = transforms.RandomApply(
    [transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01)],
    p=0.5)

train_transform = transforms.Compose([
    transforms.Resize(256, bilinear),           # 短边缩放至256
    transforms.RandomHorizontalFlip(p=0.3),     # 随机水平翻转
    transforms.RandomRotation(15, bilinear),    # 随机旋转
    transforms.RandomCrop(224),                 # 中心裁剪224x224
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

### ---------- 交叉熵损失 (标签平滑) ----------
class CEloss_smooth(nn.Module):
    """Cross entropy loss with label smoothing"""
    def __init__(self, num_classes:int, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, pred, target):
        log_prob = self.log_softmax(pred)
        with torch.no_grad():
            smooth_labels = torch.full_like(log_prob, self.smoothing / (self.num_classes-1))
            smooth_labels.scatter_(1, target.unsqueeze(1), 1-self.smoothing)
        return torch.mean(-torch.sum(smooth_labels * log_prob, dim=-1))

### ---------- 学习率调度器 ----------
def create_scheduler(optimizer, total_epochs:int, warmup_epochs:int=5):
    """Create a learning rate scheduler with warmup and cosine annealing"""
    def lr_lambda(current:int) -> float:
        if current < warmup_epochs: # Warmup阶段
            return (current + 1) / warmup_epochs
        else: # 余弦退火阶段
            progress = current / total_epochs
            return max(0.5 * (1 + np.cos(np.pi * progress)), 0.01)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler

### ---------- 训练 ----------
class Trainer():
    def __init__(self, version:str):
        self.model = getModel(version, load_weight=False).to(device)
        self.criterion = CEloss_smooth(self.model.getClassNum())
        
    def TRAIN(self, epochs:int, train_batch:int, init_lr:float, accu:int=1, eval_batch:int=0):
        """Train the model.
        Args:
            epochs (int): Num of epochs to train.
            train_batch (int): Batch size for training. (B = train_batch * accu)
            init_lr (float): Initial learning rate.
            accu (int, optional): Num of gradient accumulations. (Default: 1)
            eval_batch (int, optional): Batch size for evaluation. Same as training batch if 0.
        """
        if accu < 1: accu = 1
        if eval_batch == 0: eval_batch = train_batch
        start_time = time()
        print(f"Epochs: {epochs},  Learning Rate: {init_lr},  Batch Size: {train_batch}")
        # 数据集加载
        train_dataset = ImageFolder(rootPath+"birdData/train", transform=train_transform)
        train_loader = DataLoader(train_dataset, train_batch, shuffle=True, num_workers=12)
        test_dataset = ImageFolder(rootPath+"birdData/val", transform=test_transform)
        test_loader = DataLoader(test_dataset, eval_batch, shuffle=False, num_workers=12)
        print(f"Data Size: {len(train_dataset)},  Batch Num: {len(train_loader)}\n")
        # 优化器 & 学习率调度器
        optimizer = optim.AdamW(self.model.parameters(), init_lr, weight_decay=0.02)
        scheduler = create_scheduler(optimizer, total_epochs=epochs)
        
        ### 训练循环
        for epoch in range(1, epochs+1):
            print(f"Epoch: {epoch} / {epochs}", end=",  ")
            self.model.train()
            optimizer.zero_grad()
            epoch_loss = 0.
            epoch_accu = 0
            
            for step, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                epoch_loss += loss.item() * images.size(0) # 累计损失
                _, preds_max = torch.max(outputs, dim=1)
                epoch_accu += torch.sum(preds_max == labels).item() #

                loss = loss * (images.size(0)/train_batch) / accu # 梯度累积需平均损失
                loss.backward()
                if (step+1) % accu == 0 or (step+1) == len(train_loader):
                    optimizer.step() # 更新参数
                    optimizer.zero_grad()
            
            scheduler.step()
            epoch_loss = epoch_loss / len(train_dataset)
            epoch_accu = epoch_accu / len(train_dataset) * 100
            print(f"Loss: {epoch_loss:.4f},  Accuracy: {epoch_accu:.2f}%")
            if epoch % 10 == 0:
                self.eval(test_loader)
                print(f"Time: {(time()-start_time)/60:.0f} min")
                self.save(rootPath+f"trained/model_{epoch}.pth")

        print("Training completed")

    @torch.no_grad()
    def eval(self, test_loader:DataLoader):
        self.model.eval()
        total_loss = 0.0
        total_correct_max, total_correct_top3 = 0, 0
        for (images, labels) in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = self.model(images)
            # 计算损失
            loss = self.criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            # 计算准确率，max & top3
            _, preds_max = torch.max(outputs, dim=1)
            total_correct_max += torch.sum(preds_max == labels).item()
            _, preds_top3 = torch.topk(outputs, k=3, dim=1)
            correct_mask = torch.eq(preds_top3, labels.view(-1, 1))
            total_correct_top3 += torch.sum(correct_mask.any(dim=1)).item()
        
        test_size = len(test_loader.dataset)
        avg_loss = total_loss / test_size
        accuracy_max = total_correct_max / test_size
        accuracy_top3 = total_correct_top3 / test_size
        print(f"Eval Loss: {avg_loss:.4f}")
        print(f"Eval Accuracy: top1={accuracy_max*100:.2f}%, top3={accuracy_top3*100:.2f}%")
    
    def save(self, path:str):
        torch.save(self.model.state_dict(), path)
        print("Model saved\n")

###
if __name__ == "__main__":
    print("Training...")
    print("Device: ", device)
    rootPath = "./"
    trainer = Trainer(version="v1base")
    trainer.TRAIN(epochs=100, train_batch=512, init_lr=1e-3)
    if torch.cuda.is_available(): torch.cuda.empty_cache()