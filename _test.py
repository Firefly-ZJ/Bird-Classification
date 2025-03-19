#####     Test     #####
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
from tqdm import tqdm

from _BirdNet import BirdNet

### ----- 测试数据集 -----
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

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

### ----- 测试 -----
def TEST(model_path):
    print("Testing...")
    num_classes = 350
    batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = ImageFolder(rootPath+"birdData/val", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    print(f"Test Size: {len(test_dataset)},  Batch Num: {len(test_loader)}")

    model = BirdNet(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    total_loss = 0.0
    total_correct_max, total_correct_top3 = 0, 0
    criterion = CEloss_smooth(num_classes, smoothing=0.1)  # 与训练一致的损失函数

    ### 测试循环
    with tqdm(total=len(test_loader), desc="Testing") as pbar:
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True) # 异步传输
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(images)
                # 计算损失
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                # 计算准确率，max & top3
                _, preds_max = torch.max(outputs, dim=1)
                total_correct_max += torch.sum(preds_max == labels).item()

                _, preds_top3 = torch.topk(outputs, k=3, dim=1)
                correct_mask = torch.eq(preds_top3, labels.view(-1, 1))
                total_correct_top3 += correct_mask.any(dim=1).sum().item()

                pbar.update(1)

    avg_loss = total_loss / len(test_dataset)
    accuracy_max = total_correct_max / len(test_dataset)
    accuracy_top3 = total_correct_top3 / len(test_dataset)
    
    print(f"Test completed")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: top1={accuracy_max*100:.2f}%, top3={accuracy_top3*100:.2f}%")

if __name__ == "__main__":
    rootPath = "./"
    TEST(rootPath + "trained/" + "model_200.pth")

# 50 epoch
# Average Loss: 4.9194
# Accuracy: top1=17.15%, top3=30.32%

# 100 epoch
# Average Loss: 4.8491
# Accuracy: top1=18.35%, top3=30.00%

# 150 epoch
# Average Loss: 4.8121
# Accuracy: top1=18.19%, top3=30.16%

# 200 epoch
# Average Loss: 4.8085
# Accuracy: top1=18.35%, top3=30.34%