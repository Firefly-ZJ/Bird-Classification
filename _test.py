#####     Test     #####
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
from tqdm import tqdm

from _BirdNet import getModel
from _train import CEloss_smooth

device = torch.device("cuda" if torch.cuda.is_available() else
                      "xpu" if torch.xpu.is_available() else "cpu")

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def calculate_f1(labels, preds, num_classes:int, eps:float=1e-6):
    """Calculate macro F1 score (not averaged)"""
    # 计算TruePositive、FalsePositive、FalseNegative
    tp, fp, fn = np.zeros(num_classes), np.zeros(num_classes), np.zeros(num_classes)
    for i in range(num_classes):
        tp[i] = np.sum((labels == i) & (preds == i))
        fp[i] = np.sum((labels != i) & (preds == i))
        fn[i] = np.sum((labels == i) & (preds != i))
    # 计算精确率和召回率、F1分数
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1_scores = 2 * (precision * recall) / (precision + recall + eps)
    return f1_scores

### ----- 测试 -----
@torch.no_grad()
def TEST(version:str, load_weight:bool|str, test_batch:int=128):
    test_dataset = ImageFolder(rootPath+"birdData/val", transform=test_transform)
    test_loader = DataLoader(test_dataset, test_batch, shuffle=True, num_workers=4)
    print(f"Test Size: {len(test_dataset)},  Batch Num: {len(test_loader)}")

    model = getModel(version, load_weight)
    model.to(device).eval()
    criterion = CEloss_smooth(model.getClassNum(), smoothing=0.1) # 与训练一致的损失函数

    total_loss = 0.0
    total_correct_max, total_correct_top3 = 0, 0
    all_preds, all_labels = [], []
    
    with tqdm(total=len(test_loader), desc="Testing") as pbar:
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
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
            
            # 收集预测值和真实标签
            all_preds.extend(preds_max.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.update(1)

    avg_loss = total_loss / len(test_dataset)
    accuracy_max = total_correct_max / len(test_dataset)
    accuracy_top3 = total_correct_top3 / len(test_dataset)
    species_num = len(test_dataset.classes)
    macro_f1 = calculate_f1(np.array(all_labels), np.array(all_preds), species_num)

    print(f"Test completed\n")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: top1={accuracy_max*100:.2f}%, top3={accuracy_top3*100:.2f}%", end=",  ")
    print(f"Macro F1: {np.mean(macro_f1):.4f}")

if __name__ == "__main__":
    print("Testing...")
    print("Device: ", device)
    rootPath = "./"
    #weight_file = rootPath + "trained/model_v1large.pth"
    TEST("v1large", load_weight=True, test_batch=128)
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# ----- V1 base ----- #
# Average Loss: 2.6212
# Accuracy: top1=62.16%, top3=74.87%,  Macro F1: 0.5781

# ----- V1 large ----- #
# Average Loss: 2.5142
# Accuracy: top1=65.11%, top3=77.08%,  Macro F1: 0.6129