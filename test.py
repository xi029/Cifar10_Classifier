import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from models import *
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas import DataFrame

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    绘制混淆矩阵的函数
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def plot_classification_report(report, title='Classification Report'):
    """
    将分类报告绘制为图片
    """
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-5]:
        row = line.split()
        report_data.append({
            'Class': row[0],
            'Precision': float(row[1]),
            'Recall': float(row[2]),
            'F1-Score': float(row[3]),
            'Support': int(row[4])
        })
    report_df = DataFrame(report_data)

    # 创建表格图片
    fig, ax = plt.subplots(figsize=(10, len(report_data) * 0.6))
    sns.heatmap(report_df.iloc[:, 1:].T, annot=report_df.iloc[:, 1:].T, fmt='.2f', cmap='YlGnBu', cbar=False)
    ax.set_xticklabels(report_df['Class'], rotation=45, ha='right')
    ax.set_yticklabels(['Precision', 'Recall', 'F1-Score', 'Support'], rotation=0)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def main():
    # 设置设备为 GPU 或 CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 数据预处理
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 加载测试集
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2)

    # 加载模型
    checkpoint = torch.load('checkpoint/DLANet_pth/best.pth')
    net = DLA()
    state_dict = checkpoint['net']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict)

    net = net.to(device)
    net.eval()  # 设置为评估模式

    # 打印模型参数量
    print("Total parameters:", sum(p.numel() for p in net.parameters()))
    print("Trainable parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))

    # 测试函数
    test_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():  # 不计算梯度
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader, desc="Testing", ncols=100)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs, targets)  # 计算损失

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 保存预测值和真实值用于混淆矩阵和分析报告
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        # 输出结果
        acc = 100. * correct / total
        print(f'Test Loss: {test_loss / len(testloader):.3f}, Test Acc: {acc:.3f}%')

    # 生成混淆矩阵
    classes = testset.classes
    cm = confusion_matrix(all_targets, all_predictions)
    #print("Confusion Matrix:\n", cm)

    # 绘制混淆矩阵
    plot_confusion_matrix(cm, classes=classes, title='Confusion Matrix for CIFAR-10')

    # 输出分类报告并绘制
    report = classification_report(all_targets, all_predictions, target_names=classes)
    #print("\nClassification Report:\n", report)
    plot_classification_report(report, title='Classification Report for CIFAR-10')

if __name__ == '__main__':
    main()
