import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from models import *  # 导入模型
from tqdm import tqdm  # 导入tqdm进度条
from torch.utils.data import random_split


def main():
    # 参数解析，支持自定义学习率和恢复训练
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')  # 设置学习率
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')  # 是否从断点恢复训练
    args = parser.parse_args()

    # 设置设备为GPU或CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # 初始化起始epoch
    best_acc = 0  # 初始化最佳准确率

    # 数据预处理部分
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 归一化
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 归一化
    ])

    # 加载训练集
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    # 划分出验证集（20%）
    train_size = int(0.8 * len(trainset))  # 80%用于训练
    val_size = len(trainset) - train_size  # 20%用于验证
    trainset, valset = random_split(trainset, [train_size, val_size])  # 划分数据集
    # 训练集，批量大小为128，随机打乱顺序
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)
    # 验证集，批量大小为128，保持顺序不变
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=128, shuffle=False, num_workers=2)
    # 加载测试集，并应用测试数据的预处理
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    # 测试集，批量大小为128，保持顺序不变
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2)

    # CIFAR-10分类标签
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 模型构建部分
    print('=======Building model=======')
    net = DLA() #DLANet、ResNet
    net = net.to(device)  # 将模型加载到设备
    if device == 'cuda':
        net = torch.nn.DataParallel(net)  # 使用多GPU并行
        cudnn.benchmark = True  # 提升性能

    # 如果设置了恢复训练，加载断点
    if args.resume:
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'  # 确保断点目录存在
        checkpoint = torch.load('checkpoint/DLANet_pth/best.pth')  # 加载断点文件
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)  # SGD优化器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)  # 余弦退火学习率


    # 定义训练函数
    def train(epoch):
        print(f'\nEpoch: {epoch}')  # 输出当前epoch
        net.train()  # 设置为训练模式
        train_loss = 0
        correct = 0
        total = 0
        # 使用 tqdm 包装 trainloader 来显示进度条
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, desc="Training", ncols=100)):
            inputs, targets = inputs.to(device), targets.to(device)  # 数据迁移到设备
            optimizer.zero_grad()  # 梯度清零
            outputs = net(inputs)  # 前向传播
            loss = criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            train_loss += loss.item()
            _, predicted = outputs.max(1)  # 获取预测值
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(f'Train Loss: {train_loss / len(trainloader):.3f}, Train Acc: {100. * correct / total:.3f}%')

    # 定义验证函数
    def val(epoch, best_acc):
        net.eval()  # 设置为验证模式
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():  # 关闭梯度计算
            # 使用 tqdm 包装 valloader 来显示进度条
            for batch_idx, (inputs, targets) in enumerate(tqdm(valloader, desc="Validating", ncols=100)):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total  # 计算验证准确率
        print(f'Val Loss: {val_loss / len(valloader):.3f}, Val Acc: {acc:.3f}%')

        # 如果验证准确率更高，则保存模型
        if acc > best_acc:
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')  # 创建断点目录
            torch.save(state, 'checkpoint/best.pth')  # 保存模型
            best_acc = acc  # 更新最佳准确率
        return best_acc

    # 主循环：训练和验证
    for epoch in range(start_epoch, start_epoch + 200):
        train(epoch)  # 训练
        best_acc = val(epoch, best_acc)  # 验证并更新最佳准确率
        scheduler.step()  # 更新学习率


if __name__ == '__main__':
    main()




