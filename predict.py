import torch
import torchvision.transforms as transforms
from PIL import Image
from models import *


# 加载最佳模型
def load_model(model_path, device):
    print('=====Loading the trained model======')
    net = DLA()
    net = torch.nn.DataParallel(net)  # 包装为 DataParallel
    net = net.to(device)
    checkpoint = torch.load(model_path, map_location=device)  # 加载模型参数
    net.load_state_dict(checkpoint['net'])  # 加载网络权重
    net = net.module  # 移除 DataParallel 的包装以方便后续使用
    net.eval()  # 设置为评估模式
    print('Model loaded successfully.')
    return net


# 图像预处理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # CIFAR-10的图像尺寸
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 与训练一致的归一化
    ])
    image = Image.open(image_path).convert('RGB')  # 打开图像并确保为RGB模式
    image = transform(image).unsqueeze(0)  # 增加batch维度
    return image


# 进行预测
def predict(net, image, device, classes):
    image = image.to(device)
    with torch.no_grad():
        outputs = net(image)  # 前向传播
        _, predicted = outputs.max(1)  # 获取预测结果
    return classes[predicted.item()]


def main():
    # 配置
    model_path = 'checkpoint/DLANet_pth/best.pth'  # 最佳模型的路径
    image_path = './predict/image.png'  # 输入图像的路径
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classes = ('plane', 'car', 'bird', 'cat', 'deer',  # CIFAR-10分类标签
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 加载模型
    net = load_model(model_path, device)

    # 图像预处理
    image = preprocess_image(image_path)

    # 进行预测
    predicted_class = predict(net, image, device, classes)
    print(f'The predicted class for the image is: {predicted_class}')


if __name__ == '__main__':
    main()
