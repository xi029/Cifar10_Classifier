import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from models import *  # 导入模型


# 加载训练好的模型
def load_model(model_path, device):
    net = DLA()  # 替换为你定义的模型
    checkpoint = torch.load(model_path)

    state_dict = checkpoint['net']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # 移除 'module.' 前缀

    # 加载状态字典
    net.load_state_dict(state_dict)

    # 如果在训练时使用了DataParallel，我们可以在这里包装它
    net = torch.nn.DataParallel(net)  # 如果训练时使用了DataParallel，加载时也要使用
    net = net.to(device)
    net.eval()  # 设置为评估模式
    return net


# 预处理图像
def preprocess_image(image):
    # 如果图像不是RGB，转换为RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 将图像调整为CIFAR-10的大小
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    image = transform(image).unsqueeze(0)  # 添加批次维度
    return image


# 预测图像的类别
def predict(model, image, device):
    image = preprocess_image(image).to(device)
    output = model(image)
    _, predicted = output.max(1)
    return predicted.item()


def main():
    st.title("MLCD_liufengyuan")

    # 显示学生信息
    st.subheader("学生姓名：刘丰源")
    st.subheader("学号：22052308")
    st.write("该软件解决了基于CIFAR-10数据集的图像分类问题。")

    # 图像输入的文件上传器
    uploaded_file = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # 使用 Streamlit 的列布局
        col1, col2 = st.columns([2, 1])  # 调整比例，左宽右窄

        with col1:
            # 显示上传的图像
            image = Image.open(uploaded_file)
            st.image(image, caption='上传的图片', use_container_width=True)  # 使用 use_container_width

        with col2:
            st.write("")  # 空行调整间距
            st.markdown("<h3 style='font-size:24px;'>预测结果</h3>", unsafe_allow_html=True)

            # 加载模型
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = load_model("checkpoint/DLANet_pth/best.pth", device)

            # 预测图像
            prediction = predict(model, image, device)
            classes = ('飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车')

            # 显示分类结果
            st.markdown(
                f"<p style='font-size:22px; color:blue;'>预测类别: <strong>{classes[prediction]}</strong></p>",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
