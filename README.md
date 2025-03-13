以下是你的 `README.md` 文件内容，你可以直接复制到你的 GitHub 项目中：

------

# 基于Streamlit的CIFAR-10分类器

## 1. 项目简介

本项目是一个基于深度学习的图像分类 Web 应用，使用 CIFAR-10 数据集训练了一个 DLA 模型，并通过 Streamlit 部署。用户可以通过 Web 界面上传图片，并实时查看分类结果。

------

## 2. 安装与部署

### 2.1 系统要求

- **操作系统**：Windows 11 / Ubuntu 18.04 及以上
- **Python 版本**：Python 3.10
- **显卡（可选）**：NVIDIA GeForce RTX 4060（加速训练和推理）

### 2.2 安装步骤

#### 1️⃣ 创建虚拟环境（基于 Conda）

```bash
conda create -n cifar10 python=3.10 -y
conda activate cifar10
```

#### 2️⃣ 安装依赖包

```bash
pip install -r requirements.txt
```

📌 **注意**：

- 若有其他需求（如缺少某些依赖），请手动更新 `requirements.txt` 并重新安装依赖。

- 若需 GPU 加速，请确保安装正确版本的 

  CUDA

   和 

  cuDNN

  ，并验证 PyTorch 的 GPU 支持：

  ```python
  import torch
  print(torch.cuda.is_available())  # 输出 True 表示支持 GPU
  ```

#### 3️⃣ 下载模型文件

将训练好的模型文件 `bast.pth` 放置在 `checkpoint/DLANet/` 目录下（若文件夹不存在，请手动创建）：

```bash
mkdir -p checkpoint/DLANet
# 手动复制 best.pth 到该目录
```

------

## 3. 运行程序

### 1️⃣ 激活虚拟环境

```bash
conda activate cifar10
```

### 2️⃣ 启动 Web 应用

```bash
streamlit run app.py
```

### 3️⃣ 访问 Web 界面

浏览器会自动打开 Streamlit 界面，若未自动打开，可在浏览器地址栏输入以下地址访问：

```
http://localhost:8501
```

------

## 4. 应用功能与使用说明

### 🎨 主界面模块

- **📂 文件上传**：支持拖拽或点击上传 `.png`, `.jpg`, `.jpeg` 格式的图片。
- **📊 结果展示**：实时显示分类结果及概率分布。

### 🛠 使用步骤

1. 运行 `streamlit run app.py` 启动应用。
2. 点击上传按钮，选择待分类的图片。
3. 模型加载后自动进行分类，并在界面显示预测类别及概率分布。

------

## 5. 依赖库（`requirements.txt`）

```plaintext
streamlit==1.41.1
numpy==1.26.4
torch==2.2.2
torchvision==0.17.2
scikit-learn==1.5.1
seaborn==0.13.2
matplotlib==3.8.3
pandas==2.2.1
tqdm==4.65.2
```

------

🚀 **欢迎 Star & Fork 支持本项目！** 🎉