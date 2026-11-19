# 实验五：基于 LeNet-5 的 MNIST 手写数字识别与特征可视化

#### 202410315058-姚钧耀
文件说明

本实验的代码在 `lenet.py` 文件中，实验所需数据集 MNIST 会在运行时自动下载到 `data` 文件夹，实验的可视化结果通过 matplotlib 直接展示。

## 一、实验目的

基于 LeNet-5 卷积神经网络实现 MNIST 手写数字数据集的分类识别，掌握卷积神经网络的基本结构（卷积层、池化层、全连接层）与前向传播流程，理解特征图可视化方法及其对模型可解释性的意义。

## 二、实验内容

### 1. 导入必要库

- `torch`：PyTorch 核心库
- `torch.nn`：神经网络模块
- `torch.nn.functional`：函数式接口（激活函数、池化、损失函数）
- `torch.optim`：优化器
- `torchvision`：数据集与图像变换
- `matplotlib.pyplot`：可视化库

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
2. 超参数设置与数据加载
设置批大小为512，并自动检测GPU是否可用。使用 torchvision.datasets.MNIST 加载训练集和测试集，通过 ToTensor() 将图像转换为张量，并用 DataLoader 封装为可迭代批次。

python
batch_size = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainloader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=batch_size, shuffle=True)
3. LeNet-5 网络结构定义
LeNet-5由两个卷积层、两个平均池化层和三个全连接层组成。conv1 输入1通道、输出6通道、卷积核5×5、padding=2；conv2 输入6通道、输出16通道、卷积核5×5；fc1 输入400、输出120；fc2 输入120、输出84；clf 输入84、输出10。激活函数统一使用Sigmoid，池化统一使用2×2、步长2的平均池化。

python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # 输入1, 输出6, 5x5, padding=2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)             # 输入6, 输出16, 5x5
        self.fc1 = nn.Linear(5 * 5 * 16, 120)                    # 400 -> 120
        self.fc2 = nn.Linear(120, 84)                            # 120 -> 84
        self.clf = nn.Linear(84, 10)                             # 84 -> 10

    def forward(self, x):
        # conv1 -> sigmoid -> avg_pool
        x = F.sigmoid(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        # conv2 -> sigmoid -> avg_pool
        x = F.sigmoid(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        # 展平，从第1维开始
        x = x.view(x.size(0), -1)

        # 全连接层1 -> sigmoid
        x = F.sigmoid(self.fc1(x))
        # 全连接层2 -> sigmoid
        x = F.sigmoid(self.fc2(x))
        # 分类层
        x = self.clf(x)
        return x
维度变化说明：输入 1×28×28 → conv1（padding=2保持尺寸）→ 6×28×28 → 平均池化 → 6×14×14 → conv2（无padding，5×5）→ 16×10×10 → 平均池化 → 16×5×5 → 展平 → 400 → 120 → 84 → 10。

4. 模型训练
模型迁移到 device，使用Adam优化器（学习率1e-2），损失函数为交叉熵。训练30个epoch，执行标准的“前向传播→计算损失→反向传播→参数更新”流程。

python
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

epochs = 30
accs, losses = [], []
for epoch in range(epochs):
    for batch_idx, (x, y) in enumerate(trainloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
5. 模型测试与评估
在测试集上使用 torch.no_grad() 关闭梯度计算，累计测试损失并以 pred.max(dim=1) 取预测类别，统计正确数量，计算准确率和平均测试损失。

python
correct = 0
testloss = 0
with torch.no_grad():
    for batch_idx, (x, y) in enumerate(testloader):
        x, y = x.to(device), y.to(device)
        out = model(x)
        testloss += F.cross_entropy(out, y).item()
        pred = out.max(dim=1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()

acc = correct / len(testloader.dataset)
testloss = testloss / (batch_idx + 1)
accs.append(acc)
losses.append(testloss)
print('epoch:{}, loss:{:.4f}, acc:{:.4f}'.format(epoch, testloss, acc))
6. 特征图可视化
取测试集最后一个批次的输入 x，分别提取 conv1 和 conv2 经Sigmoid激活后的特征图。对通道维度求和（sum(0)）将多通道特征图压缩为单通道灰度图。绘制三行：原始输入图像、conv1特征图、conv2特征图（先经2×2平均池化）。

python
feature1 = F.sigmoid(model.conv1(x))
# feature1 = F.avg_pool2d(feature1, kernel_size=2, stride=2)
feature2 = F.sigmoid(model.conv2(feature1))
feature2 = F.avg_pool2d(feature2, kernel_size=2, stride=2)

n = 5
img = x.detach().cpu().numpy()[:n]
feature_map1 = feature1.detach().cpu().numpy()[:n]
feature_map2 = feature2.detach().cpu().numpy()[:n]

fig, ax = plt.subplots(3, n, figsize=(10, 10))
for i in range(n):
    ax[0, i].imshow(img[i].sum(0), cmap='gray')
    ax[1, i].imshow(feature_map1[i].sum(0), cmap='gray')
    ax[2, i].imshow(feature_map2[i].sum(0), cmap='gray')
plt.show()
```
## 三、实验结果与分析

### 1. 训练与测试性能分析

在30个epoch的训练中，模型使用Adam优化器快速收敛。由于MNIST任务相对简单且LeNet-5结构适中，30个epoch后测试准确率可稳定在98%以上。

测试损失随epoch增加持续下降并趋于平稳，说明模型未出现明显过拟合，训练稳定且泛化能力良好。

### 2. 特征图可视化分析

- **第一行（原始输入）**：显示原始28×28手写数字，笔画清晰。
- **第二行（conv1特征图）**：保留了数字的整体轮廓与笔画走向，体现了卷积层对局部边缘和纹理的初步提取能力。
- **第三行（conv2特征图）**：分辨率显著降低，但响应集中在数字的关键笔画区域，说明网络已从低级边缘特征逐步抽象为更具语义的局部结构特征。

对比可见，随着网络加深，特征图的空间分辨率下降、抽象程度提高，直观反映了卷积神经网络“由浅入深、由局部到全局”的特征提取机制。

### 3. 影响因素分析

- **网络结构**：padding=2使conv1输出与输入同尺寸，保留更多空间信息；平均池化适合MNIST这类笔画连续的图像。
- **激活函数**：Sigmoid在浅层网络中能有效工作。
- **超参数**：`batch_size=512` 较大，单次更新梯度稳定；`lr=1e-2` 偏大，若训练出现震荡可适当降低。

## 四、实验总结与心得

### 1. 技术收获

- **CNN结构理解**：通过手动补全LeNet-5各层参数，理解了卷积层通道数变化、池化层降维、全连接层展平衔接的完整流程。
- **PyTorch实践**：掌握了 `nn.Module` 子类的定义方式、`forward` 中函数式API与 `nn` 模块的配合使用，以及标准的训练循环。
- **特征可视化**：学会了通过 `detach().cpu().numpy()` 将GPU张量转为数组，利用 `sum(0)` 对通道求和展示特征图。

### 2. 实践体会

- **代码细节的重要性**：补全代码时需仔细核对维度与参数，避免机械抄写导致语法或逻辑错误。
- **维度推导能力**：LeNet-5的 `fc1` 输入维度由输入尺寸和卷积池化操作共同决定，推导过程锻炼了对CNN空间维度的敏感度。
- **可视化辅助理解**：特征图可视化让模型变得可观察，帮助建立了对卷积层级特征的直观认知，是调试和解释CNN的重要工具。
