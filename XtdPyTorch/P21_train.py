import torch.optim
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from MyModel import MyModel

""" 1. 准备数据集 """
train_data = torchvision.datasets.CIFAR10(
    "dataset", train=True, download=True,
    transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(
    "dataset", train=False, download=True,
    transform=torchvision.transforms.ToTensor())

train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度：{train_data_size}")
print(f"测试数据集的长度：{test_data_size}")

""" 2. 利用 DataLoader 加载数据集 """
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

""" 3. 搭建神经网络 """
model = MyModel()

""" 4. 创建损失函数 """
loss_fn = nn.CrossEntropyLoss()

""" 5. 优化器 """
# 1e-2 = 1 * (10)^(-2) = 0.01
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

""" 6. 设置训练网络的一些参数 """
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

for i in range(epoch):
    print(f"------ 第 {i + 1} 轮训练开始 ------")

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        # loss, loss.item(): item方法就是tensor取值
        print(f"训练次数：{total_train_step}, Loss = {loss}")
