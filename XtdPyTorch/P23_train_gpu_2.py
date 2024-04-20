import torch.optim
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from MyModel import MyModel
from torch.utils.tensorboard import SummaryWriter
import time

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
model = model.to(device)

""" 4. 创建损失函数 """
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

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

writer = SummaryWriter("logs")
start_time = time.time()

for i in range(epoch):
    print(f"------ 第 {i + 1} 轮训练开始 ------")

    # 训练步骤开始
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            # loss, loss.item(): item方法就是tensor取值
            print(f"训练次数：{total_train_step}, Loss = {loss}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()

            # argmax(1), 横向就是，单张图片的各种类别概率，求最大
            # accuracy = (outputs.argmax(1) == torch.tensor(targets)).sum()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"整体测试集上的 Loss: {total_test_loss}")
    print(f"整体测试集上的正确率: {total_accuracy / test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy",
                      total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    torch.save(model, f"models/model {i}")
    # torch.save(model.state_dict(), f"models/model {i}")
    print("模型已保存")

writer.close()
