from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
import torchvision
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch


class SequentialModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        output = self.model(x)
        return output


dataset = torchvision.datasets.CIFAR10(
    "dataset", train=False, download=True,
    transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=1)

model = SequentialModel()
loss = CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        imgs, targets = data
        outputs = model(imgs)
        result_loss = loss(outputs, targets)
        # print(result)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss += result_loss
    print(running_loss)
