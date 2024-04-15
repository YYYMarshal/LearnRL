import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        output = self.maxpool1(x)
        return output


def fun_01():
    x = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)
    x = torch.reshape(x, (-1, 1, 5, 5))
    print(x.shape)

    model = MyModel()
    output = model(x)
    print(output)


def fun_02():
    dataset = torchvision.datasets.CIFAR10(
        "dataset", train=False,
        transform=torchvision.transforms.ToTensor(), download=True)
    dataloader = DataLoader(dataset, batch_size=64)
    model = MyModel()
    writer = SummaryWriter("logs_maxpool")
    for i, data in enumerate(dataloader):
        imgs, targets = data
        writer.add_images("input", imgs, i)

        output = model(imgs)
        writer.add_images("output", output, i)
    writer.close()


# fun_01()
fun_02()
