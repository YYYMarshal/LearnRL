import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu1 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        # output = self.relu1(x)
        output = self.sigmoid1(x)
        return output


def fun_relu():
    x = torch.Tensor([[1, -0.5],
                      [-1, 3]])
    x_input = torch.reshape(x, (-1, 1, 2, 2))
    print(x.shape)
    print(x_input.shape)

    model = MyModel()
    output = model(x_input)
    print(output.shape)
    print(output)


def fun_sigmoid():
    dataset = torchvision.datasets.CIFAR10(
        "dataset", train=False, download=True,
        transform=torchvision.transforms.ToTensor()
    )
    dataloader = DataLoader(dataset, batch_size=64)
    model = MyModel()
    writer = SummaryWriter("logs")
    for i, data in enumerate(dataloader):
        imgs, targets = data
        writer.add_images("input", imgs, i)
        output = model(imgs)
        writer.add_images("output", output, i)
    writer.close()


# fun_relu()
fun_sigmoid()
