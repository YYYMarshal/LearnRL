import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn


class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(196608, 10)

    def forward(self, x):
        output = self.linear(x)
        return output


def main():
    dataset = torchvision.datasets.CIFAR10(
        "dataset", train=False, download=True,
        transform=torchvision.transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

    model = MyModel()

    for data in dataloader:
        imgs, targets = data

        # torch.Size([64, 3, 32, 32])
        print(imgs.shape)

        # output = torch.reshape(imgs, (1, 1, 1, -1))
        output = torch.flatten(imgs)

        # torch.Size([1, 1, 1, 196608])
        print(output.shape)

        output = model(output)
        print(output.shape)

        print("-----------------")


main()
