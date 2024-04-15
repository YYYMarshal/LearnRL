import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(
    "dataset", train=False,
    transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)


class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=3,
            stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


model = MyModel()
print(model)

writer = SummaryWriter("logs")

for i, data in enumerate(dataloader):
    imgs, targets = data
    output = model(imgs)
    # torch.Size([64, 3, 32, 32]) torch.Size([64, 6, 30, 30])
    print(imgs.shape, output.shape)

    writer.add_images("input", imgs, i)

    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, i)

writer.close()
