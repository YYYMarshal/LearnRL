import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        y = x + 1
        return y


def main():
    model = MyModel()
    x = torch.tensor(1.0)
    output = model(x)
    print(output)


main()
