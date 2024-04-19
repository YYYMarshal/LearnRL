import torchvision
from torch.nn import Linear

# train_data = torchvision.datasets.ImageNet(
#     "dataset", split="train", transform=torchvision.transforms.ToTensor())

# pretrained=False 改为weight=None  pretrained=True改weights='DEFAULT'
vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

# print(vgg16_false)
# print(vgg16_true)

dataset = torchvision.datasets.CIFAR10(
    "dataset", train=True, download=True,
    transform=torchvision.transforms.ToTensor())

# vgg16_true.add_module("add_linear", Linear(1000, 10))
vgg16_true.classifier.add_module("add_linear", Linear(1000, 10))
# print(vgg16_true)

vgg16_false.classifier[6] = Linear(4096, 10)
print(vgg16_false)
