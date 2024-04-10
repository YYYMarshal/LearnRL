import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_tramsforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(
    root="dataset", train=True, transform=dataset_tramsforms, download=True)
test_set = torchvision.datasets.CIFAR10(
    root="dataset", train=False, transform=dataset_tramsforms, download=True)


def fun01():
    print(test_set[0])
    print(test_set.classes)

    img, target = test_set[0]
    print(img)
    print(target)
    print(test_set.classes[target])
    img.show()


# tensorboard --logdir=XtdPyTorch\dataset_transforms --port=6007
def fun02():
    print(test_set[0])
    writer = SummaryWriter("dataset_transforms")
    for i in range(10):
        img, target = test_set[i]
        writer.add_image("test_set", img, i)
    writer.close()


fun02()
