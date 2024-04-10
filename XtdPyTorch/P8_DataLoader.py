import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10(
    "dataset", train=False, transform=torchvision.transforms.ToTensor())
"""
总数 ÷ batach_size = a ··· b
drop_last：设置为 True 时，余数 b 会被舍去，设置为 False 时，则不会舍去。
"""
test_loader = DataLoader(
    dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("DataLoader")

for epoch in range(2):
    for i, data in enumerate(test_loader):
        imgs, targets = data
        """
        当上面的 batch_size=4 时，下面的输出结果为：torch.Size([4, 3, 32, 32])
        代表的是：4张图，3通道（RGB），32 * 32 的像素值
        """
        # print(imgs.shape)
        # print(targets)
        writer.add_images(f"Epoch {epoch}", imgs, i)
writer.close()
