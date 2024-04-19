import torch
import torchvision


def model_save():
    vgg16 = torchvision.models.vgg16(pretrained=False)
    # 需要提前手动新建一个 models 文件夹
    # 保存了 模型结构 + 模型参数
    torch.save(vgg16, "models/vgg16_method1.pth")

    # 保存了 模型参数 （官方推荐）
    torch.save(vgg16.state_dict(), "models/vgg16_method2.pth")


def model_load():
    # model = torch.load("models/vgg16_method1.pth")
    # print(model)

    model = torch.load("models/vgg16_method2.pth")
    vgg16 = torchvision.models.vgg16(pretrained=False)
    vgg16.load_state_dict(model)
    print(vgg16)


# model_save()
model_load()
