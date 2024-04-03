from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

"""
1. Transforms 的使用
2. Tensor 数据类型
"""

"""
绝对路径：S:\YYYXUEBING\Project\PyCharm\EnvRL\XtdPyTorch\hymenoptera_data\train\ants_image\0013035.jpg
相对路径：XtdPyTorch/hymenoptera_data/train/ants_image/0013035.jpg
"""
# 这里因为 hymenoptera_data 文件夹和该代码文件同级，所以下面的路径应该这样书写
img_path = r"hymenoptera_data/train/ants_image/0013035.jpg"
img = Image.open(img_path)
print(img)

writer = SummaryWriter("logs")

# tensor_trans = transforms.ToTensor()
# tensor_img = tensor_trans(img)
tensor_img = transforms.ToTensor()(img)
print(tensor_img)

writer.add_image("Tensor_Image", tensor_img)
writer.close()
