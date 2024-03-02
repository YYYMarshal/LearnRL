from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np


def convert_image(img_path):
    img = Image.open(img_path)
    print(type(img))
    img_array = np.array(img)
    print(type(img_array))
    print(img_array.shape)
    return img_array


def fun_01():
    writer = SummaryWriter("logs")
    # img_path = r"hymenoptera_data/train/ants_image/0013035.jpg"
    img_path = r"hymenoptera_data/train/bees_image/16838648_415acd9e3f.jpg"
    img_array = convert_image(img_path)
    writer.add_image("test", img_array, global_step=2, dataformats="HWC")
    for i in range(100):
        writer.add_scalar("y = x^3", i * i * i, i)
    writer.close()


fun_01()
