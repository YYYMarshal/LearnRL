from torch.utils.data import Dataset
from PIL import Image


class MyData(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        pass


# help(Dataset)
img_path = r"C:\Users\YYYXB\Desktop\hymenoptera_data\train\ants\0013035.jpg"
print(img_path)
img = Image.open(img_path)
print(img.size)
img.show()