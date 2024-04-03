from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
img_path = r"images/YYYMarshal.jpg"
img = Image.open(img_path)
print(img)

img.show()

""" ToTensor """
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

""" Normalize """
print(img_tensor[0][0][0])

# trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# trans_norm = transforms.Normalize([1, 3, 5], [3, 2, 1])
trans_norm = transforms.Normalize([6, 3, 2], [9, 3, 5])

img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])

# writer.add_image("Normalize", img_norm)
# writer.add_image("Normalize", img_norm, 1)
writer.add_image("Normalize", img_norm, 2)

writer.close()
