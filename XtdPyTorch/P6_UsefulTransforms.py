from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
# 这里需要放入一个宽屏（非正方形即可）的图片才能更好的观看后面的效果
img_path = r"images/screen.png"
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

""" Resize """
print("=== Resize ===")
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
print(img_resize)
img_resize.show()

img_resize_tensor = trans_totensor(img_resize)
writer.add_image("Resize", img_resize_tensor, 0)

""" Compose Resize 2 """
print("=== Compose Resize 2 ===")
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_tensor_2 = trans_compose(img)
writer.add_image("Resize", img_resize_tensor_2, 1)

""" RandomCrop """
print("=== RandomCrop ===")
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

trans_random = transforms.RandomCrop((500,1000))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCropHW", img_crop, i)

writer.close()
