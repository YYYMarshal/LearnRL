from PIL import Image
import torchvision
import torch

img_path = r"images/dog.jpg"
img = Image.open(img_path)
print(img)
img = img.convert("RGB")

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()])
img = transform(img)
print(img.shape)

model = torch.load("models/model 0")
# model = torch.load("models/model 0", map_location=torch.device("cpu"))
print(model)

img = torch.reshape(img, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(img)
print(output)
print(output.argmax(1))
