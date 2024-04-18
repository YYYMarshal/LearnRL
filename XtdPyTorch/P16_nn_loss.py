import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)
# torch.Size([3])
print(inputs.shape)
print(targets.shape)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))
# torch.Size([1, 1, 1, 3])
print(inputs.shape)
print(targets.shape)

loss = L1Loss(reduction="sum")
result = loss(inputs, targets)
# tensor(2.)
print(result)

loss = MSELoss()
result = loss(inputs, targets)
# tensor(1.3333)
print(result)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss = CrossEntropyLoss()
result = loss(x, y)
# tensor(1.1019)
print(result)
