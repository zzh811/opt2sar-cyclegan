import torch.utils.data.distributed
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import os
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
 
classes = ('opt', 'sar')
 
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class TempModel(nn.Module):
#     def __init__(self):
#         super(TempModel,self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, (7, 7))
#     def forward(self, inp):
#         return self.conv1(inp)

# model = TempModel()
# model.load_state_dict(torch.load('resnet.pt'),False)
# model.eval()
model = models.resnet18()
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2) 
model.load_state_dict(torch.load('resnet.pt'),False)
model.eval()
model.to(DEVICE)

dataset_test = datasets.ImageFolder('data/datatest', transform_test)
print(len(dataset_test))
# 对应文件夹的label
 
for index in range(len(dataset_test)):
    item = dataset_test[index]
    img, label = item
    img.unsqueeze_(0)
    data = Variable(img).to(DEVICE)
    output = model(data)
    _, pred = torch.max(output.data, 1)
    #print(pred.data.item())
    print('Image Name:{},predict:{}'.format(dataset_test.imgs[index][0], classes[pred.data.item()]))
    index += 1



# image_PIL = Image.open('data/datatest/test/sar1.png')
# #
# image_tensor = transform_test(image_PIL)
# # 以下语句等效于 image_tensor = torch.unsqueeze(image_tensor, 0)
# image_tensor.unsqueeze_(0)
# # 没有这句话会报错
# image_tensor = image_tensor.to(DEVICE)
 
# out = model(image_tensor)
# pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in out]).to(DEVICE)
# print(classes[pred])



