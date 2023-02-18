import torch

from model import *

import torchvision
import torchvision.transforms as transforms

from data import *
from score_utils import *


transform = transforms.Compose([
    ELA((299,299)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


testset = TestDataset(transform = transform)
testloader = DataLoader(testset, batch_size=8, shuffle=False, drop_last=False)

model = Xception(num_classes=2).cuda()

model.load_state_dict(torch.load('./checkpoints/xception_28.pt'))

test_avg = AverageMeter()

model.eval()
with torch.no_grad():
    test_acc = AverageMeter()
    with torch.no_grad():

        for data, target in testloader:

            data = data.cuda()
            target = target.cuda()

            pred = model(data)
            print(pred)
            num_data = len(data)
            test_acc.update(Accuracy(pred, target), num_data)
    print(test_acc.avg)