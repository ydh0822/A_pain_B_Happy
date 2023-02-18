import torch
import torch.nn as nn

class SeperableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeperableConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.pointwise(x)
        return x
    
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        # if out_channels != in_channels, skip connection with shape equalization needed
        if out_channels != in_channels or strides != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        rep = []
        for i in range(reps):
            # last block increases its channel in second conv layer
            if grow_first:
                inc = in_channels if (i == 0) else out_channels
                outc = out_channels
            else:
                inc = in_channels
                outc = in_channels if (i < (reps-1)) else out_channels
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeperableConv2d(inc, outc, 3, stride=1, padding=1))
            rep.append(nn.BatchNorm2d(outc))

        # first block doesn't start with relu
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace = False)

        # if strides is 2, MaxPool layer added
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        
        x += skip
        return x
    
class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x
    
class Xception(nn.Module):

    def __init__(self, num_classes=1000, in_channels=3, drop_rate = 0., global_pool = 'avg'):
        super(Xception, self).__init__()

        self.drop_rate = drop_rate
        self.global_pool = global_pool
        self.num_classes = num_classes
        self.num_features = 1048

        self.entry_flow = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),

            Block(64, 128, 2, 2, start_with_relu=False),
            Block(128, 256, 2, 2),
            Block(256, 728, 2, 2)
        )

        self.middle_flow = nn.Sequential(
            Block(728, 728, 3, 1),
            Block(728, 728, 3, 1),
            Block(728, 728, 3, 1),
            Block(728, 728, 3, 1),
            Block(728, 728, 3, 1),
            Block(728, 728, 3, 1),
            Block(728, 728, 3, 1),
            Block(728, 728, 3, 1)
        )

        self.exit_flow = nn.Sequential(
            Block(728, 1024, 2, 2, grow_first=False),

            SeperableConv2d(1024, 1536, 3, 1, 1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),

            SeperableConv2d(1536, self.num_features, 3, 1, 1),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1,1)),
            Flatten(),
            nn.Linear(self.num_features, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        return x