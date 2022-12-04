import torch
from torch import nn

class Block_conv(torch.nn.Module):
    """
    Helper Class which implements the intermediate Convolutions
    """
    def __init__(self, in_channels, out_channels):
        super(Block_conv, self).__init__()
        self.step = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU())

    def forward(self, X):
        return self.step(X)


class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # enconder
        self.block1 = Block_conv(1, 16)
        self.block2 = Block_conv(16, 32)
        self.block3 = Block_conv(32, 64)
        self.block4 = Block_conv(64, 128)
        self.block5 = Block_conv(128, 256)
        self.down = nn.MaxPool2d(2)
        # decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.block6 = Block_conv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.block7 = Block_conv(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.block8 = Block_conv(64, 32)
        self.up4 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.block9 = Block_conv(32, 16)

        self.out = nn.Conv2d(16, 4, 1)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        x1_use = self.block1(x)
        x1 = self.down(x1_use)
        x2_use = self.block2(x1)
        x2 = self.down(x2_use)
        x3_use = self.block3(x2)
        x3 = self.down(x3_use)
        x4_use = self.block4(x3)
        x4 = self.down(x4_use)
        x5 = self.block5(x4)

        x5 = self.up1(x5)
        x6 = self.block6(torch.cat((x5, x4_use), dim=1))
        x6 = self.up2(x6)
        x7 = self.block7(torch.cat((x6, x3_use), dim=1))
        x7 = self.up3(x7)
        x8 = self.block8(torch.cat((x7, x2_use), dim=1))
        x8 = self.up4(x8)
        x9 = self.block9(torch.cat((x8, x1_use), dim=1))
        out = self.out(x9)

        return out

if __name__=='__main__':
    model = UNet()
    test_input = torch.rand(1, 1, 256, 256)
    out = model(test_input)
    print(out.size())
    print(out.argmax(axis=1).shape)
