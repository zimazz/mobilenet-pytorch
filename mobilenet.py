import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileNet(nn.Module):
	def __init__(self, in_channels, n_classes):
		super(MobileNet, self).__init__()

		def conv_bn(in_channels, out_channels, stride):
			return nn.Sequential(
				nn.Conv2d(in_channels, out_channels, 3, stride, 1),
				nn.BatchNorm2d(out_channels),
				nn.ReLU(inplace=True))

		def conv_dw(in_channels, out_channels, stride):
			return nn.Sequential(
				nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels),
				nn.BatchNorm2d(in_channels),
				nn.ReLU(inplace=True),

				nn.Conv2d(in_channels, out_channels, 1, 1, 0),
				nn.BatchNorm2d(out_channels),
				nn.ReLU(inplace=True),)


		self.model = nn.Sequential(
			conv_bn(in_channels, 32, 2),
			conv_dw(32, 64, 1),
			conv_dw(64, 128, 2),
			conv_dw(128, 128, 1),
			conv_dw(128, 256, 2),
			conv_dw(256, 256, 1),
			conv_dw(256, 512, 2),
			conv_dw(512, 512, 1),
			conv_dw(512, 512, 1),
			conv_dw(512, 512, 1),
			conv_dw(512, 512, 1),
			conv_dw(512, 512, 1),
			conv_dw(512, 1024, 2),
			conv_dw(1024, 1024, 1),
			nn.AdaptiveAvgPool2d(1)
			)
		self.fc = nn.Linear(1024, n_classes)

	def forward(self, x):
		x = self.model(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x


model = MobileNet(in_channels=3, n_classes=1000)
x = torch.randn(2, 3, 224, 224)
y = model(x)
print(y.size())