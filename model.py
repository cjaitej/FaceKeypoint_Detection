import torch.nn as nn
import torch

class Detection(nn.Module):  #output shape: Linear--->136
    def __init__(self, in_c, num_out=136):
        super(Detection, self).__init__()
        self.config = [
            (64, 3, 2),
            [3],
            (128, 3, 2),
            [4],
            (256, 3, 2),
            [6],
            (512, 3, 2),
            [3],
        ]
        self.in_channels = in_c
        self.layers = self.__create_layers()
        self.max_pool = nn.MaxPool2d(2)
        self.avg_pool = nn.AvgPool2d(2)
        self.flatten_layer = nn.Flatten()
        self.output_layer = nn.Linear(4608, num_out)
        self.activation = nn.Sigmoid()

    def __create_layers(self):
        layer = nn.ModuleList()
        in_channels = self.in_channels
        for object in self.config:
            if isinstance(object, tuple):
                out_channels, kernel, stride = object
                layer.append(
                    ConvBlock(in_channels,
                              out_channels,
                              kernel_size=kernel,
                              stride=stride)
                    )
                in_channels = out_channels
            elif isinstance(object, list):
                num_repeat = object[0]
                layer.append(
                    ResBlock(in_channels, num_repeat)
                )

        return layer

    def forward(self, x):
        x = self.layers[0](x)
        for layer in self.layers[1:]:
            x = layer(x)
        x = self.avg_pool(x)
        x = self.flatten_layer(x)
        return self.activation(self.output_layer(x))



class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_c,
                              out_channels=out_c,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_c)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, channels, num_repeat):
        super(ResBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_repeat):
            self.layers.append(
                nn.Sequential(
                ConvBlock(channels, channels//2, kernel_size=1),
                ConvBlock(channels//2, channels, kernel_size=3, padding=1)
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


if __name__ == "__main__":
    img = torch.rand(10, 1, 128, 128)
    model = Detection(in_c=1)
    out = model(img)
    print(out.shape)