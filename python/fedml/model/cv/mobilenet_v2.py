import torch
import torch.nn as nn
import torch.nn.functional as F

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])
        
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2_MNIST(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, dropout_rate=0.2):
        super(MobileNetV2_MNIST, self).__init__()
        input_channel = 16
        last_channel = 128

        # [t, c, n, s]
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 2, 2],
            [6, 64, 2, 1],  # keep stride=1 to preserve spatial info
        ]
        
        input_channel = _make_divisible(input_channel * width_mult)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult))

        features = [
            nn.Conv2d(1, input_channel, 3, 1, 1, bias=False),  # 1-channel input
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        ]
        
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel
        
        features.extend([
            nn.Conv2d(input_channel, self.last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True),
        ])
        
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.last_channel, num_classes),
        )

        self.apply(_weights_init)

    def forward(self, x):
        x = self.features(x)                   # [batch, C, H, W]
        x = F.adaptive_avg_pool2d(x, (1, 1))   # [batch, C, 1, 1]
        x = torch.flatten(x, 1)                # [batch, C]
        x = self.classifier(x)                 # [batch, num_classes]
        return x
