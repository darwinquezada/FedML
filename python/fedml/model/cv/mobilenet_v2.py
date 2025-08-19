import torch
import torch.nn as nn
from torchvision import models

class MobileNetV2(nn.Module):
    """MobileNetV2 wrapper that adapts to dynamic input shapes and channels."""

    def __init__(self, input_shape=(3, 32, 32), num_classes=10):
        super(MobileNetV2, self).__init__()

        in_channels = input_shape[0]
        self.base_model = models.mobilenet_v2(pretrained=True)

        # Adapt the first conv layer if input is not 3-channel (e.g. MNIST)
        if in_channels != 3:
            old_conv = self.base_model.features[0][0]
            self.base_model.features[0][0] = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )

        # Compute the output feature size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.base_model.features(dummy)
            feature_dim = out.shape[1]

        # Replace the classifier to match the number of classes
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)