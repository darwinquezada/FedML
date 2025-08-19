import torch
import torch.nn as nn
import torch.nn.functional as F

def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, dropout_rate=0.5):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Calculate the input features for classifier dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_output = features(dummy_input)
            dummy_output = self.avgpool(dummy_output)
            classifier_input_size = dummy_output.view(1, -1).size(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, num_classes),
        )
        self.apply(_weights_init)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# VGG configurations
cfgs = {
    'VGG6': [64, 'M', 128, 'M', 256, 'M', 512, 'M'],  # Very light version
    'VGG8': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg6(num_classes=1000, batch_norm=False, pretrained=False, path=None, **kwargs):
    """
    VGG-6 model (very light version)
    Architecture: Conv64 → MaxPool → Conv128 → MaxPool → Conv256 → MaxPool → Conv512 → MaxPool
    """
    model = VGG(make_layers(cfgs['VGG6'], batch_norm=batch_norm), num_classes=num_classes, **kwargs)
    
    if pretrained:
        _load_pretrained_weights(model, path)
    
    return model

def vgg8(num_classes=1000, batch_norm=False, pretrained=False, path=None, **kwargs):
    """VGG-8 model (custom smaller version)"""
    model = VGG(make_layers(cfgs['VGG8'], batch_norm=batch_norm), num_classes=num_classes, **kwargs)
    if pretrained:
        _load_pretrained_weights(model, path)
    return model

def vgg11(num_classes=1000, batch_norm=False, pretrained=False, path=None, **kwargs):
    """VGG-11 model"""
    model = VGG(make_layers(cfgs['VGG11'], batch_norm=batch_norm), num_classes=num_classes, **kwargs)
    if pretrained:
        _load_pretrained_weights(model, path)
    return model

def vgg13(num_classes=1000, batch_norm=False, pretrained=False, path=None, **kwargs):
    """VGG-13 model"""
    model = VGG(make_layers(cfgs['VGG13'], batch_norm=batch_norm), num_classes=num_classes, **kwargs)
    if pretrained:
        _load_pretrained_weights(model, path)
    return model

def vgg16(num_classes=1000, batch_norm=False, pretrained=False, path=None, **kwargs):
    """VGG-16 model"""
    model = VGG(make_layers(cfgs['VGG16'], batch_norm=batch_norm), num_classes=num_classes, **kwargs)
    if pretrained:
        _load_pretrained_weights(model, path)
    return model

def vgg19(num_classes=1000, batch_norm=False, pretrained=False, path=None, **kwargs):
    """VGG-19 model"""
    model = VGG(make_layers(cfgs['VGG19'], batch_norm=batch_norm), num_classes=num_classes, **kwargs)
    if pretrained:
        _load_pretrained_weights(model, path)
    return model

def _load_pretrained_weights(model, path):
    """Helper function to load pretrained weights"""
    if path is None:
        raise ValueError("Path to pre-trained weights must be provided when pretrained=True.")
    
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)