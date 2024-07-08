import torch.nn as nn
import torchvision.models as models

def create_model(config):
    architecture = config.get('model_config.architecture')
    embedding_size = config.get('model_config.embedding_size')
    pretrained = config.get('model_config.pretrained')

    if architecture == 'MobileNetV2':
        base_model = models.mobilenet_v2(pretrained=pretrained)
        feature_extractor = base_model.features
        last_channel = base_model.last_channel
    elif architecture == 'MobileNetV3':
        base_model = models.mobilenet_v3_large(pretrained=pretrained)
        feature_extractor = base_model.features
        last_channel = base_model.classifier[-1].in_features
    elif architecture == 'EfficientNet-Lite':
        base_model = models.efficientnet_b0(pretrained=pretrained)
        feature_extractor = base_model.features
        last_channel = base_model.classifier[-1].in_features
    elif architecture == 'SqueezeNet':
        base_model = models.squeezenet1_1(pretrained=pretrained)
        feature_extractor = base_model.features
        last_channel = 512  # Adjust depending on the actual implementation

    model = TripletNetwork(feature_extractor, last_channel, embedding_size)
    return model

class TripletNetwork(nn.Module):
    def __init__(self, feature_extractor, last_channel, embedding_size):
        super(TripletNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.embedding_layer = nn.Linear(last_channel, embedding_size)

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.embedding_layer(x)
        return x

    def forward(self, anchor, positive, negative):
        return self.forward_once(anchor), self.forward_once(positive), self.forward_once(negative)

