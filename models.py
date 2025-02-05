# 큰 그림을 먼저 그리고 구현하자

# VGG19 pre train load
# VGG19 conv layer 분리
# deep image repre

# 스켈레톤 코드를 짜보자.

import torch
import torch.nn as nn
import torchvision.models as models

class StyleTranfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_dict = {
            'conv1_1': 0,  # style
            'conv2_1': 5,  # style
            'conv3_1': 10,  # style
            'conv4_1': 19,  # style
            'conv5_1': 28,  # style
            'conv4_2': 21,  # content
        }
        self.vgg19_model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.vgg19_features = self.vgg19_model.features
        #ToDo: conv layer 분리
        self.style_layer = [self.conv_dict['conv1_1'], self.conv_dict['conv2_1'], self.conv_dict['conv3_1'], self.conv_dict['conv4_1'], self.conv_dict['conv5_1']]
        self.content_layer = [self.conv_dict['conv4_2']]
        pass

    def forward(self, x, mode:str):
        features = []
        if mode=='style':
            for idx in range(len(self.vgg19_features)):
                x = self.vgg19_features[idx](x)
                if idx in self.style_layer:
                    features.append(x)
        elif mode=='content':
            for idx in range(len(self.vgg19_features)):
                x = self.vgg19_features[idx](x)
                if idx in self.content_layer:
                    features.append(x)

        return features
    

def main():
    model = StyleTranfer()
    data = torch.rand(1,3,224,224)
    result = model(data, mode='style')

    print(result)


if __name__ == "__main__":
    main()