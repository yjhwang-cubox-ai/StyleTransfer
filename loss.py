# content loss
# vgg19 feature map -> deep image representation
# MSE

# style loss
# gram matrix -> function
# MSE

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor, y:torch.Tensor):
        # x torch.Tensor, shape (b,c,h,w) -> (b,c, h*w)
        # MSE loss
        loss = F.mse_loss(x,y)
        return loss

class StyleLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def gram_matrix(self, x:torch.Tensor):
        """
        x: torch.Tensor shape (b,c,h,w)
        reshape (b,c,h,w) -> (b,c,h*w)
        dim (b, N, M)
        transpose
        matrix mul
        """
        b, c, h, w = x.size()
        features = x.view(b,c,h*w)
        features_T = features.transpose(1,2)
        G = torch.matmul(features, features_T)
        # GPT 가 제시한 코드
        #gram = torch.bmm(features, features.transpose(1, 2)) 

        return G.div(b*c*h*w)

    def forward(self, x, y):
        # gram matrix 로 구현되는 style matrix 를 구하고
        # MSE 를 구한다.
        Gx = self.gram_matrix(x)
        Gy = self.gram_matrix(y)
        loss = F.mse_loss(Gx, Gy)
        return loss