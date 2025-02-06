import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
from PIL import Image

from models import StyleTranfer
from loss import ContentLoss, StyleLoss

def pre_processing(image: Image.Image) -> torch.Tensor:
    # 이미지 리사이즈
    # 텐서로 변환
    # 이미지 normalizarion 
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    preprocessing: torch.Tensor = T.Compose([
        T.Resize((512,512)),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

    return preprocessing(image).unsqueeze(0)

def post_processing(img_tensor: torch.Tensor) -> Image.Image:
    # shape: b,c,h,w
    image = img_tensor.to('cpu').detach().numpy()
    image = image.squeeze()
    image = image.transpose(1,2,0)
    # denorm
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = image*std + mean
    image = image.clip(0,1)*255
    image = image.astype(np.uint8)
    
    return Image.fromarray(image)

def train():
    # load data
    content_image = Image.open('content.jpg')
    content_image = pre_processing(content_image)

    style_image = Image.open('style.jpg')
    style_image = pre_processing(style_image)

    # load model
    model = StyleTranfer().eval()
    # Load loss
    content_loss = ContentLoss()
    style_loss = StyleLoss()

    # Hyperparameter
    alpha = 1 
    beta = 1
    lr = 0.01

    # optimizer
    x = torch.randn(1,3,512,512, requires_grad=True)
    optimizer = optim.Adam([x], lr=lr)

    # train loop
    steps = 1000
    for step in range(steps):
        x_content_list = model(x, mode='content')
        y_content_list = model(content_image, mode='content')

        x_style_list = model(x, mode='style')
        y_style_list = model(style_image, mode='style')

        loss_c = 0
        loss_s = 0
        loss_total = 0

        for x_content, y_content in zip(x_content_list, y_content_list):
            loss_c += content_loss(x_content, y_content)
        
        for x_style, y_style in zip(x_style_list, y_style_list):
            loss_s += style_loss(x_style, y_style)

        loss_total = alpha*loss_c + beta*loss_s

        #optimizer.step()
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        print(f"loss_c: {loss_c}")
        print(f"loss_s: {loss_s}")
        print(f"loss_total: {loss_total}")

if __name__ == "__main__":
    train()