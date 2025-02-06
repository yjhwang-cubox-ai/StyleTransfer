import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
from PIL import Image
from tqdm import tqdm

from models import StyleTranfer
from loss import ContentLoss, StyleLoss

import imageio

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    content_image = pre_processing(content_image).to(DEVICE)

    style_image = Image.open('style.jpg')
    style_image = pre_processing(style_image).to(DEVICE)

    # load model
    model = StyleTranfer().eval().to(DEVICE)
    # Load loss
    content_loss = ContentLoss()
    style_loss = StyleLoss()

    # Hyperparameter
    alpha = nn.Parameter 
    beta = 1
    lr = 0.01

    # optimizer
    x = torch.randn(1,3,512,512).to(DEVICE).requires_grad_(True)

    optimizer = optim.Adam([x], lr=lr)

    # train loop
    steps = 1000
    frame_interval = 1
    frames = []

    with tqdm(range(steps), desc="Training", unit="step") as pbar:
        for step in pbar:
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

            # print(f"loss_c: {loss_c}")
            # print(f"loss_s: {loss_s}")
            # print(f"loss_total: {loss_total}")

            # tqdm 진행바에 손실 값 출력 (소수점 4자리까지)
            pbar.set_postfix({
                "loss_c": f"{loss_c.item():.4f}",
                "loss_s": f"{loss_s.item():.4f}",
                "loss_total": f"{loss_total.item():.4f}"
            })

            if step % frame_interval == 0:
                frame = post_processing(x)
                frames.append(np.array(frame))
    
    # 학습 완료 후, imageio를 사용하여 GIF로 저장 (fps는 원하는 속도로 조절)
    imageio.mimsave('training_progress.gif', frames, fps=5)


if __name__ == "__main__":
    train()