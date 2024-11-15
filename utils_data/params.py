import torch
import torch.nn as nn
from lib.Network_Res2Net_GRA_NCD import Network
# 定义您的深度学习模型
model = Network()
net = Network(imagenet_pretrained=False)
net.eval()
# 计算模型的总参数量
total_params = sum(p.numel() for p in net.parameters())
print(f"总参数量: {total_params}")