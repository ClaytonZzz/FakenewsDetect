import torch
from torch import nn
import torch.nn.functional as F

from transformer_layer import *

"""
模型定义
一个简单的纯transformer模型
""" 

class mymodel(nn.Module):
    def __init__(self):
        
        super(mymodel,self).__init__()
        self.encoder = TransformerEncoder(num_patches=3, input_dim=768, depth=2, heads=8, mlp_dim=1024, dim_head=64)
        # 最后的结果是一个二分类问题，所以最后一层的输出是1
        self.fc = nn.Sequential(
            nn.Linear(768*3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )


    def forward(self,inputs):
        x = self.encoder(inputs)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
if __name__ == '__main__':
    model = mymodel()
    inputs = torch.randn(4, 3, 768)
    output = model(inputs)
    print(output.shape)
