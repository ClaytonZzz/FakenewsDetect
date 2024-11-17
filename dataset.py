import torch
from tqdm import tqdm  # 导入 tqdm 库
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
from config_fn import train_npz_dir

"""
数据集加载
"""

# 自定义Dataset类，用于数据加载
class CustomDataset(Dataset):
    def __init__(self, data_dir,embedding_dim=768,data_type = 3):
        self.data_dir = data_dir  #数据路径
        self.embedding_dim = embedding_dim
        self.data_type = data_type
        self.labels = [] # 标签
        self.data = []  #数据

        self.process_data()  #数据加载的方法

    # TODO 未来使用html里的文本的时候需要用到
    def htmltext_path(self,file_dir,role_select):
        pass

    def process_data(self):
        # 读取数据
        data_np = np.load(self.data_dir)
        fetaure_np = np.concatenate([data_np['AccountName'], data_np['Title'], data_np['ReportContent']], axis=1)
        fetaure_np = fetaure_np.reshape(-1, self.data_type, self.embedding_dim)
        # 对label_np 增加一个维度
        labels_np = data_np['label'].reshape(-1, 1)

        # numpy to tensor
        self.data = torch.tensor(fetaure_np, dtype=torch.float32)
        self.labels = torch.tensor(labels_np, dtype=torch.float32)


    def __len__(self):
        # 这里不能用len(self.data) 应为这里的data是一个列表，所以应该用len(self.data[0]) 或者 len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

if __name__ == '__main__':
    # Noxi
    #加载pre_dataset:
    train_dataset = CustomDataset(train_npz_dir)
    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1,prefetch_factor=4, pin_memory=True)
    print(len(trainloader))
    for i, (data,label) in enumerate(trainloader):
        print(data.shape)
        print(label.shape)
        break