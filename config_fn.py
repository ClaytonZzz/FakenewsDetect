# 存放 目录文件 配置

import os

# 本地路径
project_root = os.path.dirname(os.path.abspath(__file__))

# html输入和输出目录路径
html_input_dir = os.path.join(project_root, 'dataset/train/html')
html_output_dir = os.path.join(project_root, 'dataset/train/html_text')

# text embedding模型路径
text_model_dir = os.path.join(project_root, 'text_model')

# 数据集csv文件路径
train_csv_dir = os.path.join(project_root, 'dataset/train/train.csv')

# 数据集npz文件路径
train_npz_dir = os.path.join(project_root, 'dataset/train/train.npz')

# 模型权重保存路径
weight_path = os.path.join(project_root, 'weights/init.pt')

# 随机种子
seed_value = 42