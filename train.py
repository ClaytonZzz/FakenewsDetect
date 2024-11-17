import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用 GPU 0 和 1  #这一行需要在import torch前面进行导入，这样才是指定卡
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm
import logging
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
from utils import set_random_seed, update_model_ema
from dataset import CustomDataset
from model_base import mymodel
from torch.utils.data import random_split
import numpy as np
import wandb # wandb ,类似tensorboard,记录数据
from config_fn import *
import copy

def train_model(args):

    # -----------------------------------------------------------------------------------------------------
    # 加载 dataset 
    print("load dataset")
    all_dataset = CustomDataset(train_npz_dir)
    # 将数据集划分为训练集和验证集 8:2
    dataset_length = len(all_dataset)
    train_length = int(dataset_length * 0.8)
    val_length = dataset_length - train_length
    train_dataset, val_dataset = random_split(all_dataset, [train_length, val_length])

    trainloader = DataLoader(train_dataset, batch_size=args.batchsize_train, shuffle=True, num_workers=2,prefetch_factor=4, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=args.batchsize_val, shuffle=False, num_workers=2,prefetch_factor=4, pin_memory=True)
    # ---------------------------------------------------------------------------------------------------------
    # 加载模型 
    model = mymodel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #数据并行，将模型分别加载到不同的GPU,然后将批次分到不同的模型上进行训练：
    model = nn.DataParallel(model)
    model = model.to(device)
    # EMA 应对过拟合
    model_ema = copy.deepcopy(model.module)
    model_ema = nn.DataParallel(model_ema)
    model_ema = model_ema.to(device)

    # ---------------------------------------------------------------------------------------------------------
    # loss
    train_loss = []
    val_loss = []

    val_loss_flag = float('inf')
    keep_train = 0
    criterion = nn.BCEWithLogitsLoss()
    # ---------------------------------------------------------------------------------------------------------
    # 优化器 optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    for epoch in tqdm(range(args.epochs)):
        logging.debug('Epoch [{}/{}], Learning Rate: {:.6f}'.format(epoch+1, args.epochs, optimizer.param_groups[0]['lr']))
        print('Epoch [{}/{}], Learning Rate: {:.6f}'.format(epoch+1, args.epochs, optimizer.param_groups[0]['lr']))
        # Log metrics from your script to W&B
        train_metrics  = {"LearningRate":optimizer.param_groups[0]['lr']}
        running_loss = 0.0

        # 设置为训练模式
        model.train()
        for i, (data,labels) in enumerate(trainloader, 0):
            data,labels = data.float().to(device), labels.float().to(device)
            optimizer.zero_grad()

            outputs = model(data)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # 更新ema_model: 应对过拟合
            update_model_ema(model, model_ema, decay=0.99)
            running_loss += loss.item()
        train_loss.append(running_loss / len(trainloader))

        val_pred = []
        val_labels = []

        # 设置为验证模式
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for i, (data,label) in enumerate(valloader, 0):
                data,labels = data.float().to(device), label.float().to(device)
                #这里输出的维度是[batch_size, 1]
                outputs = model_ema(data)

                loss = criterion(outputs, labels)

                #用于直接显示得分情况：
                val_pred.append(outputs.cpu().numpy())
                val_labels.append(labels.cpu().numpy())


                running_loss += loss.item()

            val_pred = np.concatenate(val_pred, axis=0) #按行进行拼接
            val_labels = np.concatenate(val_labels, axis=0)
            # 计算得分
            # shape = (batch_size, 1) -> (batch_size,)
            val_pred = val_pred.flatten()
            val_labels = val_labels.flatten()
            # 计算 f1_score
            f1_score_val = f1_score(val_labels,val_pred > 0.5)
            print("f1_score_val : " + str(f1_score_val))
            val_metrics = {"val/f1_score":f1_score_val}

            val_loss.append(running_loss / len(valloader))
            scheduler.step(val_loss[-1])

        logging.debug(f"Epoch {epoch+1}, Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}")
        print(f"Epoch {epoch+1}, Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}")   
        Loss_metrics = {"train/Loss":train_loss[-1],"val/Loss":val_loss[-1]}
        data_res = {**train_metrics,**val_metrics,**Loss_metrics}
        wandb.log(data_res)

        if val_loss[-1]<val_loss_flag:
            keep_train = 0
            val_loss_flag = val_loss[-1]
            torch.save(model.state_dict(), args.weight_path+ 'init.pt')
        else:
            keep_train+=1
        
        if keep_train > 60:
            break
    # Close  wandb run
    wandb.finish()


if __name__ == "__main__":
    # Start a new run to track this script
    wandb.init(
        # dir="/home/code",
        # Set the project where this run will be logged
        project="NCCCU2024",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10) 这里写好这次试验的备注，比如 "noxi_nopartern"
        name=f"init",
        # Track hyperparameters and run metadata 用这个代替 args = parse_arguments()
        config={
            "save_dir": 'Cross_CEAM',
            "train_data_path": train_npz_dir,
            "weight_path": weight_path,
            "epochs":100,
            "batchsize_train": 128,
            "batchsize_val": 128,
            "learning_rate": 1e-4,
            "seed_value": seed_value,
        }
    )
    args = wandb.config
    set_random_seed(args.seed_value)
    train_model(args)