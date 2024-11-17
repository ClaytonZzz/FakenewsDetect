import os
os.environ['HF_HUB_OFFLINE'] = "1"
import sys
import requests
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from model_base import mymodel
from config_fn import text_model_dir,weight_path
# 以上为依赖包引入部分, 请根据实际情况引入
# 引入的包需要安装的, 请在requirements.txt里列明, 最好请列明版本



# 以下为逻辑函数, main函数的入参和最终的结果输出不可修改
def main(to_pred_dir, result_save_path):
    # run_py = os.path.abspath(__file__)
    # model_dir = os.path.dirname(run_py)

    to_pred_dir = os.path.abspath(to_pred_dir)
    testa_csv_path = os.path.join(to_pred_dir, "testa_x", "testa_x.csv")
    # testa_html_dir = os.path.join(to_pred_dir, "testa_x", "html")
    # testa_image_dir = os.path.join(to_pred_dir, "testa_x", "image")

    #=-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==
    # 以下区域为预测逻辑代码, 下面的仅为示例
    # 请选手根据实际模型预测情况修改
    
    testa = pd.read_csv(testa_csv_path)
    id = testa["id"].tolist()
    official_account_name = testa["Ofiicial Account Name"].tolist()
    title = testa["Title"].tolist()
    report_content = testa["Report Content"].tolist()

    # text embeddings
    # print("text embeddings")
    m = SentenceTransformer("shibing624/text2vec-base-chinese",cache_folder=text_model_dir)
    AccountNameEmbeddings = m.encode(official_account_name)
    TitleEmbeddings = m.encode(title)
    ReportContentEmbeddings = m.encode(report_content)

    # list to numpy
    print("list to numpy")
    AccountNameEmbeddings = np.array(AccountNameEmbeddings)
    TitleEmbeddings = np.array(TitleEmbeddings)
    ReportContentEmbeddings = np.array(ReportContentEmbeddings)
    id = np.array(id)

    feature_np = np.concatenate([AccountNameEmbeddings, TitleEmbeddings, ReportContentEmbeddings], axis=1).reshape(-1, 3, 768)
    #=-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==
    # model
    print("model")
    model = mymodel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model = model.to(device)
    # numpy to tensor
    print("numpy to tensor")
    feature_tensor = torch.tensor(feature_np, dtype=torch.float32)
    feature_tensor = feature_tensor.to(device)
    # load checkpoint
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint,strict=True)
    # eval()函数的作用是将模型设置为评估模式，只对BatchNorm和Dropout有影响
    model.eval()
    with torch.no_grad():
        outputs = model(feature_tensor)

        outputs_np = outputs.cpu().numpy()
        # shape = (batch_size, 1) -> (batch_size,)
        outputs_np = outputs_np.flatten()

    # test = testa[["id", "label"]]
    #=-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==
    # 将numpy 结果写入
    test = pd.DataFrame()
    test["id"] = id
    test["label"] = outputs_np
    test["label"] = test["label"].apply(lambda x: 1 if x > 0.5 else 0)
    print(test["label"].value_counts())
    # 结果输出到result_save_path
    test.to_csv(result_save_path, index=None)

if __name__ == "__main__":
    # 以下代码请勿修改, 若因此造成的提交失败由选手自负
    to_pred_dir = sys.argv[1]  # 所需预测的文件夹路径
    result_save_path = sys.argv[2]  # 预测结果保存文件路径，已指定格式为csv
    main(to_pred_dir, result_save_path)
