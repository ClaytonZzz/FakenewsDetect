import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = "1"
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from config_fn import text_model_dir, train_csv_dir,train_npz_dir

m = SentenceTransformer("shibing624/text2vec-base-chinese",cache_folder=text_model_dir)

train_dataset = pd.read_csv(train_csv_dir)
# print(train_dataset.head())

# text list
print("text list")
AccountName = train_dataset["Ofiicial Account Name"].tolist()
Title = train_dataset["Title"].tolist()
ReportContent = train_dataset["Report Content"].tolist()
# num list
id = train_dataset["id"].tolist()
label = train_dataset["label"].tolist()

# text embeddings
print("text embeddings")
AccountNameEmbeddings = m.encode(AccountName)
TitleEmbeddings = m.encode(Title)
ReportContentEmbeddings = m.encode(ReportContent)

# list to numpy
print("list to numpy")
AccountNameEmbeddings = np.array(AccountNameEmbeddings)
TitleEmbeddings = np.array(TitleEmbeddings)
ReportContentEmbeddings = np.array(ReportContentEmbeddings)

id = np.array(id)
label = np.array(label)

print("save embeddings")
np.savez(train_npz_dir, AccountName=AccountNameEmbeddings, Title=TitleEmbeddings, ReportContent=ReportContentEmbeddings, id = id,label=label)

print("done")





