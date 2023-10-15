# モジュールのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.functional import accuracy
import torchsummary
from torchsummary import summary
from pytorch_lightning.loggers import CSVLogger
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# モデルの構造の定義
class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1718, 200)
        self.fc2 = nn.Linear(200, 19)
    
    def forward(self, x):
        h = self.fc1(x)
        h = F.relu(h)
        h = self.fc2(h)
        return h

# インスタンス化
net = Net().cpu().eval()

# 重みのロード（cpu環境）
net.load_state_dict(torch.load("kansa.pt", map_location=torch.device("cpu")))

# Vectorizerのロード
file_name = "file_name"
vectorizer = None
with open(file_name, 'rb') as f:
    vectorizer = pickle.load(f)

# 以上で、重み・構造・Vectorizerのロードが完了

# 入力データのベクトル化

# 入力データの読み込み
df = pd.read_excel("Kansa_pred.xlsx")

# 特徴量変換（分かち書き → Vectorizer）
import MeCab
mecab = MeCab.Tagger("Owakati")
texts = df["監査項目"].values
wakati_texts = [mecab.parse(text).strip() for text in texts]
bow_vec = vectorizer.fit_transform(wakati_texts).toarray()
bow_tensor = torch.tensor(bow_vec, dtype=torch.float32)

# インスタンス化
app = FastAPI()

# トップページ
@app.get('/')
async def index():
    return {"Kansa": "kansa_prediction"}

