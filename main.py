# モジュールのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

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

# インスタンス化
app = FastAPI()

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
file_name = "params"
vectorizer = None
with open(file_name, 'rb') as f:
    vectorizer = pickle.load(f)

# 以上で、重み・構造・Vectorizerのロードが完了

class DataFrameRequest(BaseModel):  # FastAPI側で受け取るデータの定義
    columns: list[str]
    index: list[int]
    data: list[list]

# トップページ
@app.get('/')
async def index():
    return {"Kansa": "kansa_prediction"}

# POSTが送信された時のと予測値の定義
@app.post("/predict")
async def make_predictions(dataframe_request: DataFrameRequest):
    print(dataframe_request)
    df_json_data = pd.DataFrame(dataframe_request.data, columns=dataframe_request.columns, index=dataframe_request.index)

    
    # 特徴量変換（分かち書き → Vectorizer）
    import MeCab
    mecab = MeCab.Tagger("Owakati")
    texts = df_json_data["監査項目"].values
    wakati_texts = [mecab.parse(text).strip() for text in texts]
    bow_vec = vectorizer.transform(wakati_texts).toarray()
    bow_tensor = torch.tensor(bow_vec, dtype=torch.float32)

    # 全データまとめて予測
    pred = []
    for i in range(bow_tensor.shape[0]):
        x = bow_tensor[i]  # i番目の行を取得
        with torch.no_grad():
            y = net(x.unsqueeze(0))  # バッチサイズ1のテンソルに変換
            y = F.softmax(y, dim=1)
            y = torch.argmax(y)  # 最も高い確率のクラスを取得
            pred.append(y.item())  # 予測値をpredリストに追加。.item()メソッドはTensorをスカラーに変換するメソッド
    print(pred)
    
    df_pred = df_json_data.copy()
    df_pred["予測文書"] = pred

    return {"result_dataframe": df_pred.to_json(orient='split')}