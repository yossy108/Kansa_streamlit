# モジュールのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

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
import openpyxl
import requests

st.set_page_config(page_title="Doc Classification", layout="wide")
st.title("csvファイルのアップロード")

uploaded_file = st.sidebar.file_uploader("file of csv", type="csv")

# アップロードしたファイルがある場合の処理内容
if not uploaded_file:
    st.info("csvファイルを選択してください")
    st.stop()
else:
    df = pd.read_csv(uploaded_file, encoding = "shift-jis")

    # st.markdown("#### write")
    # st.write(df)

    st.markdown("#### table")
    st.table(df)

# Predictionボタンが押された時の処理
if st.sidebar.button("Predict"):

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
    params = "params"
    vectorizer = None
    with open(params, 'rb') as f:
        vectorizer = pickle.load(f)

    # 特徴量変換（分かち書き → Vectorizer）
    import MeCab
    mecab = MeCab.Tagger("Owakati")
    texts = df["監査項目"].values
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

    df_pred = df.copy()
    df_pred["予測文書"] = pred

    # st.markdown("#### 予測ラベル")
    # st.table(df_pred)

    label_mapping = {
        0:"文書管理基準書",
        1:"品質マニュアル",
        2:"教育・訓練基準書",
        3:"製品検査基準書",
        4:"受入検査基準書",
        5:"識別管理基準書",
        6:"監査基準書",
        7:"保留品管理基準書",
        8:"是正予防管理基準書",
        9:"電材機材 工程管理基準書",
        10:"電材機材 変更管理基準書",
        11:"電材機材 品質計画文書管理基準書",
        12:"電材機材 検査測定機器管理基準書",
        13:"電材機材 設備管理基準書",
        14:"電材機材 顧客関連プロセス管理基準書",
        15:"電材機材 設計・開発管理基準書",
        16:"電材機材 購買管理基準書",
        17:"電材機材 製造設計管理基準書",
        18:"電材機材 品質アセスメント管理基準書",
    }

    df_pred_textname = df_pred.copy()
    df_pred_textname["予測文書"] = df_pred_textname["予測文書"].replace(label_mapping)

    st.markdown("#### 予測文書")
    st.table(df_pred_textname)