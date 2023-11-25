# モジュールのインポート
import numpy as np
import pandas as pd
import streamlit as st
import json
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

    st.markdown("#### 原文")
    st.table(df)

# Predictionボタンが押された時の処理
if st.sidebar.button("Predict"):
    
    # dfをdict型に変換
    df_dict = df.to_dict(orient="split")
    print(df_dict)
    
    # 予測の実行
    response = requests.post("http://localhost:8000/predict", json=df_dict) # FastAPIにデータを送信し（.postメソッド）、FastAPIから返ってきたデータをresponseに格納
    result_json = response.json()["result_dataframe"] # FastAPIから返ってきたデータをjson形式に変換
    result_dataframe = pd.read_json(result_json, orient="split") # jsonファイルをデータフレーム形式に変換

    # st.markdown("#### 予測ラベル")
    # st.table(df_pred)

    label_mapping = {
        0:"保留品管理規定",
        1:"受入検査規定",
        2:"品質マニュアル",
        3:"教育・訓練規定",
        4:"文書管理規定",
        5:"是正予防管理規定",
        6:"監査規定",
        7:"製品検査規定",
        8:"識別管理規定",
        9:"品質アセスメント管理規定",
        10:"品質計画文書管理規定",
        11:"変更管理規定",
        12:"工程管理規定",
        13:"検査測定機器管理規定",
        14:"製造設計管理規定",
        15:"設備管理規定",
        16:"設計・開発管理規定",
        17:"購買管理規定",
        18:"顧客関連プロセス管理規定",
    }

    df_pred_textname = result_dataframe.copy()
    df_pred_textname["予測文書"] = df_pred_textname["予測文書"].replace(label_mapping)

    st.markdown("#### 予測文書")
    st.table(df_pred_textname)