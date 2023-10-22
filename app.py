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
    
    # jsonファイルに変換
    df_json = df.to_json(force_ascii=False,orient="split")
    
    # 予測の実行
    response = requests.post("http://localhost:8000/predict", json={"dataframe_request": {"data": df_json}})
    result_json = response.json()["result_dataframe"]
    result_dataframe = pd.DataFrame.from_dict(json.loads(result_json), orient="split")

    # st.markdown("#### 予測ラベル")
    # st.table(df_pred)

    label_mapping = {
        0:"保留品管理基準書",
        1:"受入検査基準書",
        2:"品質マニュアル",
        3:"教育・訓練基準書",
        4:"文書管理基準書",
        5:"是正予防管理基準書",
        6:"監査基準書",
        7:"製品検査基準書",
        8:"識別管理基準書",
        9:"品質アセスメント管理基準書",
        10:"品質計画文書管理基準書",
        11:"変更管理基準書",
        12:"工程管理基準書",
        13:"検査測定機器管理基準書",
        14:"製造設計管理基準書",
        15:"設備管理基準書",
        16:"設計・開発管理基準書",
        17:"購買管理基準書",
        18:"顧客関連プロセス管理基準書",
    }

    df_pred_textname = result_dataframe.copy()
    df_pred_textname["予測文書"] = df_pred_textname["予測文書"].replace(label_mapping)

    st.markdown("#### 予測文書")
    st.table(df_pred_textname)