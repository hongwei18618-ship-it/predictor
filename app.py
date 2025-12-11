import streamlit as st
import joblib
import pandas as pd

st.title("ETH 价格预测模型 (10m / 30m / 1h)")
st.write("上传特征 CSV 文件，我将预测未来是否上涨。")

uploaded = st.file_uploader("上传特征 CSV 文件")

if uploaded:
    df = pd.read_csv(uploaded)
    model = joblib.load("model.joblib")
    prob = model.predict_proba(df)[0][1]
    st.write(f"📈 上涨概率：{prob * 100:.2f}%")
