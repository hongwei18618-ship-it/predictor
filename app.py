import streamlit as st
import pandas as pd
import joblib
import requests

model = joblib.load("eth_model.pkl")

# === 使用 Coingecko 公共 API（Streamlit Cloud 100% 可访问） ===
def get_live_eth(limit=120):
    url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
    params = {"vs_currency": "usd", "days": "1"}
    data = requests.get(url, params=params).json()

    prices = data["prices"][-limit:]
    df = pd.DataFrame(prices, columns=["time", "close"])
    df["open"] = df["close"].shift(1)
    df["high"] = df["close"].rolling(3).max()
    df["low"] = df["close"].rolling(3).min()
    df["volume"] = 1000   # Coingecko 免费版无 volume，设定常数即可

    df.dropna(inplace=True)
    return df

def make_features(df):
    df["return"] = df["close"].pct_change()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["vol_chg"] = df["volume"].pct_change()
    df.dropna(inplace=True)
    return df[["return", "ma5", "ma20", "vol_chg"]]

st.title("🚀 ETH 实时涨跌预测模型（无需 Binance API）")

if st.button("获取行情 & 预测"):
    df = get_live_eth()
    X = make_features(df)
    prob = model.predict_proba(X.iloc[-1:])[0][1]
    pred = "📈 上涨" if prob > 0.5 else "📉 下跌"

    st.subheader("预测结果：")
    st.write(f"**{pred}**（上涨概率：{prob*100:.2f}%）")

    st.line_chart(df["close"])

st.write("---")
st.info("数据来自 Coingecko（完全免费 & 无 API 限制）。适用于 Streamlit Cloud。")
