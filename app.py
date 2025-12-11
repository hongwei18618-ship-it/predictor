import streamlit as st
import pandas as pd
import joblib
import ccxt

# === 加载模型 ===
model = joblib.load("eth_model.pkl")

# === 获取 ETH 实时数据 ===
def get_live_eth(limit=100):
    ex = ccxt.binance()
    data = ex.fetch_ohlcv("ETH/USDT", timeframe="1m", limit=limit)
    df = pd.DataFrame(
        data, 
        columns=["time","open","high","low","close","volume"]
    )
    return df

# === 特征构建 ===
def make_features(df):
    df["return"] = df["close"].pct_change()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["vol_chg"] = df["volume"].pct_change()
    df.dropna(inplace=True)
    return df[["return", "ma5", "ma20", "vol_chg"]]

# ============ Streamlit UI ==============
st.title("🚀 ETH 实时涨跌预测模型")

if st.button("获取最新行情 & 预测"):
    df = get_live_eth()
    X = make_features(df)
    
    prob = model.predict_proba(X.iloc[-1:])[0][1]
    pred = "📈 上涨" if prob > 0.5 else "📉 下跌"

    st.subheader("预测结果：")
    st.write(f"**{pred}**（上涨概率：{prob*100:.2f}%）")

    st.line_chart(df["close"])

st.write("---")
st.info("模型由随机森林训练，数据来源：Binance 1m K线")
