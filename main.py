import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
import requests
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

app = FastAPI()

# Load model and scaler
model_url = https://huggingface.co/indraaz/back1/resolve/main/model.pkl
scaler_url = https://huggingface.co/indraaz/back1/resolve/main/scaler.pkl


# --- Fetch Live Stock Data (FREE API Source) ---
def get_live_data(symbol):
    url = f"https://stock-nse-india.vercel.app/api/quote-equity?symbol={symbol}"
    try:
        res = requests.get(url).json()
        candles = res['data']['priceHistory'][-60:]   # last 60 candles (1 min)
        df = pd.DataFrame(candles)

        df.rename(columns={
            "closePrice": "close",
            "openPrice": "open",
            "lowPrice": "low",
            "highPrice": "high",
            "volumeTraded": "volume"
        }, inplace=True)

        return df
    except:
        return None


def add_indicators(df):
    df["rsi"] = RSIIndicator(df["close"]).rsi()

    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    df["ema20"] = df["close"].ewm(span=20).mean()

    bb = BollingerBands(df["close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()

    df.dropna(inplace=True)
    return df



def process_and_predict(symbol):
    df = get_live_data(symbol)

    if df is None:
        return {"error": f"Stock {symbol} not found or data missing"}

    df = add_indicators(df)

    # Use last candle for prediction
    latest = df.iloc[-1]

    features = np.array([[
        latest["close"],
        latest["rsi"],
        latest["macd"],
        latest["macd_signal"],
        latest["ema20"],
        latest["bb_high"],
        latest["bb_low"]
    ]])

    scaled = scaler.transform(features)
    pred = model.predict(scaled)[0]

    entry_price = latest["close"]

    if pred == 1:
        return {
            "symbol": symbol,
            "signal": "BUY",
            "entry": entry_price,
            "target": round(entry_price * 1.005, 2),       # +0.5%
            "stop_loss": round(entry_price * 0.995, 2)     # -0.5%
        }
    else:
        return {
            "symbol": symbol,
            "signal": "SELL",
            "entry": entry_price,
            "target": round(entry_price * 0.995, 2),       # -0.5%
            "stop_loss": round(entry_price * 1.005, 2)
        }


@app.get("/")
def home():
    return {"status": "Backend Running Successfully"}


@app.get("/predict/{symbol}")
def predict(symbol: str):
    return process_and_predict(symbol.upper())
