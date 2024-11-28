import yfinance as yf
import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime
from utils import *


parser = argparse.ArgumentParser()
# the reference code for stocks
# Chinese Form: 000001.SZ
# US Form: TSLA
parser.add_argument("market", help="USA or Chinese stock market", type=str, default='CN')
parser.add_argument("stock_code", help="The code of selected stock", type=str, default='000001.SZ')
args = parser.parse_args()
CODE = args.stock_code
MARKET = args.market
MODEL = 'LSTM'
HIDDEN_SIZE = 32
NUM_LAYER = 2

if MARKET == "CN":
    # Get the info of stocks by tushare, Chinese A stock
    import tushare as ts
    ts.set_token('8d8723af16218a29d27bb57f93d0e8a642e481c4d0a0bc9b4da7348d')
    pro = ts.pro_api()

    df = pro.daily(ts_code=CODE, start_date='20210601', end_date=datetime.today().strftime("YYYYMMDD"))
    
    df.sort_values(by='trade_date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df['High'] = df['high']
    df['Low'] = df['low']
    df['Open'] = df['open']
    df['Close'] = df['close']
    df['Volume'] = df['vol']
    
    stock_data = df
    stock_data.set_index(pd.to_datetime(df['trade_date']), inplace=True)
    
else:
    # Create a Ticker object

    stock = yf.Ticker(CODE)
    stock_data = stock.history(period='1y')
    start = "2024-01-01"
    if MARKET=='HK':
        start = "2023-01-01"
    stock_data = yf.download(CODE, start=start, end=datetime.today())


from factors_lib import Momentum, Short_Term_Reversion, FHW_Approaching, Oversold_Reverse_Score, ForceIndex, CCG_Score, ILLIQ_Factor


def calculate_vwap(high, low, close, volume):
    # Step 1: Calculate the typical price for each period
    typical_price = (high + low + close) / 3
    
    # Step 2: Calculate the total price-volume product
    price_volume_product = typical_price * volume
    
    # Step 3: VWAP is the price-volume product divided by the volume
    vwap = price_volume_product / volume
    
    return vwap

target = "Close"
stock_data["ClosePrice"] = stock_data["Close"]
stock_data["HighPrice"] = stock_data["High"]


stock_data['Momentum'] = Momentum(df=stock_data, n_months=3, cal_choose=target)
stock_data["Momentum"] = stock_data["Momentum"].fillna(0)
stock_data["STR"] = Short_Term_Reversion(df=stock_data, cal_choose=target, n_days=5)
stock_data["STR"] = stock_data["STR"].fillna(0)

stock_data["VWAP"] = calculate_vwap(high=stock_data["High"], low=stock_data["Low"], close=stock_data["Close"], volume=stock_data["Volume"])
stock_data["FRX"] = ForceIndex(pool_df=stock_data)
stock_data["FHW"] = FHW_Approaching(input_df=stock_data) * -1
stock_data["OSS"] = Oversold_Reverse_Score(df=stock_data)
stock_data["CCG"] = CCG_Score(stock_data)
stock_data["ILLIQ"] = ILLIQ_Factor(stock_data, 9)

seq_length = 14
predict_length = 7
feature_list = ['High', 'Low', 'Open', 'Close', 'Volume', 'OSS', 'CCG', 'Momentum', 'ILLIQ']
saved_price = stock_data['Close'].copy()
saved_label = minmax_scale(stock_data['Close'].copy())



# plot pics
if MARKET == 'CN':
    plt.figure(figsize=(12, 8), facecolor='lightgrey')
else:
    plt.figure(figsize=(12, 12), facecolor='lightgrey')

stock_data['Close'] = saved_price
plt.subplot(4, 1, (1, 2))  # Double size by merging 3 rows
plt.plot(stock_data["Close"], c='#7076cc', lw=1.8, label='Stock Price')
plt.plot(stock_data["Close"].rolling(10, 1).mean(), alpha = 1, label='10 MA', ls='dashdot')
plt.plot(stock_data["Close"].rolling(50, 1).mean(), alpha = 1, label='50 MA', ls='dashdot', c='g')
plt.legend(loc='upper left', fontsize=14)
plt.title(f'{CODE} Close Prices', fontdict={'size': 16})
plt.grid()

ax2 = plt.gca().twinx()
ax2.bar(stock_data.index, stock_data["Volume"], alpha=0.6, color='#e6332a', width=0.8, label='Volume')
ax2.set_ylabel('Volume')
ax2.legend(loc='upper right')


# CCG Module
plt.subplot(4, 1, 3)
stock_data['CCG_V'] = stock_data['CCG'] * stock_data['Volume']
plt.scatter(stock_data.index[stock_data["CCG"] > 0.9],
            stock_data["CCG_V"].values[stock_data["CCG"] > 0.9],
            s=100, alpha=0.8, c='#E06500', edgecolors='#6B4200', label='Over Sell')
plt.scatter(stock_data.index[stock_data["CCG"] < -0.9],
            stock_data["CCG_V"].values[stock_data["CCG"] < -0.9],
            s=100, alpha=0.8, c='#00E005', marker='^', edgecolors='#003366', label='Over Bought')
plt.plot(stock_data['CCG_V'], c='#8B582E', ls='--')
plt.legend(fontsize=14)
plt.grid()
plt.title('CCG')


# OSS Module
plt.subplot(4, 1, 4)
plt.plot(stock_data["OSS"], c='brown')
max_value = stock_data["OSS"].max()
max_position = stock_data["OSS"].idxmax()
plt.scatter(max_position, max_value, color='grey', s=40, label=f'Highest: {max_value:.2f} on {max_position.date()}')
plt.legend(fontsize=14)
plt.grid()
plt.title('OSS')


plt.tight_layout()
plt.savefig(f'figs/{MARKET} {CODE} {datetime.today().strftime("%Y-%m-%d")}.png')
