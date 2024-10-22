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
    start = "2023-10-01"
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

seq_length = 21
predict_length = 7
feature_list = ['High', 'Low', 'Open', 'Close', 'OSS', 'CCG', 'Momentum', 'ILLIQ']
saved_price = stock_data['Close'].copy()
saved_label = minmax_scale(stock_data['Close'].copy())


from KSL_system import KumaModel
reg_model = KumaModel(
            model_name='drnn', 
            input_size=len(feature_list),
            hidden_size=32,
            output_size=1,
            num_layers=2,
            dropout_rate=0.3)
reg_model.set_train_params(loss_type='ic')

reg_model.model.load_state_dict(torch.load('kuma_models/my_model.pth', weights_only=True))
my_scaler = joblib.load('kuma_models/scaler.gz')
stock_data.fillna(0, inplace=True)
my_scaler = StandardScaler()
stock_data[feature_list] = my_scaler.fit_transform(stock_data[feature_list])

stock_data['Label'] = saved_label
sequences, labels = Creat_Sequence(
                        stock_data, 
                        feature_list, 
                        seq_length=seq_length, 
                        predict_length=predict_length)



# reg_model.train_model(sequences, y=labels, seq_length=seq_length, epochs=2000)
y_scaled, past_prices, future_prices = buying_index(reg_model, sequences, predict_length)


# plot pics
if MARKET == 'CN':
    plt.figure(figsize=(12, 8), facecolor='lightgrey')
else:
    plt.figure(figsize=(12, 12), facecolor='lightgrey')

stock_data['Close'] = saved_price
plt.subplot(6, 1, (1, 2))  # Double size by merging 3 rows
plt.plot(stock_data["Close"], c='#7076cc', lw=1.5, label='Stock Price')
plt.plot(stock_data["Close"].rolling(10, 1).mean(), lw=0.5, alpha = 1, label='10 MA')
plt.plot(stock_data["Close"].rolling(50, 1).mean(), lw=0.5, alpha = 1, label='50 MA')
plt.legend(loc='upper left')
plt.title(f'{CODE} Close Prices')
plt.grid()

ax2 = plt.gca().twinx()
ax2.bar(stock_data.index, stock_data["Volume"], alpha=0.6, color='#e6332a', width=0.4, label='Volume')
ax2.set_ylabel('Volume')
ax2.legend(loc='upper right')

if MARKET != 'CN':
# for other markets, provide with GRU analysis
    plt.subplot(6, 1, 3)
    plt.scatter(range(predict_length), past_prices, 
                color='yellow', s=100, label=f'past {predict_length} days', edgecolors='black')
    plt.scatter(range(predict_length, 2*predict_length), future_prices, alpha=0.7,
                color='blue', s=100, label=f'future {predict_length} days', edgecolors='black')
    plt.plot(y_scaled[-2*predict_length:], lw=1.2, ls='--', alpha=0.7, c='purple')
    trend_line = pd.DataFrame(y_scaled)[-2*predict_length:].rolling(3, 1).mean().values
    plt.plot(trend_line, c='r', alpha=0.5, label='trend line')
    x_ticks = np.arange(-predict_length, predict_length + 1)
    x_labels = [f"T{(i if i == 0 else ('+' if i > 0 else '-') + str(abs(i)))}" for i in x_ticks]

    plt.legend(loc='upper left')
    plt.title('BUX')
    plt.xticks(ticks=np.arange(2 * predict_length + 1), labels=x_labels)


else:
    plt.subplot(6, 1, 3)
    plt.plot(stock_data["STR"], c='g')
    plt.grid()
    plt.title('STR')

# STR Module
# plt.subplot(6, 1, 3)
# plt.plot(stock_data["STR"], c='g')
# plt.grid()
# plt.title('STR')

# FRX Module
plt.subplot(6, 1, 4)
plt.plot(stock_data['FRX'], c='r')
plt.grid()
plt.title('FRX')

# FHW Module
# plt.subplot(6, 1, 5)
# plt.plot(stock_data["FHW"], c='orange')
# plt.grid()
# plt.title('FHW')

# CCG Module
plt.subplot(6, 1, 5)
stock_data['CCG_V'] = stock_data['CCG'] * stock_data['Volume']
plt.scatter(stock_data.index[stock_data["CCG"] > 0.9],
            stock_data["CCG_V"].values[stock_data["CCG"] > 0.9],
            s=100, alpha=0.8, c='#E06500', edgecolors='#6B4200')

plt.scatter(stock_data.index[stock_data["CCG"] < -0.9],
            stock_data["CCG_V"].values[stock_data["CCG"] < -0.9],
            s=100, alpha=0.8, c='#00E005', marker='^', edgecolors='#003366')

plt.plot(stock_data['CCG_V'], c='#8B582E', ls='--')
# plt.ylim([-2, 2])
plt.grid()
plt.title('CCG')

# OSS Module
plt.subplot(6, 1, 6)
plt.plot(stock_data["OSS"], c='brown')
plt.grid()
plt.title('OSS')


plt.tight_layout()
plt.savefig(f'figs/{MARKET} {CODE} {datetime.today().strftime("%Y-%m-%d")}.png')
