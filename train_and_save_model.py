import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from factors_lib import *
from sklearn.preprocessing import StandardScaler, minmax_scale
from utils import Creat_Sequence, buying_index, calculate_vwap
from KSL_system import KumaModel
import torch
import joblib

# Define the ticker symbol for Dow Jones Industrial Average
MARKET = "US"
PERIOD = '1y'
feature_list = ['High', 'Low', 'Open', 'Close', 'OSS', 'CCG', 'Momentum', 'ILLIQ']

reg_model = KumaModel(
    model_name='drnn', 
    input_size=len(feature_list),
    hidden_size=32,
    output_size=1,
    num_layers=2,
    dropout_rate=0.3,
    is_plot=True)
reg_model.set_train_params(loss_type='ic')


def train_and_save(code='^DJI', market='US', period='1y', seq_length=14, predict_length=7, epochs=15000):
    # Create a Ticker object
    stock = yf.Ticker(code)
    stock_data = stock.history(period=period)

    target = "Close"
    stock_data["ClosePrice"] = stock_data["Close"]
    stock_data["HighPrice"] = stock_data["High"]
    stock_data['Momentum'] = Momentum(df=stock_data, n_months=3, cal_choose=target).fillna(0)
    stock_data["STR"] = Short_Term_Reversion(df=stock_data, cal_choose=target, n_days=5).fillna(0)
    stock_data["VWAP"] = calculate_vwap(high=stock_data["High"], low=stock_data["Low"], close=stock_data["Close"], volume=stock_data["Volume"])
    stock_data["FHW"] = FHW_Approaching(input_df=stock_data) * -1
    stock_data["OSS"] = Oversold_Reverse_Score(df=stock_data)
    stock_data["CCG"] = CCG_Score(stock_data)
    stock_data["ILLIQ"] = ILLIQ_Factor(stock_data, 9)

    # the labels should be condsidering future changes
    saved_label = minmax_scale((stock_data['High'].shift(-1)).fillna(0).copy())
    # saved_label = minmax_scale(stock_data['Close'].copy())
    # load the original scaler for the first candidate
    try:
        scaler = joblib.load('kuma_models/scaler.gz')
    except:
        scaler = StandardScaler()
        stock_data.fillna(0, inplace=True)
        scaler.fit(stock_data[feature_list])

    stock_data[feature_list] = scaler.transform(stock_data[feature_list])
    stock_data['Label'] = saved_label
    sequences, labels = Creat_Sequence(stock_data, feature_list, seq_length=seq_length, predict_length=predict_length)

    # set to train

    reg_model.train_model(sequences, y=labels, seq_length=seq_length, epochs=epochs)

    torch.save(reg_model.model.state_dict(), 'kuma_models/my_model.pth')
    joblib.dump(scaler, 'kuma_models/scaler.gz')

code_list = ['NVDA']


if __name__ == '__main__':

    for CODE in code_list:
        train_and_save(CODE, MARKET)



# y_scaled, past_prices, future_prices = buying_index(reg_model, sequences, predict_length)
# plt.figure(figsize=(10, 4))
# plt.scatter(range(predict_length), past_prices, 
#             color='yellow', s=100, label=f'past {predict_length} days', edgecolors='black')
# plt.scatter(range(predict_length, 2*predict_length), future_prices, alpha=0.7,
#             color='blue', s=100, label=f'future {predict_length} days', edgecolors='black')
# plt.plot(y_scaled[-2*predict_length:], lw=1.2, ls='--', alpha=0.7, c='purple')
# trend_line = pd.DataFrame(y_scaled)[-2*predict_length:].rolling(3, 1).mean().values
# plt.plot(trend_line, c='r', alpha=0.5, label='trend line')
# x_ticks = np.arange(-predict_length, predict_length + 1)
# x_labels = [f"T{(i if i == 0 else ('+' if i > 0 else '-') + str(abs(i)))}" for i in x_ticks]

# plt.xticks(ticks=np.arange(2 * predict_length + 1), labels=x_labels)
# plt.legend(loc='upper left')
# plt.show()