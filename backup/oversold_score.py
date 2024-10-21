import pandas as pd
import datetime
import csv
import yfinance as yf
import argparse

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

    df = pro.daily(ts_code=CODE, start_date='20220601', end_date='20240928')
    df.sort_values(by='trade_date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['date'] = pd.to_datetime(df['trade_date'])

    stock_data = df

else:
    # Get the info of stocks by yfinance, US Stocks
    stock = yf.Ticker(CODE)
    stock_data = stock.history(period='5y')
    stock_data['high'] = stock_data['High'].copy()
    stock_data['close'] = stock_data['Close'].copy()
    stock_data['date'] = pd.to_datetime(stock_data.index)

def cal_day_interval(stock_data: pd.DataFrame, max_target='High'):
    """
    calculate the days interval number of current price and maximum price
    """
    
    max_high_idx = stock_data[max_target].idxmax()
    date_max_high = stock_data.loc[max_high_idx, 'date']
    date_latest_close = stock_data['date'].iloc[-1]

    days_difference = (date_latest_close - date_max_high).days

    return days_difference


def price_recovery_rate(stock_data: pd.DataFrame, n=5, target = 'High'):
    previous_price = stock_data[target].values[-1*n]

    return 100 * (stock_data['close'].values[-1] - previous_price) / previous_price


def Over_Sold_Score(stock_data:pd.DataFrame, is_ratio_drop = True, is_print = False):

    """
    Based on the recent trend of Chinese A stock, designed a score to measure 
    how 'overso'
    
    """
    # calculate the PD3 parameter
    PD3 = (max(stock_data['high'].values) - stock_data['close'].values[-1]) 
    if is_ratio_drop:
        PD3 = PD3 / stock_data['close'].values[-1]
        
    # days interval
    L = cal_day_interval(stock_data=stock_data, max_target='high')  

    # recovery rate
    Recovery_week = price_recovery_rate(stock_data=stock_data, target='high')

    if PD3 < 0:
        # over-bought right now
        return -1
    
    if is_print:
        print(f"Drop from highest ratio: {PD3*100:.1f} %, between {L} days. Recovered {Recovery_week:.2f} % last week")
    
    return PD3 * L / Recovery_week


OS = Over_Sold_Score(stock_data, is_print=True)
data = [datetime.date.today(), MARKET, CODE, f'{OS:.2f}']
print(f"\n{datetime.date.today()} {MARKET}, {CODE}, OS Value: {OS:.2f}")

csv_filename = 'stock_inquiry_records.csv'
# Write to CSV file
with open(csv_filename, mode='a+', newline='') as file:
    writer = csv.writer(file)
    
    # Check if file is empty, write header
    file.seek(0)
    if not file.read(1):  # Check if file is empty
        writer.writerow(['Date', 'Market', 'Code', 'OS Value'])
    
    # Write the data row
    writer.writerow(data)
