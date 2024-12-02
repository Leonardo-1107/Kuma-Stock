import requests
import yfinance as yf
import pandas as pd

def get_NDX_tickers():
    """
    Get all the info of stocks in NDX, including tickers
    """
    headers={"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"}
    res=requests.get("https://api.nasdaq.com/api/quote/list-type/nasdaq100",headers=headers)
    main_data=res.json()['data']['data']['rows']

    return [stock['symbol'] for stock in main_data]


def get_sp500_stockinfo_df()->pd.DataFrame:
    """
    Get all the info of stocks in SP500, including tickers and GICS sector
    """
    tickers = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

    # keep the info of ticks industrials
    stock_info = tickers[['Symbol', 'GICS Sector']]
    return stock_info


if __name__ == '__main__':
    get_NDX_tickers()