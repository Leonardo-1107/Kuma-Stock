import requests

def get_NDX_tickers():
    
    headers={"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"}
    res=requests.get("https://api.nasdaq.com/api/quote/list-type/nasdaq100",headers=headers)
    main_data=res.json()['data']['data']['rows']

    print([stock['symbol'] for stock in main_data])
    return [stock['symbol'] for stock in main_data]

if __name__ == '__main__':
    get_NDX_tickers()