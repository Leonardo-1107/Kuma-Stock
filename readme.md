# Kuma Stock Analysis

## Introduction
This project is designed to analyis the trading trend of **single stock** or **group of stocks** via graphic methods. 

## Code
*  `Single_analysis.py`: analysis a single share based on chosen factors.
*  `Multiple_analysis.ipynb`: evaluate performance of multiple stocks in a sector.
*  `factors_lib.py`: functions to get various factors. detailed information can be seen inside from the notes.
*  `get_tickers.py`: functions to get stock tickers for inquery. Now support *SP500* and *Nasdaq 100*.
*  `utils.py`: other functions.

## Execute Instructions

Environments
```
pip install -r requirements.txt
```

Single stock analysis:
```
python Single_analysis.py [MARKET] [STOCK TICKER]
```
where MARKET could be `US`, `HK` and `CN`. Because of the difference of CN and other markets, two distinct form of tickers are usd. For `CN` market, the ticker's from is `000001.SZ` etc, and for other markets, the form is usually upper letters `APPL` .etc.