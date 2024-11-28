import pandas as pd
import numpy as np


def PD_standardlization(data):
    return (data-data.mean())/data.std()


def Rise_N(pool_df, N=5):
    """
    涨幅因子
    计算间隔N日收盘价之差
    """
    
    df = pool_df[['TradingDay', 'ContractId', 'ClosePrice', 'PrevClosePrice']].copy()
    df.columns = ['date', 'code', 'cp', 'pcp']
    df.loc[:, 'change'] = df['cp'] - df['pcp']
    result = df.groupby('code')['change'].rolling(window=N, min_periods=N).mean().reset_index(level=0, drop=True)*100
    
    return result


def Return_std_N(pool_df, N=5):
    """
    The standard derviation of the return within N days
    收益标准差
    """

    df = pool_df[['TradingDay', 'ContractId', 'Return']].copy()
    df.columns = ['date', 'code', 'return']
    
    result = df.groupby('code')['return'].rolling(window=N, min_periods=N).std().reset_index(level=0, drop=True)*100
    
    return result


def Bias_N(pool_df, N=5, select = 'ClosePrice'):
    """
    乖离率因子：
    计算对应参数与N日均线表现之差
    除了PrevPrice，其他均要考虑未来数据的问题
    """

    df = pool_df[['TradingDay', 'ContractId', select]].copy()
    df['Target'] = pd.to_numeric(df[select])
    df['rolling_mean'] = df.groupby('ContractId')['Target'].rolling(window=N, min_periods=1).mean().reset_index(level=0, drop=True)
    result = (df['Target'] - df['rolling_mean']) / df['rolling_mean']*100

    return result


def TV_Bias_N(pool_df, N=5):
    """
    换手率均线因子：
    计算对应参数与N日均线表现之差
    """

    df = pool_df.copy()
    df['Amount'] = pd.to_numeric(df['Amount'])
    df['FloatAShare'] = pd.to_numeric(df['FloatAShare'])

    df['Target'] = df['Amount']/df['FloatAShare'] * 100

    df['rolling_mean'] = df.groupby('ContractId')['Target'].rolling(window=N, min_periods=1).mean().reset_index(level=0, drop=True)
    result = (df['Target'] - df['rolling_mean']) / df['rolling_mean']*100
    
    return result


def Cross(pool_df, N1=5, N2=20):
    """
    均线交叉因子：
    股价上涨时小均线交叉大均线会形成金叉，代表未来一段时间内看涨
    """

    df = pool_df[['TradingDay', 'ContractId', 'PrevClosePrice']].copy()
    df['Target'] = pd.to_numeric(df['PrevClosePrice'])
    df['rolling_mean_1'] = df.groupby('ContractId')['Target'].rolling(window=N1, min_periods=1).mean().reset_index(level=0, drop=True)
    df['rolling_mean_2'] = df.groupby('ContractId')['Target'].rolling(window=N2, min_periods=1).mean().reset_index(level=0, drop=True)
    
    result = (df['rolling_mean_1'] - df['rolling_mean_2']) / df['rolling_mean_2']*100

    return result


def ForceIndex(pool_df: pd.DataFrame, N=5):
    """
    Force Index effectively combines stock price changes and trading volume in a 
    multiplied form to assess the strength of stock price trends.
    """
    df = pool_df.copy()
    delta_price = df['ClosePrice'].pct_change(N).fillna(0).values

    df['force_index'] = delta_price * df['Volume']
    try:
        res = df.groupby('ContractId')['force_index'].rolling(window=N, min_periods=1).mean().reset_index(level=0, drop=True)
    except:
        # single share
        res = df['force_index'].values
    return res


def Volume_std_N(pool_df, N=20):
    """
    情绪类因子 
    N 日内成交量标准差
    """

    df = pool_df[['TradingDay', 'ContractId', 'Volume']].copy()
    df.columns = ['date', 'code', 'Volume']
    df['Volume'] = pd.to_numeric(df["Volume"])

    result = df.groupby('code')['Volume'].rolling(window=N).std().reset_index(level=0, drop=True)

    return result


def Daily_std_dev(pool_df, halflife = 42, N=252):
    """
    新日收益率标准差
    计算公式： sqrt(sum(w_t*(r_t - r_mean)**2))
    其中r_t为过去252个交易日的日收益率, w_t为半衰期为42个交易日的指数权重, 满足w(t-42)=0.5*w(t)
    """

    df = pool_df[['TradingDay', 'ContractId', 'Return']].copy()
    df.columns = ['date', 'code', 'return']
    # 计算每个股票的日收益率
    df['return'] = pd.to_numeric(df['return'])
    halflife = 42
    weights = np.power(0.5, np.arange(len(df) - 1, -1, -1) / halflife)

    # 计算加权标准差
    def weighted_std(arr):
        weighted_var = np.sum(weights[-len(arr):] * (arr - arr.mean())**2)
        return np.sqrt(weighted_var)

    # 计算每个股票的daily_standard_deviation
    result = df.groupby('code')['return'].rolling(window=N, min_periods=5).apply(weighted_std, raw=False).reset_index(level=0, drop=True)
    
    return result


def Volume_Weighted_Return(pool_df, N=1):
    """
    计算 N 日收益除以交易量滚动平均（当天收入 return / 当天市值 market_value）。
    """

    df = pool_df[['TradingDay', 'ContractId', 'Return', 'Volume']].copy()
    
    df['date'] = pd.to_datetime(df['TradingDay'])
    df['Return'] = pd.to_numeric(df['Return'])
    df['Volume'] = pd.to_numeric(df['Volume'] )

    df['daily_return_over_market_value'] = df['Return'] / np.log(df['Volume'])
    
    # 计算 N 日滚动平均
    rolling_average = df.groupby('ContractId')['daily_return_over_market_value'].rolling(window=N).mean().reset_index(level=0, drop=True)

    return rolling_average


def MV_weighted_return(pool_df, N=5):
    """
    小市值收益因子
    旧收益率：5日内Close Price 变化幅度 / 估算市值
    新收益率：Return * AdjFactor 调整后的收益率 / 估算市值
    """
    
    df = pool_df.copy()
    
    df['Return'] = pd.to_numeric(df['Return'])
    df['TotalShare'] = pd.to_numeric(df['TotalShare'])
    df['ClosePrice'] = pd.to_numeric(df['ClosePrice'] )
    df['AdjFactor'] = pd.to_numeric(df['AdjFactor'] )

    # RET = df['ClosePrice']
    # RET = RET.pct_change(5)*100
    RET = df['Return']*df['AdjFactor']
    MV = df['TotalShare']*df['ClosePrice']

    RET = PD_standardlization(RET)
    MVW = PD_standardlization(1/MV)

    df['MV_RET'] =  RET + MVW

    if N==1:
        return df['MV_RET'] 
    
    rolling_average = df.groupby('ContractId')['MV_RET'].rolling(window=N).mean().reset_index(level=0, drop=True)

    return rolling_average


def MV(pool_df):
    """
    小市值收益因子
    """  
    df = pool_df.copy()
    
    df['TotalShare'] = pd.to_numeric(df['TotalShare'])
    df['ClosePrice'] = pd.to_numeric(df['ClosePrice'] )

    MV = df['TotalShare']*df['ClosePrice']

    return 1/MV


def PE(pool_df):
    """
    PE 指标因子
    """

    df = pool_df[['ClosePrice', 'AdjFactor', 'PrevClosePriceAdj']].copy()

    df['ClosePrice'] = pd.to_numeric(df['ClosePrice'])
    df['AdjFactor'] = pd.to_numeric(df['AdjFactor'] )
    df['PrevClosePriceAdj'] = pd.to_numeric(df['PrevClosePriceAdj'] )
    df.loc[:, 'PE'] = df['ClosePrice'] * df['AdjFactor'] / df['PrevClosePriceAdj']

    return df['PE']


def PEG(pool_df, N=5):
    """
    模仿 PEG 指标因子, 但由于 G 是估算得来，并不准确
    """

    df = pool_df[['ContractId', 'Return', 'ClosePrice', 'AdjFactor', 'PrevClosePrice']].copy()

    df['Return'] = pd.to_numeric(df['Return'])
    df['ClosePrice'] = pd.to_numeric(df['ClosePrice'])
    df['PrevClosePrice'] = pd.to_numeric(df['PrevClosePrice'])
    df['AdjFactor'] = pd.to_numeric(df['AdjFactor'] )

    df['PE'] = df['Return'] * df['AdjFactor'] 
    df['G'] = (df['ClosePrice'] - df['PrevClosePrice']) / df['PrevClosePrice']

    return df['G'] / df['PE']


def WPEG(pool_df, N=42):
    """
    半衰期加权的 PEG 指标因子
    """

    df = pool_df[['ContractId', 'Return', 'ClosePrice', 'AdjFactor', 'PrevClosePrice']].copy()

    df['Return'] = pd.to_numeric(df['Return'])
    df['ClosePrice'] = pd.to_numeric(df['ClosePrice'])
    df['PrevClosePrice'] = pd.to_numeric(df['PrevClosePrice'])
    df['AdjFactor'] = pd.to_numeric(df['AdjFactor'] )

    PE = df['Return'] * df['AdjFactor'] 
    G = (df['ClosePrice'] - df['PrevClosePrice']) / df['PrevClosePrice']
    
    df['PEG'] = G / PE 

    halflife = 42
    weights = np.power(0.5, np.arange(len(df) - 1, -1, -1) / halflife)

    def weighted_std(arr):
        weighted_var = np.sum(weights[-len(arr):] * (arr - arr.mean())**2)
        return np.sqrt(weighted_var)

    # 计算每个股票的daily_standard_deviation
    result = df.groupby('ContractId')['PEG'].rolling(window=N, min_periods=5).apply(weighted_std, raw=False).reset_index(level=0, drop=True)
    return result

def FHW_Approaching(input_df:pd.DataFrame):
    
    """模拟的羊群因子"""
    df = input_df.copy()
    for i in df.columns:
        try:
            df[i] = pd.to_numeric(df[i])
        except:
            continue
    # Approaching the purchase intensity via vwap
    df['PurchaseIntensity'] = PD_standardlization((df["ClosePrice"] - df["VWAP"])/df["VWAP"]) + PD_standardlization(df["Volume"])
    df['AvgPurchaseIntensity'] = df['PurchaseIntensity'].rolling(window=df.shape[0], min_periods=1).mean()

    df['FHW_Numerator'] = (df['PurchaseIntensity'] - df['AvgPurchaseIntensity'])**2
    df['FHW_Denominator'] = df['FHW_Numerator'].mean()

    input_df["FHW"] = df['FHW_Numerator'] / df['FHW_Denominator'] * (-1)
    return input_df["FHW"]

def Momentum(df, n_months, cal_choose = 'Volume', exclude_recent_months=1):
    """
    Momentum factors via 21 days multiple the month

    Args:
        cal_choose (str): The target used for calculation.
        n_days (int): The number of days over which the reversion is calculated. By default, it is set to 5.
    """
    df[cal_choose] = pd.to_numeric(df[cal_choose])
    n_days = n_months * 21
    exclude_days = exclude_recent_months * 21
    # Calculate the momentum factor
    df[f'Momentum_{n_months}M'] = df[cal_choose].rolling(window=n_days).apply(
        lambda x: (x[-exclude_days-1] - x[0]) / x[0] if len(x) > exclude_days else np.nan,
        raw=True
    )
    return df[f'Momentum_{n_months}M']

def Short_Term_Reversion(df, cal_choose = 'Volume', n_days=5):
    """
    Short-term reversal effect factor

    Args:
        n_days (int): The number of days over which the reversion is calculated. By default, it is set to 5.

    """
    reversion_df = df.copy()
    reversion_df[f'Reversion_{n_days}D'] = reversion_df[cal_choose].rolling(window=n_days).apply(
        lambda y: -np.sum(np.diff(np.log(y))) if len(y) == n_days else np.nan
    )

    return reversion_df[f'Reversion_{n_days}D']


def Oversold_Reverse_Score(df:pd.DataFrame, day_length=400):
    """
    Oversold factor to measure how over-sold the stock is and therefore indicate the potential

    Args:
        df: input Dataframe
        day_length: days range for period-maximum price
    
    """
    # difference between current Open Price and the highest Close Price 
    rolling_max_high = df["HighPrice"].rolling(window=day_length, min_periods=1).max()
    PD = (rolling_max_high - df["ClosePrice"] ) / df["ClosePrice"]
    PD.fillna(0, inplace=True)
    
    # days interval between max and current
    rolling_max_idx = df['HighPrice'].rolling(window=day_length, min_periods=1).apply(lambda x: len(x) - 1 - np.argmax(x[::-1]), raw=True)
    L = df.reset_index(drop=True).index.values - rolling_max_idx.values

    # normalise the price
    min_price = df['ClosePrice'].min()
    max_price = df['ClosePrice'].max()
    NPrice = (df['ClosePrice'] - min_price)/(max_price - min_price) + 1
    df["OS_score"] = PD.values * L / NPrice.values
    
    return df["OS_score"].values


def CCG_Score(df:pd.DataFrame, window_length=9, interval=4):
    """
    Continous Change (CCG) factor

    Args:
        - window_length: the window for calculating the number of days
        - interval: compare to  [T - interval] days
    The CCG Score calcultes the days number of price rising or falling 
    in given length of periods
    - 1 when the stock price has risen over,
    - 0 when the price remains unchanged,
    - -1 when the price has fallen.

    """
    df = df.copy()
    cal_rise_num = df['ClosePrice'] - df['ClosePrice'].shift(interval).fillna(0)
    df['Rise_Bool'] = np.where(cal_rise_num > 0, 1, np.where(cal_rise_num < 0, -1, 0))

    CCG = df['Rise_Bool'].rolling(window=window_length, min_periods=1).mean().values * -1

    return CCG



from sklearn.preprocessing import minmax_scale
def ILLIQ_Factor(df: pd.DataFrame, window_length=9):
    """
    Calculate the ILLIQ factor for a given stock.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'ClosePrice' and 'Volume'.
        window_length (int): Number of days over which the ILLIQ is calculated.
        
    Returns:
        np.ndarray: ILLIQ values for each window of time.
    """
    df = df.copy()
    
    df['Return'] = df['ClosePrice'].pct_change().fillna(0)
    df['Abs_Return'] = df['Return'].abs()
    
    # Calculate the ILLIQ factor as the mean of (Abs_Return / Volume) over the window_length
    df['ILLIQ'] = df['Abs_Return'] / df['Volume']
    
    ILLIQ = df['ILLIQ'].rolling(window=window_length, min_periods=1).mean().values
    
    return np.log(ILLIQ + 1)