import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import talib as ta
from talib import abstract
import yfinance as yfin
from arch import arch_model
import quantstats

def get_llt(df, a):
    llt = np.zeros(len(df))
    llt[0] = df['close'].iloc[0]
    llt[1] = (1 - a) * df['close'].iloc[0] + a * df['close'].iloc[1]
    for n in range(2, len(df)):
        llt[n] = ((a - (a ** 2)/4) * df['close'].iloc[n]) + (((a ** 2)/2) * df['close'].iloc[n - 1]) - ((a - (3 * (a ** 2))/4) * df['close'].iloc[n - 2]) + ((2 * (1 - a)) * llt[n - 1]) - (((1 - a) ** 2) * llt[n - 2])
    
    df['llt'] = llt
    df['llt_slope'] = np.nan

    for i in range(0, len(df)):
        df.loc[df.index[i], 'llt_slope'] = df['llt'].iloc[i] - df['llt'].iloc[i - 1]
    return df


def get_bolinger_band(df, p, q, rolling, num_vol, distance_threshold, retrace_threshold):
    model = arch_model(df['llt'].dropna(), vol = 'Garch', p = p, q = q, rescale = False)
    garch_fit = model.fit(disp = 'off')
    df['garch_volitility'] = garch_fit.conditional_volatility

    df['bolinger_upper'] = df['llt'] + num_vol * df['garch_volitility']
    df['bolinger_lower'] = df['llt'] - num_vol * df['garch_volitility']

    df['bolinger_upper_smooth'] = df['bolinger_upper'].ewm(span = rolling).mean()
    df['bolinger_lower_smooth'] = df['bolinger_lower'].ewm(span = rolling).mean()

    df = get_middle(df)

    df['upper_max'] = df['bolinger_upper'].rolling(window = rolling).max()
    df['lower_min'] = df['bolinger_lower'].rolling(window = rolling).min()

    df['upper_retrace'] = (df['upper_max'] - df['bolinger_upper']) / df['upper_max']
    df['lower_retrace'] = (df['bolinger_lower'] - df['lower_min']) / df['lower_min']

    df['upper_deviation'] = abs(df['bolinger_upper'] - df['middle']) / df['middle']
    df['lower_deviation'] = abs(df['middle'] - df['bolinger_lower']) / df['middle']

    df['upper_rise'] = df['upper_deviation'] >= distance_threshold
    df['lower_down'] = df['lower_deviation'] >= distance_threshold
    df['upper_down'] = df['upper_retrace'] >= retrace_threshold
    df['lower_rise'] = df['lower_retrace'] >= retrace_threshold

    return df

def get_middle(df):
    df['peak'] = np.nan
    df['upper_change'] = np.nan
    df['lower_change'] = np.nan
    df['middle'] = np.nan

    for i in range(len(df)):
        df.loc[df.index[i], 'upper_change'] = abs(df['bolinger_upper_smooth'].iloc[i] - df['bolinger_upper_smooth'].iloc[i - 1])
        df.loc[df.index[i], 'lower_change'] = abs(df['bolinger_lower_smooth'].iloc[i] - df['bolinger_lower_smooth'].iloc[i - 1])
        
    df['peak'] = df['upper_change'] > df['lower_change']

    df['middle'] = np.where(df['peak'], df['bolinger_lower_smooth'], df['bolinger_upper_smooth'])

    df['middle'] = df['middle'].rolling(window = 50, min_periods = 1).mean()
        
    return df

def llt_strategy(df, feePaid, a, p, q, num_vol, rolling, distance_threshold, retrace_threshold, offset_threshold):
    
    equity = pd.DataFrame(index = df.index)

    df = get_llt(df, a)
    df = get_bolinger_band(
        df = df, p = p, q = q, rolling = rolling, num_vol = num_vol,
        distance_threshold = distance_threshold, retrace_threshold = retrace_threshold)

    BS = None
    highest = None
    t = 0
    df[['Bull', 'Bear']] = False

    df = df.assign(
        position = np.zeros(len(df)),
        buy = np.zeros(len(df)),
        sell = np.zeros(len(df)),
        sellshort = np.zeros(len(df)),
        buytocover = np.zeros(len(df)),
        buy_price = np.zeros(len(df)),
        sell_price = np.zeros(len(df)),
        short_price = np.zeros(len(df)),
        buytocover_price = np.zeros(len(df)),
        buy_time = np.nan,
        sell_time = np.nan,
        sellshort_time = np.nan,
        buytocover_time = np.nan,
        hold_duration = np.nan,
        profit_list = np.zeros(len(df)),
        profit_fee_list = np.zeros(len(df)),
        profit_fee_list_realized = np.zeros(len(df))
    )
    
    df[['buy_time', 'sell_time', 'sellshort_time', 'buytocover_time']] = df[['buy_time', 'sell_time', 'sellshort_time', 'buytocover_time']].apply(pd.to_datetime, errors='coerce')
    
    for i in range(2, len(df) - 1):
        
        condition1 = abs(df['bolinger_upper_smooth'].iloc[i] - df['middle'].iloc[i]) > abs(df['middle'].iloc[i] - df['bolinger_lower_smooth'].iloc[i]) # 峰向上
        condition2 = abs(df['bolinger_upper_smooth'].iloc[i] - df['middle'].iloc[i]) < abs(df['middle'].iloc[i] - df['bolinger_lower_smooth'].iloc[i]) # 峰向下
        condition3 = (df['bolinger_upper_smooth'].iloc[i] - df['bolinger_upper_smooth'].iloc[i - 1]) > 0 # 上軌向上
        condition4 = (df['bolinger_upper_smooth'].iloc[i] - df['bolinger_upper_smooth'].iloc[i - 1]) < 0 # 上軌向下
        condition5 = (df['bolinger_lower_smooth'].iloc[i] - df['bolinger_lower_smooth'].iloc[i - 1]) > 0 # 下軌向上
        condition6 = (df['bolinger_lower_smooth'].iloc[i] - df['bolinger_lower_smooth'].iloc[i - 1]) < 0 # 下軌向下
        
        entryLong = (df['llt_slope'].iloc[i] > 0) & (((df['upper_rise'].iloc[i] == True) & condition1) | ((df['lower_rise'].iloc[i] == True) & condition2))
        # exitLong = (df['close'].iloc[i] < df['llt'].iloc[i])
        entryShort = (df['llt_slope'].iloc[i] < 0) & (((df['upper_down'].iloc[i] == True) & condition1) | ((df['lower_down'].iloc[i] == True) & condition2))
        # exitShort = (df['close'].iloc[i] > df['llt'].iloc[i])
        
        df.at[df.index[i], 'Bull'] = (condition1 & condition3) | (condition2 & condition5)
        df.at[df.index[i], 'Bear'] = (condition1 & condition4) | (condition2 & condition6)

        if BS == None:
            if df['Bull'].iloc[i]:
                if entryLong:
                    BS = 'B'
                    highest = df['open'].iloc[t]
                    t = i + 1
                    df.at[df.index[t], 'buy'] = t
                    df.at[df.index[t], 'buy_price'] = df['open'].iloc[t]
                    df.at[df.index[t], 'buy_time'] = df.index[t]
                    df.at[df.index[t], 'position'] += 1

            elif df['Bear'].iloc[i]:
                if entryShort:
                    BS = 'S'
                    highest = df['open'].iloc[t]
                    t = i + 1
                    df.at[df.index[t], 'sellshort'] = t
                    df.at[df.index[t], 'short_price'] = df['open'].iloc[t]
                    df.at[df.index[t], 'sellshort_time'] = df.index[t]
                    df.at[df.index[t], 'position'] -= 1
                    
        elif BS == 'B':
            
            highest = max(highest, df['open'].iloc[i - 1])
            df.at[df.index[i + 1], 'position'] = df.at[df.index[i], 'position']
            profit = 200 * (df['open'].iloc[i + 1] - df['open'].iloc[i]) * df['position'].iloc[i + 1]
            df.at[df.index[i], 'profit_list'] = profit

            if df['close'].iloc[i] < (highest * (1 - offset_threshold)):
                pl_round = 200 * (df['open'].iloc[i + 1] - df['open'].iloc[t]) * df['position'].iloc[i]
                profit_fee = profit - feePaid * 2
                df.at[df.index[i + 1], 'position'] -= 1
                df.at[df.index[i], 'profit_fee_list'] = profit_fee
                df.at[df.index[i + 1], 'sell_price'] = df['open'].iloc[i + 1]
                df.at[df.index[i + 1], 'sell_time'] = df.index[i + 1]
                df.at[df.index[i + 1], 'hold_duration'] = pd.date_range(start = df.index[t], end = df.index[i], freq = 'B').size
                df.at[df.index[i + 1], 'sell'] = i + 1
                BS = None
                highest = None

                df.at[df.index[i + 1], 'profit_fee_list_realized'] = pl_round - feePaid * 2

            elif i == len(df) - 2:
                if df['position'].iloc[len(df) - 2] != 0:
                    unit = df['position'].iloc[len(df) - 2]
                    profit_fee = profit - feePaid * 2
                    pl_round = 200 * (df['open'].iloc[i + 1] - df['open'].iloc[t]) * df['position'].iloc[i]
                    df.at[df.index[i + 1], 'position'] -= unit
                    df.at[df.index[i], 'profit_fee_list'] = profit_fee
                    df.at[df.index[i + 1], 'sell_price'] = df['open'].iloc[i + 1]
                    df.at[df.index[i + 1], 'sell_time'] = df.index[i + 1]
                    df.at[df.index[i + 1], 'hold_duration'] = pd.date_range(start = df.index[t], end = df.index[i], freq = 'B').size
                    df.at[df.index[i + 1], 'sell'] = i + 1
                    BS = None
                    highest = None

            else:
                profit_fee = profit
                df.at[df.index[i], 'profit_fee_list'] = profit_fee

        elif BS == 'S':
            
            highest = min(highest, df['open'].iloc[i])
            df.at[df.index[i + 1], 'position'] = df.at[df.index[i], 'position']
            profit = 200 * (df['open'].iloc[i + 1] - df['open'].iloc[i]) * df['position'].iloc[i + 1]
            df.at[df.index[i], 'profit_list'] = profit

            if df['close'].iloc[i] > (highest * (1 + offset_threshold)):
                pl_round = 200 * (df['open'].iloc[i + 1] - df['open'].iloc[t]) * df['position'].iloc[i]
                profit_fee = profit - feePaid * 2
                df.at[df.index[i + 1], 'position'] += 1
                df.at[df.index[i], 'profit_fee_list'] = profit_fee
                df.at[df.index[i + 1], 'buytocover_price'] = df['open'].iloc[i + 1]
                df.at[df.index[i + 1], 'buytocover_time'] = df.index[i + 1]
                df.at[df.index[i + 1], 'hold_duration'] = pd.date_range(start = df.index[t], end = df.index[i], freq = 'B').size
                df.at[df.index[i + 1], 'buytocover'] = i + 1
                BS = None
                highest = None

                profit_fee_realized = pl_round - feePaid * 2
                df.at[df.index[i + 1], 'profit_fee_list_realized'] = profit_fee_realized

            elif i == len(df) - 2:
                if df['position'].iloc[len(df) - 2] != 0:
                    unit = df['position'].iloc[len(df) - 2]
                    profit_fee = profit - feePaid * 2
                    pl_round = 200 * (df['open'].iloc[i + 1] - df['open'].iloc[t]) * df['position'].iloc[i]
                    df.at[df.index[i + 1], 'position'] += unit
                    df.at[df.index[i], 'profit_fee_list'] = profit_fee
                    df.at[df.index[i + 1], 'sell_price'] = df['open'].iloc[i + 1]
                    df.at[df.index[i + 1], 'buytocover_time'] = df.index[i + 1]
                    df.at[df.index[i + 1], 'hold_duration'] = pd.date_range(start = df.index[t], end = df.index[i], freq = 'B').size
                    df.at[df.index[i + 1], 'buytocover'] = i + 1
                    BS = None
                    highest = None

            else:
                profit_fee = profit
                df.at[df.index[i], 'profit_fee_list'] = profit_fee

        else:
            print('error')
                
    df['strategy_ret'] = df['profit_list'].cumsum()
    equity['profitfee'] = df['profit_fee_list'].cumsum()

    return df, equity

def get_performance(df, equity, fund = 1000000):
    duration = (equity.index[-1] - equity.index[0]).days
    risk_free_rate = 0.04/252
    equity['equity'] = equity['profitfee'] + fund
    equity['drawdown_percent'] = (equity['equity']/equity['equity'].cummax()) - 1
    equity['drawdown'] = equity['equity'] - equity['equity'].cummax()
    profit = equity['profitfee'].iloc[-1]
    ret = equity['equity'].iloc[-1]/equity['equity'].iloc[0] - 1
    mdd = abs(equity['drawdown_percent'].min())
    calmarRatio = ret / mdd
    tradeTimes = min((df['buy'] > 0).sum(), (df['sell'] > 0).sum()) + min((df['sellshort'] > 0).sum(), (df['buytocover'] > 0).sum())
    average_holding_duration = df['hold_duration'].sum() / tradeTimes
    winRate = len([i for i in df['profit_fee_list_realized'] if i > 0]) / tradeTimes
    try:
        profitFactor = sum([i for i in df['profit_fee_list_realized'] if i>0]) / abs(sum([i for i in df['profit_fee_list_realized'] if i < 0]))
    except:
        profitFactor = None
    mean_ret = df['profit_list'].mean()
    std_ret = df['profit_list'].std()
    sharp = (mean_ret - risk_free_rate) / std_ret

    print('Duration : ', duration, 'days')
    print('Average holding duration', average_holding_duration, 'days')
    print('Profit : ', profit)
    print('Return : ', ret)
    print('Max DrawDown : ', mdd)
    print('Caimar Ratio : ', calmarRatio)
    print('Trade Times : ', tradeTimes)
    print('Win Rate : ', winRate)
    print('Profit Factor : ', profitFactor)
    print('Sharp Ratio : ', sharp)