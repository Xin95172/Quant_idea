import pandas as pd

def optimize_by_date_range(df, optimize_fn, optimize_indicator, date_ranges, ascending = False):
    results = []
    for start_date, end_date in date_ranges:
        # 筛选指定日期范围内的数据
        period_df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # 检查是否有数据
        if period_df.empty:
            print(f"No data for period {start_date} to {end_date}")
            continue
        
        # 执行优化
        results_df = optimize_fn(
            a_range=a, o_range=o,
            upper_threshold_range=upper_threshold, lower_threshold_range=lower_threshold,
            retrace_u_threshold_range=retrace_u_threshold, retrace_l_threshold_range=retrace_l_threshold,
            df=period_df, fund=fund, feePaid=feePaid, p=p, q=q, rolling=rolling, num_vol=num_vol
        )
        
        # 提取最佳参数
        best_params = extract_best_params(results_df, optimize_indicator, ascending)
        best_params['start_date'] = start_date
        best_params['end_date'] = end_date
        results.append(best_params)
    
    return pd.DataFrame(results)

# 指定日期范围
date_ranges = [
    ('2023-01-01', '2023-03-31'),
    ('2023-04-01', '2023-06-30'),
    ('2023-07-01', '2023-09-30')
]

# 使用该函数进行优化
optimized_params_by_dates = optimize_by_date_range(df, optimize, 'sharp_ratio', date_ranges)

# 保存结果
optimized_params_by_dates.to_csv('optimized_params_by_dates.csv', index=False)
