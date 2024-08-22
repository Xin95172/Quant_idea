import pandas as pd

def ensure_iterable(param):
    if param is None:
        return []
    if isinstance(param, (float, int)):
        return [param]
    return param

def walk_forward(
        strategy,
        data_full,
        warmup_bars,
        lookback_bars,
        validation_bars,
        fund,
        feePaid,
        optimize_fn = None,
        **strategy_params):
    
    stats_master = []
    params_results = []

    a_range = strategy_params.get('a', [])
    o_range = strategy_params.get('o', [])
    upper_threshold_range = strategy_params.get('upper_threshold', [])
    lower_threshold_range = strategy_params.get('lower_threshold', [])
    retrace_u_threshold_range = strategy_params.get('retrace_u_threshold', [])
    retrace_l_threshold_range = strategy_params.get('retrace_l_threshold', [])
    p = strategy_params.get('p', None)
    q = strategy_params.get('q', None)
    rolling = strategy_params.get('rolling', None)
    num_vol = strategy_params.get('num_vol', None)
    
    for i in range(lookback_bars + warmup_bars, len(data_full) - validation_bars, validation_bars):
        print(f'Processing index: {i}')

        sample_data = data_full.iloc[i - lookback_bars : i]
        validation_data = data_full.iloc[i - warmup_bars : i + validation_bars]
        
        print(f'Sample data:\n{sample_data.head()}')
        print(f'Strategy parameters before optimization: {strategy_params}')
        
        if optimize_fn is not None:
            try:
                optimize_params_df = optimize_fn(
                    a_range = ensure_iterable(a_range),
                    o_range = ensure_iterable(o_range),
                    upper_threshold_range = ensure_iterable(upper_threshold_range),
                    lower_threshold_range = ensure_iterable(lower_threshold_range),
                    retrace_u_threshold_range = ensure_iterable(retrace_u_threshold_range),
                    retrace_l_threshold_range = ensure_iterable(retrace_l_threshold_range),
                    df = sample_data,
                    fund = fund,
                    feePaid = feePaid,
                    p = p,
                    q = q,
                    rolling = rolling,
                    num_vol = num_vol
                )
                
                if optimize_params_df.empty:
                    print("Optimization returned an empty DataFrame. Using default strategy parameters.")
                    best_params = strategy_params
                else:
                    print("Optimization Params DataFrame:")
                    print(optimize_params_df)
                    
                    if 'sharp_ratio' in optimize_params_df.columns:
                        best_params = optimize_params_df.sort_values(by='sharp_ratio', ascending=False).iloc[0].to_dict()
                        print(f'Best params: {best_params}')
                        strategy_params.update(best_params)
                    else:
                        print("sharp_ratio column not found in optimize_params_df")
                        print("Using default strategy parameters.")
                        best_params = strategy_params
                
                stats_master.append(best_params)
                params_results.append(best_params)
                
            except Exception as e:
                print(f"Optimization failed with error: {e}")
                best_params = strategy_params
        
    params_results_df = pd.DataFrame(params_results)
    print(f'Params results DataFrame:\n{params_results_df.head()}')
    
    return stats_master, params_results_df