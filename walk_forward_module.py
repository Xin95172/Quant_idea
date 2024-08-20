import pandas as pd

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

    for i in range(lookback_bars + warmup_bars, len(data_full) - validation_bars, validation_bars):

        print(f'Processing index: {i}')

        sample_data = data_full.iloc[i - lookback_bars : i]
            
        if optimize_fn is not None:
            optimize_params = optimize_fn(sample_data, strategy, **strategy_params)
            strategy_params.update(optimize_params)

        validation_data = data_full.iloc[i - warmup_bars : i + validation_bars]

        resault = strategy(validation_data, fund = fund, feePaid = feePaid, **strategy_params)

        stats_master.append(resault)
        params_results.append(strategy_params.copy())
    
    params_results_df = pd.DataFrame(params_results)
    return stats_master, params_results_df