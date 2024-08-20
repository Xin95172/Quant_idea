import pandas as pd
from operator import itemgetter

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
    """
    Walk-forward optimization for trading strategy.

    Parameters:
    - strategy: The strategy function to test.
    - data_full: The complete dataset.
    - warmup_bars: Number of bars for the warm-up period.
    - lookback_bars: Number of bars to use for historical data.
    - validation_bars: Number of bars for validation.
    - fund: Initial fund amount.
    - feePaid: Transaction fees.
    - optimize_fn: Function to optimize parameters (optional).
    - **strategy_params: Additional parameters for the strategy and optimization function.

    Returns:
    - stats_master: List of results from each walk-forward step.
    - params_results_df: DataFrame containing the parameters used in each step.
    """
    
    stats_master = []
    params_results = []

    # Unpack strategy parameters
    a_range = strategy_params.get('a_range', [])
    o_range = strategy_params.get('o_range', [])
    upper_threshold_range = strategy_params.get('upper_threshold_range', [])
    lower_threshold_range = strategy_params.get('lower_threshold_range', [])
    retrace_u_threshold_range = strategy_params.get('retrace_u_threshold_range', [])
    retrace_l_threshold_range = strategy_params.get('retrace_l_threshold_range', [])
    p = strategy_params.get('p', None)
    q = strategy_params.get('q', None)
    rolling = strategy_params.get('rolling', None)
    num_vol = strategy_params.get('num_vol', None)
    
    for i in range(lookback_bars + warmup_bars, len(data_full) - validation_bars, validation_bars):
        print(f'Processing index: {i}')

        # Define the sample and validation periods
        sample_data = data_full.iloc[i - lookback_bars : i]
        validation_data = data_full.iloc[i - warmup_bars : i + validation_bars]
        
        # Debug information
        print(f'Sample data:\n{sample_data.head()}')
        print(f'Strategy parameters before optimization: {strategy_params}')
        
        # Optimize parameters if optimization function is provided
        if optimize_fn is not None:
            optimize_params_df = optimize_fn(
                a_range=a_range, 
                o_range=o_range,
                upper_threshold_range=upper_threshold_range, 
                lower_threshold_range=lower_threshold_range, 
                retrace_u_threshold_range=retrace_u_threshold_range, 
                retrace_l_threshold_range=retrace_l_threshold_range,
                df=sample_data,
                fund=fund,
                feePaid=feePaid,
                p=p,
                q=q,
                rolling=rolling,
                num_vol=num_vol
            )
            
            metrics_df = pd.json_normalize(optimize_params_df['metrics'])
            optimize_params_df = pd.concat([optimize_params_df.drop(columns='metrics'), metrics_df], axis=1)

            if 'metrics' in optimize_params_df.columns:
                metrics_df = pd.json_normalize(optimize_params_df['metrics'])
                optimize_params_df = pd.concat([optimize_params_df.drop(columns='metrics'), metrics_df], axis=1)
                
                print("Updated Results DataFrame:")
                print(optimize_params_df.head())
                print("Columns:", optimize_params_df.columns)
                
                if 'sharp_ratio' in optimize_params_df.columns:
                    best_params = optimize_params_df.sort_values(by='sharp_ratio', ascending=False).iloc[0]
                    print(f'Best params: {best_params}')
                    strategy_params.update(best_params.to_dict())
                else:
                    print("sharp_ratio column not found in optimize_params_df")
                    print("Using default strategy parameters.")
            else:
                print("Metrics column not found in optimize_params_df")
                print("Using default strategy parameters.")
        
        
        # Run the strategy with the optimized parameters
        result = strategy(validation_data, fund = fund, feePaid=feePaid, **strategy_params)
        print(f'Result: {result}')  # Debug information
        
        if result is not None:
            stats_master.append(result)
        else:
            print(f'Warning: Result is None for index {i}')
        
        # Save parameters used in this step
        params_results.append(strategy_params.copy())
    
    # Convert parameters results to DataFrame
    params_results_df = pd.DataFrame(params_results)
    print(f'Params results DataFrame:\n{params_results_df.head()}')
    
    return stats_master, params_results_df
