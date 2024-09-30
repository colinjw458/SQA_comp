import pandas as pd

### data
def synchronize_data(df, symbols, freq='1min'):
    """
    Synchronize the DataFrame to have entries for all symbols at each timestamp.
    
    :param df: Original DataFrame with columns ['ts_event', 'open', 'high', 'low', 'close', 'volume', 'symbol']
    :param symbols: List of unique stock symbols.
    :param freq: Frequency for resampling (default '1mim' for 1 minute).
    :return: Synchronized DataFrame with flattened columns.
    """
    # Convert 'ts_event' to datetime if not already
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    
    # Set 'ts_event' as index
    df = df.set_index('ts_event')
    
    # Create a date range covering the entire period at 1-minute intervals
    full_time_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    
    # Initialize an empty list to hold DataFrames for each symbol
    symbol_dfs = []
    
    for symbol in symbols:
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df = symbol_df.drop(columns=['symbol'])  # Remove 'symbol' column as it's now redundant
        # Reindex to include all timestamps, forward fill missing data
        symbol_df = symbol_df.reindex(full_time_index, method='ffill')
        # If there are still missing values at the start, fill them with the first available data
        symbol_df = symbol_df.fillna(method='bfill')
        # Add a suffix to column names to indicate the symbol
        symbol_df = symbol_df.add_suffix(f'_{symbol}')
        symbol_dfs.append(symbol_df)
    
    # Concatenate all symbol DataFrames along columns
    synchronized_df = pd.concat(symbol_dfs, axis=1)
    
    # Reset index to make 'ts_event' a column again
    synchronized_df = synchronized_df.reset_index().rename(columns={'index': 'ts_event'})
    
    return synchronized_df