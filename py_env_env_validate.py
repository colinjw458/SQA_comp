# validate_environment.py

import pandas as pd
from py_env_env import TradingEnvironment
from tf_agents.environments import utils
from py_env_preprocess import synchronize_data

def verify_environment():
    # Load your data
    df = pd.read_csv('Data/HistoricalEquityData_m.csv')  # Ensure this file exists and is correctly formatted

    # List of unique symbols
    symbols = df['symbol'].unique().tolist()

    # Synchronize the DataFrame without filling missing data
    df_synchronized = synchronize_data(df, symbols, freq='1min')

    # Initialize the environment
    environment = TradingEnvironment(df_synchronized, symbols, initial_balance=100000, max_steps=68)

    # Validate the environment
    # try:
    utils.validate_py_environment(environment, episodes=5)
    #     print("Environment successfully validated!")
    # except Exception as e:
    #     print(f"Environment validation failed: {e}")

if __name__ == "__main__":
    verify_environment()