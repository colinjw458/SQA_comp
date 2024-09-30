import csp
from csp import ts
import pandas as pd
import numpy as np
from datetime import timedelta
from q_learn import PortfolioEnv, DDPGAgent
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import logging

warnings.filterwarnings("ignore", message="The argument 'infer_datetime_format' is deprecated")
logging.basicConfig(level=logging.INFO)

# Read the CSV file
sim_df = pd.read_csv("Data/HistoricalEquityData.csv", low_memory=False)

# Use categorical data types where possible to reduce memory usage
sim_df["symbol"] = sim_df["symbol"].astype("category")
sim_df["publisher_id"] = sim_df["publisher_id"].astype("category")
sim_df["instrument_id"] = sim_df["instrument_id"].astype("category")

# Convert ts_event to datetime and sort
sim_df["ts_event"] = pd.to_datetime(sim_df["ts_event"])
sim_df = sim_df.sort_values("ts_event")

# Get unique symbols and timestamps
symbols = sim_df["symbol"].unique()
timestamps = sim_df["ts_event"].unique()

# Initial balance for each portfolio
INITIAL_BALANCE = 10000

# Set the number of steps for training use ~80% of data for training
TRAINING_STEPS = 800 

# Create three environments with random initial weights
env1 = PortfolioEnv(INITIAL_BALANCE, symbols)
env2 = PortfolioEnv(INITIAL_BALANCE, symbols)
env3 = PortfolioEnv(INITIAL_BALANCE, symbols)

# Create agents for each environment
state_size = 2 * len(symbols) + 1  # price and holdings for each stock + cash
action_size = len(symbols)
agent1 = DDPGAgent(state_size, action_size)
agent2 = DDPGAgent(state_size, action_size)
agent3 = DDPGAgent(state_size, action_size)

# Prepare data for CSP curves
prices_data = []
for timestamp in timestamps:
    price_dict = {symbol: sim_df[(sim_df["ts_event"] == timestamp) & (sim_df["symbol"] == symbol)]["close"].values[0]
                  if len(sim_df[(sim_df["ts_event"] == timestamp) & (sim_df["symbol"] == symbol)]) > 0
                  else np.nan
                  for symbol in symbols}
    prices_data.append((timestamp, price_dict))

total_steps = len(timestamps)
progress_bar = tqdm(total=(total_steps*3 + 812), desc="Simulation Progress", unit="step")

# Global variables
current_step = 0
is_training = True
best_env = None
best_agent = None

@csp.node
def update_portfolio(prices: ts[dict], env: PortfolioEnv, agent: DDPGAgent) -> ts[float]:
    global current_step, is_training, best_env, best_agent
    
    if csp.ticked(prices):
        price_dict = prices
        
        env.update_prices(price_dict)
        
        if not env.is_initialized:
            return env.get_portfolio_value()

        state = env.get_state()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        
        if is_training and current_step >= TRAINING_STEPS:
            is_training = False
        
        if is_training:
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > agent.batch_size:
                agent.train()
        elif env == best_env:
            agent.noise.std_deviation = max(agent.noise.std_deviation * 0.99, 0.01)
        
        current_step += 1

        portfolio_value = env.get_portfolio_value()
        if np.isnan(portfolio_value):
            logging.warning(f"NaN portfolio value detected at step {current_step}")
            portfolio_value = INITIAL_BALANCE  # Use initial balance as fallback

        return portfolio_value
    
    return env.get_portfolio_value()

@csp.node
def update_progress(trigger: ts[bool]) -> ts[None]:
    if csp.ticked(trigger):
        progress_bar.update(1)
    return None

@csp.graph
def portfolio_manager_graph():
    prices = csp.curve(typ=dict, data=prices_data)

    progress_ticker = csp.timer(timedelta(seconds=1))

    portfolio1_value = update_portfolio(prices, env1, agent1)
    portfolio2_value = update_portfolio(prices, env2, agent2)
    portfolio3_value = update_portfolio(prices, env3, agent3)
    
    csp.add_graph_output("portfolio1_value", portfolio1_value)
    csp.add_graph_output("portfolio2_value", portfolio2_value)
    csp.add_graph_output("portfolio3_value", portfolio3_value)

    progress_update = update_progress(progress_ticker)
    csp.add_graph_output("progress", progress_update)

def run_simulation():
    global current_step, is_training, best_env, best_agent
    
    current_step = 0
    is_training = True
    best_env = None
    best_agent = None

    results = csp.run(
        portfolio_manager_graph,
        starttime=timestamps[0],
        endtime=timestamps[-1],
        realtime=False
    )
    
    # Determine the best portfolio based on training results
    train_value1 = results["portfolio1_value"][TRAINING_STEPS-1][1]
    train_value2 = results["portfolio2_value"][TRAINING_STEPS-1][1]
    train_value3 = results["portfolio3_value"][TRAINING_STEPS-1][1]
    
    best_value = max(train_value1, train_value2, train_value3)
    if best_value == train_value1:
        best_portfolio_name = "Portfolio 1"
        best_env, best_agent = env1, agent1
    elif best_value == train_value2:
        best_portfolio_name = "Portfolio 2"
        best_env, best_agent = env2, agent2
    else:
        best_portfolio_name = "Portfolio 3"
        best_env, best_agent = env3, agent3
    
    return results, best_portfolio_name


def plot_results(results, best_portfolio_name, sim_df, symbols):
    plt.figure(figsize=(15, 10))
    
    # Plot portfolio values and individual stock performance
    for i, portfolio_name in enumerate(['Portfolio 1', 'Portfolio 2', 'Portfolio 3']):
        values = [v[1] for v in results[f"portfolio{i+1}_value"]]
        normalized_values = [(v / INITIAL_BALANCE - 1) * 100 for v in values]  # Convert to percentage change
        
        plt.plot(timestamps[:TRAINING_STEPS], normalized_values[:TRAINING_STEPS], 
                 label=f"{portfolio_name} (Training)", alpha=0.7)
        
        if portfolio_name == best_portfolio_name:
            plt.plot(timestamps[TRAINING_STEPS:], normalized_values[TRAINING_STEPS:], 
                     label=f"{portfolio_name} (Testing)", linewidth=2)

    # Plot individual stock values
    for symbol in symbols:
        stock_data = sim_df[sim_df['symbol'] == symbol]
        initial_price = stock_data['close'].iloc[0]
        normalized_prices = [(price / initial_price - 1) * 100 for price in stock_data['close']]
        plt.plot(stock_data['ts_event'], normalized_prices, label=symbol, linestyle='--', alpha=0.5)

    plt.axvline(x=timestamps[TRAINING_STEPS], color='r', linestyle='--', label='Train/Test Split')

    plt.title("Portfolio and Stock Performance (Training and Testing Phases)")
    plt.xlabel("Time")
    plt.ylabel("Percentage Change")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_final_portfolio_composition(best_env):
    portfolio_value = best_env.get_portfolio_value()
    cash_percentage = (best_env.balance / portfolio_value) * 100
    stock_percentages = {symbol: (best_env.shares[symbol] * best_env.last_known_prices[symbol] / portfolio_value) * 100 
                         for symbol in best_env.symbols}

    labels = ['Cash'] + list(stock_percentages.keys())
    sizes = [cash_percentage] + list(stock_percentages.values())

    plt.figure(figsize=(10, 10))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title(f"Final Portfolio Composition")
    plt.show()

if __name__ == "__main__":
    results, best_portfolio_name = run_simulation()
    
    progress_bar.close()

    plot_results(results, best_portfolio_name, sim_df, symbols)
    plot_final_portfolio_composition(best_env)

    print(f"\nInitial Investment: ${INITIAL_BALANCE:.2f}")
    
    test_values = [v[1] for v in results[f"portfolio{['1', '2', '3'][['Portfolio 1', 'Portfolio 2', 'Portfolio 3'].index(best_portfolio_name)]}_value"][TRAINING_STEPS:]]
    
    final_value = test_values[-1]
    total_return = (final_value - INITIAL_BALANCE) / INITIAL_BALANCE
    
    print(f"\nTesting Results:")
    print(f"Final Value: ${final_value:.2f}")
    print(f"Total Return: {total_return:.2%}")

    # Calculate and print Sharpe ratio for the testing phase
    risk_free_rate = 0.02  # Assume 2% risk-free rate
    returns = np.diff(test_values) / test_values[:-1]
    excess_returns = returns - risk_free_rate / 252  # Daily excess returns
    sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns) if len(excess_returns) > 0 else np.nan
    print(f"Sharpe Ratio (Testing Phase): {sharpe_ratio:.4f}")

    print(f"\nBest performing portfolio selected for testing: {best_portfolio_name}")

    # Print final portfolio composition
    print("\nFinal Portfolio Composition:")
    portfolio_value = best_env.get_portfolio_value()
    print(f"Cash: ${best_env.balance:.2f} ({(best_env.balance / portfolio_value) * 100:.2f}%)")
    for symbol in best_env.symbols:
        stock_value = best_env.shares[symbol] * best_env.last_known_prices[symbol]
        print(f"{symbol}: ${stock_value:.2f} ({(stock_value / portfolio_value) * 100:.2f}%)")
