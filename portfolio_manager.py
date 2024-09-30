import csp
from csp import ts
import pandas as pd
import numpy as np
from datetime import timedelta
from q_learn import PortfolioEnv, DQNAgent, BATCH_SIZE
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", message="Not memoizing output of.*: unhashable type: 'numpy.ndarray'")

# Read the CSV file
sim_df = pd.read_csv("HistoricalEquityData.csv", low_memory=False)

# Use categorical data types where possible to reduce memory usage
sim_df["symbol"] = sim_df["symbol"].astype("category")
sim_df["publisher_id"] = sim_df["publisher_id"].astype("category")
sim_df["instrument_id"] = sim_df["instrument_id"].astype("category")

# Convert ts_event to datetime and sort
sim_df["ts_event"] = pd.to_datetime(sim_df["ts_event"], errors="coerce", infer_datetime_format=True)
sim_df = sim_df.sort_values("ts_event")

# Calculate returns for each stock separately
sim_df["returns"] = sim_df.groupby("symbol")["close"].pct_change()

# Get unique symbols and timestamps
symbols = sim_df["symbol"].unique()
timestamps = sim_df["ts_event"].unique()

# Initial balance for each portfolio
INITIAL_BALANCE = 10000

# Set the number of steps for training 
TRAINING_STEPS = 500

# Create three environments with random initial weights
env1 = PortfolioEnv(sim_df, INITIAL_BALANCE, symbols)
env2 = PortfolioEnv(sim_df, INITIAL_BALANCE, symbols)
env3 = PortfolioEnv(sim_df, INITIAL_BALANCE, symbols)

# Create agents for each environment
state_size = 3 * len(symbols) + 1  # 3 features per stock + 1 for Sharpe ratio
action_size = 5
agent1 = DQNAgent(state_size, action_size, len(symbols))
agent2 = DQNAgent(state_size, action_size, len(symbols))
agent3 = DQNAgent(state_size, action_size, len(symbols))

# Prepare data for CSP curves
prices_data = []
returns_data = []
for timestamp in timestamps:
    price_array = np.array([
        sim_df[(sim_df["ts_event"] == timestamp) & (sim_df["symbol"] == symbol)]["close"].values[0]
        if len(sim_df[(sim_df["ts_event"] == timestamp) & (sim_df["symbol"] == symbol)]) > 0
        else np.nan
        for symbol in symbols
    ])
    returns_array = np.array([
        sim_df[(sim_df["ts_event"] == timestamp) & (sim_df["symbol"] == symbol)]["returns"].values[0]
        if len(sim_df[(sim_df["ts_event"] == timestamp) & (sim_df["symbol"] == symbol)]) > 0
        else np.nan
        for symbol in symbols
    ])
    prices_data.append((timestamp, price_array))
    returns_data.append((timestamp, returns_array))

total_steps = len(timestamps)
progress_bar = tqdm(total=total_steps, desc="Simulation Progress", unit="step")

# Global variables
current_step = 0
is_training = True
best_env = None
best_agent = None

@csp.node
def update_portfolio(prices: ts[np.ndarray], returns: ts[np.ndarray], env: PortfolioEnv, agent: DQNAgent) -> ts[float]:
    global current_step, is_training, best_env, best_agent
    
    if csp.ticked(prices):
        price_array = prices
        returns_array = returns
        
        env.update_prices(price_array)  # Update the latest prices in the environment
        
        if not env.initialized and not np.isnan(price_array).all():
            env.initialize_portfolio(price_array)
            return env.get_portfolio_value()

        if env.initialized:
            state = env._get_state(price_array, returns_array)
            action = agent.act(state)
            next_state, reward, done = env.step(action, price_array, returns_array)
            
            if is_training:
                agent.remember(state, action, reward, next_state, done)
                if len(agent.memory) > BATCH_SIZE:
                    agent.replay(BATCH_SIZE)
            elif env == best_env:
                agent.epsilon = max(agent.epsilon * 0.99, 0.01)  # Gradually reduce epsilon
            
            current_step += 1
            
            if current_step == TRAINING_STEPS:
                is_training = False

        return env.get_portfolio_value()
    
    return env.get_portfolio_value()

@csp.node
def update_progress(trigger: ts[bool]) -> ts[None]:
    if csp.ticked(trigger):
        progress_bar.update(1)
    return None

@csp.graph
def portfolio_manager_graph():
    prices = csp.curve(typ=np.ndarray, data=prices_data)
    returns = csp.curve(typ=np.ndarray, data=returns_data)

    # Create a ticker for progress updates
    progress_ticker = csp.timer(timedelta(seconds=1))

    # Update all portfolios throughout the entire simulation
    portfolio1_value = update_portfolio(prices, returns, env1, agent1)
    portfolio2_value = update_portfolio(prices, returns, env2, agent2)
    portfolio3_value = update_portfolio(prices, returns, env3, agent3)
    
    csp.add_graph_output("portfolio1_value", portfolio1_value)
    csp.add_graph_output("portfolio2_value", portfolio2_value)
    csp.add_graph_output("portfolio3_value", portfolio3_value)

    # Add progress update
    progress_update = update_progress(progress_ticker)
    csp.add_graph_output("progress", progress_update)

def run_simulation():
    global current_step, is_training, best_env, best_agent
    
    # Reset global variables
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

def plot_results(results, best_portfolio_name):
    plt.figure(figsize=(12, 6))
    
    # Plot full results for all portfolios during training
    plt.plot(timestamps[:TRAINING_STEPS], [v[1] for v in results["portfolio1_value"][:TRAINING_STEPS]], label="Portfolio 1 (Training)", alpha=0.7)
    plt.plot(timestamps[:TRAINING_STEPS], [v[1] for v in results["portfolio2_value"][:TRAINING_STEPS]], label="Portfolio 2 (Training)", alpha=0.7)
    plt.plot(timestamps[:TRAINING_STEPS], [v[1] for v in results["portfolio3_value"][:TRAINING_STEPS]], label="Portfolio 3 (Training)", alpha=0.7)
    
    # Plot only the best portfolio for testing phase
    if best_portfolio_name == "Portfolio 1":
        test_values = [v[1] for v in results["portfolio1_value"][TRAINING_STEPS:]]
    elif best_portfolio_name == "Portfolio 2":
        test_values = [v[1] for v in results["portfolio2_value"][TRAINING_STEPS:]]
    else:
        test_values = [v[1] for v in results["portfolio3_value"][TRAINING_STEPS:]]
    
    plt.plot(timestamps[TRAINING_STEPS:], test_values, label=f"{best_portfolio_name} (Testing)", linewidth=2)

    plt.axvline(x=timestamps[TRAINING_STEPS], color='r', linestyle='--', label='Train/Test Split')

    plt.title("Portfolio Performance (Training and Testing Phases)")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    results, best_portfolio_name = run_simulation()
    
    # Close the progress bar
    progress_bar.close()

    # Plot results
    plot_results(results, best_portfolio_name)

    # Print final portfolio values and returns
    print("\nInitial Investment: ${:.2f}".format(INITIAL_BALANCE))
    
    if best_portfolio_name == "Portfolio 1":
        test_values = [v[1] for v in results["portfolio1_value"][TRAINING_STEPS:]]
    elif best_portfolio_name == "Portfolio 2":
        test_values = [v[1] for v in results["portfolio2_value"][TRAINING_STEPS:]]
    else:
        test_values = [v[1] for v in results["portfolio3_value"][TRAINING_STEPS:]]

    final_value = test_values[-1]
    total_return = (final_value - INITIAL_BALANCE) / INITIAL_BALANCE
    
    print(f"\nTesting Results:")
    print(f"Final Value: ${final_value:.2f}")
    print(f"Total Return: {total_return:.2%}")

    # Calculate and print Sharpe ratio for the testing phase
    risk_free_rate = 0.02  # Assume 2% risk-free rate
    returns = np.diff(test_values) / test_values[:-1]
    excess_returns = returns - risk_free_rate / 252  # Daily excess returns
    sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    print(f"\nSharpe Ratio (Testing Phase): {sharpe_ratio:.4f}")

    # Print which portfolio was selected for testing
    print(f"\nBest performing portfolio selected for testing: {best_portfolio_name}")