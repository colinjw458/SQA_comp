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
sim_df = pd.read_csv("his_short.csv", low_memory=False)

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

# Set the number of steps for training (easily changeable)
TRAINING_STEPS = 150

# Create two portfolios with random weights
def create_portfolio():
    weights = np.random.dirichlet(np.ones(len(symbols)))
    return dict(zip(symbols, weights))

portfolio1 = create_portfolio()
portfolio2 = create_portfolio()

# Initialize environments and agents for each symbol in each portfolio
envs1 = {symbol: PortfolioEnv(sim_df[sim_df["symbol"] == symbol], INITIAL_BALANCE * portfolio1[symbol]) for symbol in symbols}
envs2 = {symbol: PortfolioEnv(sim_df[sim_df["symbol"] == symbol], INITIAL_BALANCE * portfolio2[symbol]) for symbol in symbols}
agents1 = {symbol: DQNAgent(state_size=4, action_size=5) for symbol in symbols}
agents2 = {symbol: DQNAgent(state_size=4, action_size=5) for symbol in symbols}

# Prepare data for CSP curves
prices_data = []
returns_data = []
for timestamp in timestamps:
    price_array = np.array([sim_df[(sim_df["ts_event"] == timestamp) & (sim_df["symbol"] == symbol)]["close"].values[0] if len(sim_df[(sim_df["ts_event"] == timestamp) & (sim_df["symbol"] == symbol)]) > 0 else np.nan for symbol in symbols])
    returns_array = np.array([sim_df[(sim_df["ts_event"] == timestamp) & (sim_df["symbol"] == symbol)]["returns"].values[0] if len(sim_df[(sim_df["ts_event"] == timestamp) & (sim_df["symbol"] == symbol)]) > 0 else np.nan for symbol in symbols])
    prices_data.append((timestamp, price_array))
    returns_data.append((timestamp, returns_array))

total_steps = len(timestamps)
progress_bar = tqdm(total=total_steps, desc="Simulation Progress", unit="step")

# Global variables
current_step = 0
is_training = True
best_portfolio = None
best_envs = None
best_agents = None

@csp.node
def update_portfolio(prices: ts[np.ndarray], returns: ts[np.ndarray], portfolio: dict, envs: dict, agents: dict) -> ts[float]:
    global current_step, is_training
    
    current_values = {symbol: envs[symbol].get_portfolio_value() for symbol in symbols}

    if csp.ticked(prices):
        price_array = prices
        returns_array = returns
        for i, symbol in enumerate(symbols):
            if not np.isnan(price_array[i]):
                env = envs[symbol]
                agent = agents[symbol]
                
                state = env._get_state()
                action = agent.act(state)
                next_state, reward, done = env.step(action, price_array[i], returns_array[i])
                
                if is_training:
                    agent.remember(state, action, reward, next_state, done)
                    if len(agent.memory) > BATCH_SIZE:
                        agent.replay(BATCH_SIZE)
                else:
                    # During testing, we use a greedy policy but don't reset epsilon immediately
                    agent.epsilon = max(agent.epsilon * 0.99, 0.01)  # Gradually reduce epsilon
                
                current_values[symbol] = env.get_portfolio_value()

        total_value = sum(current_values.values())
        current_step += 1
        
        if current_step == TRAINING_STEPS:
            is_training = False

        return total_value
    
    return sum(current_values.values())

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

    # Update both portfolios throughout the entire simulation
    portfolio1_value = update_portfolio(prices, returns, portfolio1, envs1, agents1)
    portfolio2_value = update_portfolio(prices, returns, portfolio2, envs2, agents2)
    
    csp.add_graph_output("portfolio1_value", portfolio1_value)
    csp.add_graph_output("portfolio2_value", portfolio2_value)

    # Add progress update
    progress_update = update_progress(progress_ticker)
    csp.add_graph_output("progress", progress_update)

def run_simulation():
    global current_step, is_training, best_portfolio, best_envs, best_agents
    
    # Reset global variables
    current_step = 0
    is_training = True
    best_portfolio = None
    best_envs = None
    best_agents = None

    results = csp.run(
        portfolio_manager_graph,
        starttime=timestamps[0],
        endtime=timestamps[-1],
        realtime=False
    )
    
    # Determine the best portfolio based on training results
    train_value1 = results["portfolio1_value"][TRAINING_STEPS-1][1]
    train_value2 = results["portfolio2_value"][TRAINING_STEPS-1][1]
    
    if train_value1 > train_value2:
        best_portfolio_name = "Portfolio 1"
    else:
        best_portfolio_name = "Portfolio 2"
    
    return results, best_portfolio_name

def plot_results(results, best_portfolio_name):
    plt.figure(figsize=(12, 6))
    
    # Plot full results for both portfolios
    plt.plot(timestamps, [v[1] for v in results["portfolio1_value"]], label="Portfolio 1", alpha=0.7)
    plt.plot(timestamps, [v[1] for v in results["portfolio2_value"]], label="Portfolio 2", alpha=0.7)
    
    # Highlight the chosen portfolio for testing phase
    if best_portfolio_name == "Portfolio 1":
        test_values = [v[1] for v in results["portfolio1_value"][TRAINING_STEPS:]]
    else:
        test_values = [v[1] for v in results["portfolio2_value"][TRAINING_STEPS:]]
    
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
    else:
        test_values = [v[1] for v in results["portfolio2_value"][TRAINING_STEPS:]]

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