import csp
from csp import ts
import pandas as pd
import numpy as np
from datetime import timedelta
from q_learn import PortfolioEnv, DQNAgent, BATCH_SIZE
import matplotlib.pyplot as plt
from tqdm import tqdm

# Read the CSV file
sim_df = pd.read_csv("his_short.csv")

# Convert ts_event to datetime and sort
sim_df["ts_event"] = pd.to_datetime(sim_df["ts_event"])
sim_df = sim_df.sort_values("ts_event")

# Calculate returns
sim_df["returns"] = sim_df.groupby("symbol")["close"].pct_change()

# Get unique symbols and timestamps
symbols = sim_df["symbol"].unique()
timestamps = sim_df["ts_event"].unique()

# Initial balance for each portfolio
INITIAL_BALANCE = 10000

# Create two portfolios with random weights
def create_portfolio():
    weights = np.random.dirichlet(np.ones(len(symbols)))
    return dict(zip(symbols, weights))

portfolio1 = create_portfolio()
portfolio2 = create_portfolio()

# Initialize environments and agents for each symbol in each portfolio
envs1 = {symbol: PortfolioEnv(sim_df[sim_df["symbol"] == symbol], INITIAL_BALANCE * portfolio1[symbol]) for symbol in symbols}
envs2 = {symbol: PortfolioEnv(sim_df[sim_df["symbol"] == symbol], INITIAL_BALANCE * portfolio2[symbol]) for symbol in symbols}
agents1 = {symbol: DQNAgent(state_size=3, action_size=3) for symbol in symbols}
agents2 = {symbol: DQNAgent(state_size=3, action_size=3) for symbol in symbols}

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

@csp.node
def update_portfolio(prices: ts[np.ndarray], returns: ts[np.ndarray], portfolio: dict, envs: dict, agents: dict) -> ts[float]:
    current_values = {symbol: INITIAL_BALANCE * weight for symbol, weight in portfolio.items()}

    if csp.ticked(prices):
        price_array = prices
        returns_array = returns
        for i, symbol in enumerate(symbols):
            if not np.isnan(price_array[i]):
                env = envs[symbol]
                agent = agents[symbol]
                
                state = env._get_state()
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                if len(agent.memory) > BATCH_SIZE:
                    agent.replay(BATCH_SIZE)
                
                current_values[symbol] = env.get_portfolio_value()

        total_value = sum(current_values.values())
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

    portfolio1_value = update_portfolio(prices, returns, portfolio1, envs1, agents1)
    portfolio2_value = update_portfolio(prices, returns, portfolio2, envs2, agents2)

    # Add progress update node
    progress_update = update_progress(progress_ticker)

    csp.add_graph_output("portfolio1", portfolio1_value)
    csp.add_graph_output("portfolio2", portfolio2_value)
    csp.add_graph_output("progress", progress_update)

def plot_results(results):
    plt.figure(figsize=(12, 6))
    for portfolio, values in results.items():
        if portfolio != "progress":  # Skip plotting progress data
            plt.plot([v[0] for v in values], [v[1] for v in values], label=portfolio)
    plt.title("Portfolio Performance Comparison")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Create a progress bar

    results = csp.run(
        portfolio_manager_graph,
        starttime=timestamps[0],
        endtime=timestamps[-1],
        realtime=False
    )
    
    # Close the progress bar
    progress_bar.close()

    # Plot results
    plot_results(results)

    # Print final portfolio values and returns
    print("\nInitial Investment for each portfolio: ${:.2f}".format(INITIAL_BALANCE))
    for portfolio, values in results.items():
        if portfolio != "progress":  # Skip printing progress data
            final_value = values[-1][1]
            total_return = (final_value - INITIAL_BALANCE) / INITIAL_BALANCE
            print(f"\n{portfolio}:")
            print(f"Final Value: ${final_value:.2f}")
            print(f"Total Return: {total_return:.2%}")
            print("Weights:")
            weights = portfolio1 if portfolio == "portfolio1" else portfolio2
            for symbol, weight in weights.items():
                print(f"  {symbol}: {weight:.2%}")