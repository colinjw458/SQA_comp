import csp
from csp import ts
import pandas as pd
from datetime import timedelta
from q_learn import PortfolioEnv, DQNAgent

# Read the CSV file
sim_df = pd.read_csv("HistoricalEquityData.csv")

# Convert ts_event to datetime and sort
sim_df["ts_event"] = pd.to_datetime(sim_df["ts_event"])
sim_df = sim_df.sort_values("ts_event")

# Calculate time deltas and returns
sim_df["timedelta"] = sim_df["ts_event"].diff().shift(-1)
sim_df["timedelta"].iloc[-1] = pd.Timedelta(seconds=0)
sim_df["returns"] = sim_df["close"].pct_change()

# Initialize environment and agent
env = PortfolioEnv(sim_df)
agent = DQNAgent(state_size=3, action_size=3)

@csp.node
def portfolio_manager(df: pd.DataFrame) -> ts[float]:
    with csp.alarms():
        event = csp.alarm(float)

    with csp.state():
        s = 0

    with csp.start():
        delta = df["timedelta"].iloc[s].total_seconds()
        csp.schedule_alarm(event, timedelta(seconds=delta), True)

    if csp.ticked(event):
        s += 1
        if s + 1 == len(df):
            return df["returns"].iloc[s]
        
        # Get current state and action
        state = env._get_state()
        action = agent.act(state)
        
        # Take action and observe result
        next_state, reward, done = env.step(action)
        
        # Remember the transition
        agent.remember(state, action, reward, next_state, done)
        
        # Train the agent
        if len(agent.memory) > BATCH_SIZE:
            agent.replay(BATCH_SIZE)
        
        delta = df["timedelta"].iloc[s].total_seconds()
        csp.schedule_alarm(event, timedelta(seconds=delta), True)
        return df["returns"].iloc[s]

@csp.graph
def my_graph():
    returns = portfolio_manager(sim_df)
    csp.print('returns', returns)

if __name__ == '__main__':
    csp.run(my_graph, starttime=sim_df["ts_event"].iloc[0], endtime=sim_df["ts_event"].iloc[-1], realtime=True)
