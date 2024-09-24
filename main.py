import csp
from csp import ts
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

sim_df = pd.read_csv("Data/preview.csv")

# timedelta captures the difference of time for each row
sim_df["ts_recv"] = pd.to_datetime(sim_df["ts_recv"])
sim_df["timedelta"] = sim_df["ts_recv"].diff().shift(-1)
sim_df["timedelta"].iloc[-1] = pd.Timedelta(seconds=0)
print(sim_df)


@csp.node
def portfolio_return(df: pd.DataFrame) -> ts[float]:
    #Alarm acts as an event trigger
    with csp.alarms():
        event = csp.alarm(float)

    with csp.state():
        s = 0

    #Schedules the alarm to trigger in x seconds
    with csp.start():
        delta = df["timedelta"].iloc[s].total_seconds()
        csp.schedule_alarm(event, timedelta(seconds=delta), True)

    #Once triggered a new alarm must be set
    #At endtime the graph will stop running automatically so no need to worry about infinite loops
    if csp.ticked(event):
        s+=1
        if(s + 1 == len(df)):
            return (df["price"][s] - df["price"][s-1]) / df["price"][s-1]
            
        delta = df["timedelta"].iloc[s].total_seconds()
        csp.schedule_alarm(event, timedelta(seconds=delta), True)
        return (df["price"][s] - df["price"][s-1]) / df["price"][s-1]

@csp.graph(memoize=False)
def my_graph():
    port_return = portfolio_return(sim_df[["timedelta","price"]])
    csp.print('returns', port_return)

#Change realtime to false for testing purposes
if __name__ == '__main__':
    csp.run(my_graph, starttime=sim_df["ts_recv"].iloc[0], endtime=sim_df["ts_recv"].iloc[-1], realtime=True)