import csp
from csp import ts
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import random

sim_df = pd.read_csv("Data/preview.csv")


@csp.node
def portfolio_return(price: ts[float], prior_close: ts[float], delta: float) -> ts[float]:
    #Alarm acts as an event trigger
    with csp.alarms():
        event = csp.alarm(float)
    with csp.state():
        s_count = 0
    #Schedules the alarm to trigger in x seconds
    with csp.start():
        print(delta)
        csp.schedule_alarm(event, timedelta(seconds=delta), True)

    #Once triggered a new alarm must be set
    #At endtime the graph will stop running automatically so no need to worry about infinite loops
    if csp.ticked(event):
        csp.schedule_alarm(event, timedelta(seconds=delta), True)
        return (open - prior_close) / prior_close

@csp.graph(memoize=False)
def my_graph():
    prior_close = 0.0
    for i, row in sim_df[["ts_recv","price"]].iterrows():
        timeNow = datetime.fromisoformat(row["ts_recv"])
        price = csp.const(row["price"])
        if(i == 0):
            prior_close = price
            deltaTemp =  datetime.fromisoformat(sim_df["ts_recv"][i + 1]) - timeNow
            time.sleep(deltaTemp.total_seconds())

        if(len(sim_df["ts_recv"]) == i + 1):
            break
        else:
            deltaTemp =  datetime.fromisoformat(sim_df["ts_recv"][i + 1]) - timeNow
            port_return = portfolio_return(price, prior_close, deltaTemp.total_seconds())
            prior_close = price

        csp.print('returns', port_return)

#Change realtime to false for testing purposes
if __name__ == '__main__':
    csp.run(my_graph, starttime=datetime.now(timezone.utc), endtime=timedelta(seconds=10), realtime=True)