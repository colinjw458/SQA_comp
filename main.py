import csp
from csp import ts
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import adapter

sim_df = pd.read_csv("Data/preview.csv")


@csp.node
def portfolio_return(open: ts[float], prior_close: ts[float]) -> ts[float]:
    #Alarm acts as an event trigger
    with csp.alarms():
        event = csp.alarm(float)
    with csp.state():
        s_count = 0
    #Schedules the alarm to trigger in x seconds
    with csp.start():
        csp.schedule_alarm(event, timedelta(seconds=1), True)

    #Once triggered a new alarm must be set
    #At endtime the graph will stop running automatically so no need to worry about infinite loops
    if csp.ticked(event):
        csp.schedule_alarm(event, timedelta(seconds=1), True)
        return (open - prior_close) / prior_close



@csp.graph
def my_graph():
    data = SimulaterEquityData(sim_df)
    print(data)
    port_return = portfolio_return(open, prior_close)
    csp.print("data", port_return)
    

    csp.print('returns', port_return)

#Change realtime to false for testing purposes
if __name__ == '__main__':
    csp.run(my_graph, starttime=datetime.now(timezone.utc), endtime=timedelta(seconds=10), realtime=True)