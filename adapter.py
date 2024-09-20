import csp
from csp.impl.pulladapter import PullInputAdapter
from csp.impl.wiring import py_pull_adapter_def
import pandas as pd
from datetime import datetime, timedelta, timezone

class EquityDataAdapter(PullInputAdapter):
    def __init__(self, df):
        self.df = df
        # Read entire csv or row
        #self.time = df["ts_recv"]
        #self.price = df["price"]
        
        super().__init__()
        # What's super

    def start(self, starttime, endtime):
        print("EquityDataAdapter::start")
        super().start(starttime, endtime)
        
    def stop(self):
        print("EquityAdapter::stop")

    def next(self):
        """
        Return tuple of datetime and price, or None if no more data is available
        """
        # if time < end time
        # time = ts_recv
        # price = price at time i
        # return time, price)
        return None
        

SimulaterEquityData = py_pull_adapter_def(
    "SimulaterEquityData",
    EquityDataAdapter,
    csp.ts[float],
    df = str
)