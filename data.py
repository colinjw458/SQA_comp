import databento as db

client = db.Historical("db-FmDMQiWCesyXUGuxgNJUxbqqrwWDy")
data = client.timeseries.get_range(
    dataset = "XNAS.ITCH",
    symbols= ["NVDA", "AAPL", "MSFT"],
    # Historical
    schema = "ohlcv-1s", # prices and volumes in 1s interval
    # Real time
    # schema = "trades", 
    # read in trades as it happens and read into csp to produce the high, low, and volumn in 1s interval
    start = "2024-09-03T09:30:00",
    end = "2024-09-03T10:30:00"
)

# Direct ts output - good for real time data - use adaptor to read in and simulate realtime 
data.replay(print)

# save to df
data.to_csv("HistoricalEquityData.csv")