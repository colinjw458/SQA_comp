import csp
from csp import ts
from datetime import datetime, timezone


@csp.node
def spread(bid: ts[float], ask: ts[float]) -> ts[float]:
    if csp.valid(bid, ask):
        return ask - bid


@csp.graph
def my_graph():
    bid = csp.const(1.0)
    ask = csp.const(2.0)
    s = spread(bid, ask)

    csp.print('spread', s)
    csp.print('bid', bid)
    csp.print('ask', ask)


if __name__ == '__main__':
    csp.run(my_graph, starttime=datetime.now(timezone.utc))