import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import pandas as pd

# Q-learning parameters
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.05
LEARNING_RATE = 0.005
MEMORY_SIZE = 20000
BATCH_SIZE = 64


# Portfolio parameters
N_ACTIONS = 5  # Sell All, Sell Half, Hold, Buy Half, Buy All


class PortfolioEnv:
    def __init__(self, data, initial_balance, symbols):
        self.data = data
        self.initial_balance = initial_balance
        self.symbols = symbols
        self.transaction_cost = 0.000  # transaction cost if you want to see how poorly the agent performs with it
        self.latest_prices = {symbol: None for symbol in symbols}
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.shares = {symbol: 0 for symbol in self.symbols}
        self.current_step = 0
        self.portfolio_values = [self.initial_balance]
        self.initialized = False
        self.previous_portfolio_value = self.initial_balance
        self.latest_prices = {symbol: None for symbol in self.symbols}
        return self._get_state()

    def initialize_portfolio(self, initial_prices):
        if not self.initialized:
            valid_prices = [(symbol, price) for symbol, price in zip(self.symbols, initial_prices) if not np.isnan(price)]
            if len(valid_prices) > 0:
                weights = np.random.dirichlet(np.ones(len(valid_prices)))
                for i, (symbol, price) in enumerate(valid_prices):
                    self.latest_prices[symbol] = price
                    weight = weights[i]
                    shares_to_buy = int((weight * self.initial_balance) / price)
                    cost = shares_to_buy * price * (1 + self.transaction_cost)
                    if cost <= self.balance:
                        self.shares[symbol] = shares_to_buy
                        self.balance -= cost
            self.initialized = True

    def step(self, action, current_prices, current_returns):
        if not self.initialized:
            self.initialize_portfolio(current_prices)
            return self._get_state(current_prices, current_returns), 0, False

        if self.current_step >= len(self.data) // len(self.symbols) - 1:
            return self._get_state(current_prices, current_returns), 0, True
        
        for i, symbol in enumerate(self.symbols):
            if not np.isnan(current_prices[i]):
                self.latest_prices[symbol] = current_prices[i]
            
            if self.latest_prices[symbol] is None:
                continue  # Skip this stock if we don't have any price data yet
            
            price = self.latest_prices[symbol]
            if action[i] == 0:  # Sell All
                sell_value = self.shares[symbol] * price
                self.balance += sell_value * (1 - self.transaction_cost)
                self.shares[symbol] = 0
            elif action[i] == 1:  # Sell Half
                shares_to_sell = self.shares[symbol] // 2
                sell_value = shares_to_sell * price
                self.balance += sell_value * (1 - self.transaction_cost)
                self.shares[symbol] -= shares_to_sell
            elif action[i] == 3:  # Buy Half
                available_balance = self.balance // (2 * len(self.symbols))
                shares_to_buy = int(available_balance / price)
                buy_cost = shares_to_buy * price * (1 + self.transaction_cost)
                if buy_cost <= self.balance:
                    self.shares[symbol] += shares_to_buy
                    self.balance -= buy_cost
            elif action[i] == 4:  # Buy All
                available_balance = self.balance // len(self.symbols)
                shares_to_buy = int(available_balance / price)
                buy_cost = shares_to_buy * price * (1 + self.transaction_cost)
                if buy_cost <= self.balance:
                    self.shares[symbol] += shares_to_buy
                    self.balance -= buy_cost

        self.current_step += 1
        next_state = self._get_state(current_prices, current_returns)
        reward = self._calculate_reward()
        done = self.current_step == len(self.data) // len(self.symbols) - 1
        
        current_value = self.get_portfolio_value()
        self.portfolio_values.append(current_value)
        
        return next_state, reward, done

    def _get_state(self, current_prices=None, current_returns=None):
        if not self.initialized:
            return np.zeros(3 * len(self.symbols) + 1)

        if current_prices is None or current_returns is None:
            current_prices = []
            current_returns = []
            for symbol in self.symbols:
                symbol_data = self.data[self.data['symbol'] == symbol]
                if self.current_step < len(symbol_data):
                    price = symbol_data['close'].iloc[self.current_step]
                    if not np.isnan(price):
                        self.latest_prices[symbol] = price
                    current_prices.append(self.latest_prices[symbol])
                    if self.current_step > 0:
                        current_returns.append(symbol_data['returns'].iloc[self.current_step])
                    else:
                        current_returns.append(0)
                else:
                    current_prices.append(self.latest_prices[symbol])
                    current_returns.append(0)

        state = []
        for i, symbol in enumerate(self.symbols):
            price = self.latest_prices[symbol]
            state.extend([
                self.balance / self.initial_balance,
                self.shares[symbol] * price / self.initial_balance if price is not None else 0,
                current_returns[i] if not np.isnan(current_returns[i]) else 0,
            ])
        state.append(self._calculate_sharpe_ratio())
        return np.array(state)

    def _calculate_reward(self):
        current_value = self.get_portfolio_value()
        portfolio_return = (current_value - self.previous_portfolio_value) / self.previous_portfolio_value
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        reward = portfolio_return + 0.5 * sharpe_ratio

        if reward < 0:
            reward *= 1.5  # Penalize losses more severely
        
        self.previous_portfolio_value = current_value
        return reward

    def _calculate_sharpe_ratio(self):
        if len(self.portfolio_values) < 2:
            return 0
        
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        if len(returns) < 2:
            return 0
        
        average_return = np.mean(returns)
        return_std = np.std(returns)
        
        if return_std == 0:
            return 0
        
        sharpe_ratio = np.sqrt(252) * average_return / return_std  # Annualized Sharpe ratio
        return sharpe_ratio

    def get_portfolio_value(self):
        return self.balance + sum(self.shares[symbol] * price 
                                  for symbol, price in self.latest_prices.items() 
                                  if price is not None)

    def update_prices(self, current_prices):
        for i, symbol in enumerate(self.symbols):
            if not np.isnan(current_prices[i]):
                self.latest_prices[symbol] = current_prices[i]

class DQNAgent:
    def __init__(self, state_size, action_size, n_stocks):
        self.state_size = state_size
        self.action_size = action_size
        self.n_stocks = n_stocks
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_size,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(self.action_size * self.n_stocks, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size, size=self.n_stocks)
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.array([np.argmax(act_values[0][i*self.action_size:(i+1)*self.action_size]) for i in range(self.n_stocks)])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_q_values = self.model.predict(next_state.reshape(1, -1), verbose=0)[0]
                target = reward + GAMMA * sum([np.amax(next_q_values[i*self.action_size:(i+1)*self.action_size]) for i in range(self.n_stocks)])
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            for i in range(self.n_stocks):
                target_f[0][i*self.action_size + action[i]] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY