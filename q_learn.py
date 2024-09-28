import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Q-learning parameters
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000
BATCH_SIZE = 32

# Portfolio parameters
N_ACTIONS = 5  # Sell All, Sell Half, Hold, Buy Half, Buy All

class PortfolioEnv:
    def __init__(self, data, initial_balance):
        self.data = data
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.shares = 0
        self.current_step = 0
        self.portfolio_values = [self.initial_balance]
        return self._get_state()

    def step(self, action):
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0, True

        current_price = self.data.iloc[self.current_step]["close"]
        
        if action == 0:  # Sell All
            self.balance += self.shares * current_price
            self.shares = 0
        elif action == 1:  # Sell Half
            shares_to_sell = self.shares // 2
            self.balance += shares_to_sell * current_price
            self.shares -= shares_to_sell
        elif action == 3:  # Buy Half
            available_balance = self.balance // 2
            shares_to_buy = available_balance // current_price
            self.shares += shares_to_buy
            self.balance -= shares_to_buy * current_price
        elif action == 4:  # Buy All
            shares_to_buy = self.balance // current_price
            self.shares += shares_to_buy
            self.balance -= shares_to_buy * current_price

        self.current_step += 1
        next_state = self._get_state()
        reward = self._calculate_reward()
        done = self.current_step == len(self.data) - 1
        
        self.portfolio_values.append(self.get_portfolio_value())
        
        return next_state, reward, done

    def _get_state(self):
        if self.current_step >= len(self.data):
            return np.array([
                self.balance / self.initial_balance,
                0,
                0,
                0
            ])

        return np.array([
            self.balance / self.initial_balance,
            self.shares * self.data.iloc[self.current_step]["close"] / self.initial_balance,
            self.data.iloc[self.current_step]["returns"] if self.current_step > 0 else 0,
            self._calculate_sharpe_ratio()
        ])

    def _calculate_reward(self):
        if self.current_step >= len(self.data):
            return 0

        portfolio_return = (self.get_portfolio_value() - self.initial_balance) / self.initial_balance
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Combine portfolio return and Sharpe ratio in the reward
        reward = portfolio_return + 0.5 * sharpe_ratio
        
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
        if self.current_step >= len(self.data):
            return self.balance

        return self.balance + self.shares * self.data.iloc[self.current_step]["close"]

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_size,)),
            keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + GAMMA * np.amax(self.model.predict(next_state.reshape(1, -1), verbose=0)[0])
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY