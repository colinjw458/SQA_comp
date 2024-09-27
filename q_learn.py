import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random

# Q-learning parameters
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000
BATCH_SIZE = 32

# Portfolio parameters
INITIAL_BALANCE = 100000
N_ACTIONS = 3  # Buy, Hold, Sell

class PortfolioEnv:
    def __init__(self, data):
        self.data = data
        self.reset()

    def reset(self):
        self.balance = INITIAL_BALANCE
        self.shares = 0
        self.current_step = 0
        return self._get_state()

    def step(self, action):
        current_price = self.data.iloc[self.current_step]["price"]
        
        if action == 0:  # Buy
            shares_to_buy = min(self.balance // current_price, 1)  # Buy 1 share at a time
            self.shares += shares_to_buy
            self.balance -= shares_to_buy * current_price
        elif action == 2:  # Sell
            if self.shares > 0:
                self.balance += current_price
                self.shares -= 1

        self.current_step += 1
        done = self.current_step == len(self.data) - 1
        next_state = self._get_state()
        reward = self._calculate_reward()
        
        return next_state, reward, done

    def _get_state(self):
        return np.array([
            self.balance / INITIAL_BALANCE,
            self.shares * self.data.iloc[self.current_step]["price"] / INITIAL_BALANCE,
            self.data.iloc[self.current_step]["returns"]
        ])

    def _calculate_reward(self):
        portfolio_value = self.balance + self.shares * self.data.iloc[self.current_step]["price"]
        return (portfolio_value - INITIAL_BALANCE) / INITIAL_BALANCE

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
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