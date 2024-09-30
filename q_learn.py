import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import logging

logging.basicConfig(level=logging.INFO)

class PortfolioEnv:
    def __init__(self, initial_balance, symbols):
        self.initial_balance = initial_balance
        self.symbols = symbols
        self.num_stocks = len(symbols)
        self.last_known_prices = {symbol: None for symbol in symbols}
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.shares = {symbol: 0 for symbol in self.symbols}
        self.portfolio_value = self.initial_balance
        self.latest_prices = {symbol: None for symbol in self.symbols}
        return self.get_state()

    def get_state(self):
        state = [self.balance / self.initial_balance]  # Normalized cash balance
        for symbol in self.symbols:
            price = self.last_known_prices[symbol]
            if price is not None:
                state.append(price / 100)  # Normalized price
                state.append(self.shares[symbol] * price / self.initial_balance)  # Normalized position value
            else:
                state.extend([0, 0])  # If no price is available yet, use zeros
        return np.array(state)

    def step(self, action):
        total_portfolio_value = self.balance + sum(self.shares[s] * self.last_known_prices[s] 
                                                   for s in self.symbols if self.last_known_prices[s] is not None)
        
        for i, symbol in enumerate(self.symbols):
            if self.last_known_prices[symbol] is None:
                continue  # Skip this stock if we don't have any price data yet

            desired_value = total_portfolio_value * (action[i] + 1) / 2  # Convert from [-1, 1] to [0, 1]
            current_value = self.shares[symbol] * self.last_known_prices[symbol]
            value_difference = desired_value - current_value

            if value_difference > 0:  # Buy
                shares_to_buy = value_difference / self.last_known_prices[symbol]
                cost = shares_to_buy * self.last_known_prices[symbol]
                if cost <= self.balance:
                    self.shares[symbol] += shares_to_buy
                    self.balance -= cost
            elif value_difference < 0:  # Sell
                shares_to_sell = -value_difference / self.last_known_prices[symbol]
                if shares_to_sell <= self.shares[symbol]:  # Ensure we don't sell more than we have
                    self.shares[symbol] -= shares_to_sell
                    self.balance += -value_difference

        new_portfolio_value = self.get_portfolio_value()
        reward = (new_portfolio_value - self.portfolio_value) / self.portfolio_value if self.portfolio_value != 0 else 0
        self.portfolio_value = new_portfolio_value

        return self.get_state(), reward, False, {}

    def update_prices(self, price_dict):
        for symbol, price in price_dict.items():
            if price is not None and not np.isnan(price):
                self.last_known_prices[symbol] = price
                self.latest_prices[symbol] = price
            elif symbol in self.last_known_prices:
                self.latest_prices[symbol] = self.last_known_prices[symbol]
        
        missing_prices = [symbol for symbol, price in self.last_known_prices.items() if price is None]
        if missing_prices:
            logging.info(f"Still missing prices for: {', '.join(missing_prices)}")

    def get_portfolio_value(self):
        return self.balance + sum(self.shares[s] * self.last_known_prices[s] 
                                  for s in self.symbols if self.last_known_prices[s] is not None)

    def all_prices_received(self):
        return all(price is not None for price in self.last_known_prices.values())

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class DDPGAgent:
    def __init__(self, state_size, action_size, hidden_size=64):
        self.state_size = state_size
        self.action_size = action_size
        
        self.gamma = 0.99
        self.tau = 0.001
        self.learning_rate = 0.0001  # Reduced learning rate
        
        self.actor = self.build_actor(hidden_size)
        self.critic = self.build_critic(hidden_size)
        self.target_actor = self.build_actor(hidden_size)
        self.target_critic = self.build_critic(hidden_size)
        
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
        self.noise = OUActionNoise(mean=np.zeros(action_size), std_deviation=float(0.1) * np.ones(action_size))  # Reduced noise
        
        self.memory = deque(maxlen=100000)
        self.batch_size = 64

        self.actor_optimizer = keras.optimizers.Adam(self.learning_rate)
        self.critic_optimizer = keras.optimizers.Adam(self.learning_rate)

    def build_actor(self, hidden_size):
        inputs = keras.layers.Input(shape=(self.state_size,))
        x = keras.layers.Dense(hidden_size, activation="relu")(inputs)
        x = keras.layers.Dense(hidden_size, activation="relu")(x)
        outputs = keras.layers.Dense(self.action_size, activation="tanh")(x)
        return keras.Model(inputs, outputs)

    def build_critic(self, hidden_size):
        state_input = keras.layers.Input(shape=(self.state_size,))
        state_out = keras.layers.Dense(16, activation="relu")(state_input)
        state_out = keras.layers.Dense(32, activation="relu")(state_out)

        action_input = keras.layers.Input(shape=(self.action_size,))
        action_out = keras.layers.Dense(32, activation="relu")(action_input)

        concat = keras.layers.Concatenate()([state_out, action_out])

        x = keras.layers.Dense(hidden_size, activation="relu")(concat)
        x = keras.layers.Dense(hidden_size, activation="relu")(x)
        outputs = keras.layers.Dense(1)(x)

        return keras.Model([state_input, action_input], outputs)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.reshape(state, [1, self.state_size])
        state = np.nan_to_num(state, nan=0.0)  # Replace NaN with 0
        action = self.actor.predict(state)[0]
        action += self.noise()
        return np.clip(action, -1, 1)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        indices = np.random.choice(len(self.memory), size=self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[idx] for idx in indices])

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Replace NaN values with 0 and convert to tensors
        states = tf.convert_to_tensor(np.nan_to_num(states, nan=0.0), dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.nan_to_num(next_states, nan=0.0), dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            target_q_values = self.target_critic([next_states, target_actions])
            target_q = rewards + self.gamma * target_q_values * (1 - dones)
            critic_value = self.critic([states, actions])
            critic_loss = tf.reduce_mean(tf.square(target_q - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor(states)
            critic_value = self.critic([states, actions])
            actor_loss = -tf.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        self.update_target(self.target_actor.variables, self.actor.variables)
        self.update_target(self.target_critic.variables, self.critic.variables)

    def update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(self.tau * b + (1 - self.tau) * a)

    def save(self, filename):
        self.actor.save(filename + "_actor.h5")
        self.critic.save(filename + "_critic.h5")

    def load(self, filename):
        self.actor = keras.models.load_model(filename + "_actor.h5")
        self.critic = keras.models.load_model(filename + "_critic.h5")
        self.target_actor = keras.models.load_model(filename + "_actor.h5")
        self.target_critic = keras.models.load_model(filename + "_critic.h5")
