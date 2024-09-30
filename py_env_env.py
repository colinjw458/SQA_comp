# RAW CODE


import numpy as np
import pandas as pd
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import logging
import tensorflow as tf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingEnvironment(py_environment.PyEnvironment):
    def __init__(self, df_pivoted, symbols, initial_balance=100000, max_steps=500):

        super(TradingEnvironment, self).__init__()
        
        self.df = df_pivoted.reset_index(drop=True)  # Ensure continuous indexing
        self.max_index = self.df.shape[0]
        start_point = (np.random.choice(np.arange(3,self.max_index - max_steps))//3) *3
        end_point = start_point + max_steps//3 *3

        self.df = self.df.loc[start_point:end_point+2].reset_index(drop=True)
        self.symbols = symbols
        self.num_stocks = len(symbols)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_step = 0
        self.done = False
        self.portfolio_value = initial_balance
        self.holdings = {symbol: 0 for symbol in self.symbols}
        self.max_steps = max_steps  
        
        
        # Each action represents the proportion to buy (positive) or sell (negative)
        # Values range from -1 to 1
        #### CONTINUOUS
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(self.num_stocks,),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name='action'
        )
        
        obs_dim = 1 + self.num_stocks * 3
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(obs_dim,),
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name='observation'
        )
        
        self._normalize_data()
    
    def _normalize_data(self):
        """
        Normalizes 'close' and 'volume' for each stock to have mean 0 and std 1.
        """
        for symbol in self.symbols:
            close_col = f'close_{symbol}'
            volume_col = f'volume_{symbol}'
            self.df[f'close_norm_{symbol}'] = (self.df[close_col] - self.df[close_col].mean()) / (self.df[close_col].std() + 1e-8)
            self.df[f'volume_norm_{symbol}'] = (self.df[volume_col] - self.df[volume_col].mean()) / (self.df[volume_col].std() + 1e-8)
    
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _get_state(self):
        if self.current_step >= len(self.df):
            return np.zeros(self._observation_spec.shape, dtype=np.float32)
        
        state = [self.balance]  # Start with current balance
        current_data = self.df.iloc[self.current_step]
        
        for symbol in self.symbols:
            state.append(current_data[f'close_norm_{symbol}'])
            state.append(current_data[f'volume_norm_{symbol}'])
            state.append(self.holdings[symbol] / (self.balance + 1)) 
        
        return np.array(state, dtype=np.float32)
    
    def _calculate_portfolio_value(self):
        portfolio_value = self.balance
        for symbol in self.symbols:
            close_price = self.df.iloc[self.current_step][f'close_{symbol}']
            portfolio_value += self.holdings[symbol] * close_price
        return portfolio_value
    
    def _execute_action(self, action_array):
        current_data = self.df.iloc[self.current_step]
        logger.info(f"Step {self.current_step}: Executing actions {action_array}")

        for idx, symbol in enumerate(self.symbols):
            act = action_array[idx]
            close_price = self.df.iloc[self.current_step][f'close_{symbol}']
            
            if act > 0:  # Buy proportion of available balance
                # Calculate the amount to invest based on the action proportion
                invest_amount = act * self.balance / self.num_stocks  # Divide by number of stocks
                shares_to_buy = invest_amount / close_price
                if shares_to_buy > 0.25:
                    cost = shares_to_buy * close_price
                    self.balance -= cost
                    self.holdings[symbol] += shares_to_buy
                    logger.info(f"Bought {shares_to_buy} shares of {symbol} at {close_price} per share. Cost: {cost:.2f}")
                else:
                    logger.info(f"Insufficient balance to buy shares of {symbol}. Required: {invest_amount:.2f} for {shares_to_buy} shares, Available: {self.balance:.2f}")
            elif act < 0:  # Sell proportion of holdings
                sell_proportion = -act  # Make it positive
                shares_to_sell = sell_proportion * self.holdings[symbol]
                if shares_to_sell > 0.25:
                    revenue = shares_to_sell * close_price
                    self.balance += revenue
                    self.holdings[symbol] -= shares_to_sell
                    logger.info(f"Sold {shares_to_sell} shares of {symbol} at {close_price} per share. Revenue: {revenue:.2f}")
                else:
                    logger.info(f"No shares to sell for {symbol}.")

    def _get_reward(self, new_portfolio_value):
        reward = (new_portfolio_value - self.portfolio_value)#/self.portfolio_value
        logger.info(f"Reward: {reward:.2f}")
        return reward
    
    def _reset(self):
        self.balance = self.initial_balance
        # Randomize holdings: set holdings randomly, but do not invest all balance
        # Initialize holdings with random number of shares between 0 and 20
        self.holdings = {symbol: np.random.randint(0, 21) for symbol in self.symbols}
        # Update balance based on initial holdings
        for symbol in self.symbols:
            close_price = self.df.iloc[self.current_step][f'close_{symbol}']
            cost = self.holdings[symbol] * close_price
            self.balance -= cost
        self.portfolio_value = self.initial_balance
        self.current_step = 0
        self.done = False
        logger.info("Environment reset with random holdings.")
        initial_state = self._get_state()
        return ts.restart(initial_state)
    
    def _step(self, action):
        if self.done:
            logger.info("Environment done. Resetting.")
            return self.reset()
        
        # Ensure action is a numpy array of floats
        if isinstance(action, tf.Tensor):
            action = action.numpy()
        elif isinstance(action, list):
            action = np.array(action)
        elif isinstance(action, np.ndarray):
            action = action
        else:
            action = np.array(action, dtype=np.float32)
        
        # Check action shape and type
        if not isinstance(action, np.ndarray) or action.shape != (self.num_stocks,) or not np.issubdtype(action.dtype, np.floating):
            raise ValueError("Action must be a numpy array of floats with shape equal to the number of stocks.")
        
        # Clip actions to the action spec bounds
        action = np.clip(action, self._action_spec.minimum, self._action_spec.maximum)
        
        # Execute the action
        self._execute_action(action)
        
        # Calculate the new portfolio value
        new_portfolio_value = self._calculate_portfolio_value()
        
        # Calculate the reward
        reward = self._get_reward(new_portfolio_value)
        
        # Update portfolio value
        self.portfolio_value = new_portfolio_value
        
        # Move to the next step
        self.current_step += 1
        if self.current_step >= len(self.df) or self.current_step >= self.max_steps:
            self.done = True
            logger.info("Reached end of data or max steps. Terminating episode.")
        
        # Get the next state
        if self.done:
            next_state = np.zeros(self._observation_spec.shape, dtype=np.float32)
            return ts.termination(next_state, reward)
        else:
            next_state = self._get_state()
            return ts.transition(next_state, reward=reward, discount=1.0)