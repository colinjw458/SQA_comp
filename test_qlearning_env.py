import base64
import numpy as np
# import PIL.Image
# import pyvirtualdisplay
import math

import tensorflow as tf

from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import ddpg_agent

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import policy_step
from tf_agents.utils import common

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import PIL.ImageDraw
import PIL.Image
from PIL import ImageFont

class PortfolioEnv(gym.env):

    STATE_ELEMENTS = 7
    STATES = ['age', 'salary', 'home_value', 'home_loan', 'req_home_pmt',
              'acct_tax_adv', 'acct_tax', "expenses", "actual_home_pmt",
              "tax_deposit",
              "tax_adv_deposit", "net_worth"]
    STATE_AGE = 0
    STOCK1_RETURN = 1
    STOCK2_RETURN = 2
    STOCK3_RETURN = 3
    STATE_HOME_REQ_PAYMENT = 4 #  RISK FACTOR

    # Action space is composed of the weight of the 3 stocks 
    ACTION_ELEMENTS = 3
    ACTION_STOCK1 = 0
    ACTION_STOCK2 = 0
    ACTION_STOCK3 = 0

    def __init__(self, goal_velocity = 0):
        self.action_space = spaces.Box(
            low = 0.0, high = 1.0,
            shape=(PortfolioEnv.ACTION_ELEMENTS,),
            dtype = np.float64
        )
        self.observation_space = spaces.box(
            low = 0, high = 2, #TBD
            shape = (PortfolioEnv.STATE_ELEMENTS,),
            dtype = np.float64
        )

        self.seed()
        self.reset()

        self.state_log = []

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _calc_portfolio_return(self):
        stock_1 = self.state[
            PortfolioEnv.STOCK1_RETURN
        ]
        stock_2 = self.state[
            PortfolioEnv.STOCK2_RETURN
        ]
        stock_3 = self.state[
            PortfolioEnv.STOCK3_RETURN
        ]

###### TBD ######
