# train_agent.py

import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import sys
import random
import matplotlib.pyplot as plt
from tf_agents.environments import tf_py_environment
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.policies import policy_saver
from tf_agents.trajectories import trajectory
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import policy_step
from tf_agents.utils import common
from tf_agents.environments import suite_gym
from tf_agents.metrics import tf_metrics

from py_env_env import TradingEnvironment
from py_env_preprocess import synchronize_data  # Will use csp to deal with syncronized data

def train_test_split(df, split_ratio=0.7):
    split_index = int(len(df) * split_ratio)
    train_data = df.iloc[:split_index]
    test_data = df.iloc[split_index:]
    return train_data, test_data

def setup_logging(log_file='training_log.txt'):
    """
    Sets up logging to output to both console and a file.

    :param log_file: The filename for the log file.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(log_file, mode='w')

    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    
    # Ensure action is a numpy array of floats
    if isinstance(action_step.action, tf.Tensor):
        action = action_step.action.numpy()
    elif isinstance(action_step.action, np.ndarray):
        action = action_step.action
    else:
        action = np.array(action_step.action, dtype=np.float32)
    
    # Validate the action shape and bounds
    # if not isinstance(action, np.ndarray) or action.shape != (environment.num_stocks,) or not np.issubdtype(action.dtype, np.floating):
    #     raise ValueError("Action must be a numpy array of floats with shape equal to the number of stocks.")
    
    # Clip actions to the action spec bounds
    action = np.clip(action, environment._action_spec.minimum, environment._action_spec.maximum)
    
    # Create a new PolicyStep with the clipped action
    action_step = policy_step.PolicyStep(action=tf.convert_to_tensor(action, dtype=tf.float32))
    
    # Execute the action
    next_time_step = environment.step(action_step.action)
    
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)


def compute_average_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for episode in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            if isinstance(action_step.action, tf.Tensor):
                action = action_step.action.numpy()
            elif isinstance(action_step.action, np.ndarray):
                action = action_step.action
            else:
                action = np.array(action_step.action, dtype=np.float32)
            
            action = np.clip(action, environment._action_spec.minimum, environment._action_spec.maximum)
            
            action_step = policy_step.PolicyStep(action=tf.convert_to_tensor(action, dtype=tf.float32))
            
            next_time_step = environment.step(action_step.action)
            episode_return += next_time_step.reward.numpy()
            time_step = next_time_step
        total_return += episode_return
    average_return = total_return / num_episodes
    return average_return

def main():
    setup_logging(log_file='training_log.txt')
    logger = logging.getLogger()
    df = pd.read_csv('Data/HistoricalEquityData.csv') 
    symbols = df['symbol'].unique().tolist()
    
    df_synchronized = synchronize_data(df, symbols, freq='1min')  ## CSP
    
    train_environment = TradingEnvironment(df_synchronized, symbols, initial_balance=100000, max_steps=68)
    tf_train_environment = tf_py_environment.TFPyEnvironment(train_environment)

    test_environment = TradingEnvironment(df_synchronized, symbols, initial_balance=100000, max_steps=68)
    tf_test_environment = tf_py_environment.TFPyEnvironment(test_environment)
    
    # Network
    actor_fc_layers = (128, 128)
    critic_obs_fc_layers = (400,)
    critic_action_fc_layers = None
    critic_joint_fc_layers = (300,)
    actor_net = actor_network.ActorNetwork(
                tf_train_environment.time_step_spec().observation,
                tf_train_environment.action_spec(),
                fc_layer_params=actor_fc_layers,
            )

    critic_net_input_specs = (tf_train_environment.time_step_spec().observation,
                                tf_train_environment.action_spec())

    critic_net = critic_network.CriticNetwork(
            critic_net_input_specs,
            observation_fc_layer_params=critic_obs_fc_layers,
            action_fc_layer_params=critic_action_fc_layers,
            joint_fc_layer_params=critic_joint_fc_layers,
        )


    # Agent
    ou_stddev = 0.2
    ou_damping = 0.15
    target_update_tau = 0.05
    target_update_period = 5
    dqda_clipping = None
    td_errors_loss_fn = tf.compat.v1.losses.huber_loss
    gamma = 0.995
    reward_scale_factor = 1.0
    gradient_clipping = None

    actor_learning_rate = 1e-4
    critic_learning_rate = 1e-3
    debug_summaries = False
    summarize_grads_and_vars = False

    global_step = tf.compat.v1.train.get_or_create_global_step()


    tf_agent = ddpg_agent.DdpgAgent(
        tf_train_environment.time_step_spec(),
        tf_train_environment.action_spec(),
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        ou_stddev=ou_stddev,
        ou_damping=ou_damping,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        dqda_clipping=dqda_clipping,
        td_errors_loss_fn=td_errors_loss_fn,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step)
    tf_agent.initialize()
    

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=tf_train_environment.batch_size,
        max_length=100  ## CAHNGE for testing
    )
    
    logger.info("Replay buffer initialized.")

    random_policy = tf_agent.collect_policy
    
    initial_collect_steps = 100 ## CAHNGE
    logger.info(f"Collecting initial {initial_collect_steps} steps...")
    for _ in range(initial_collect_steps):
        collect_step(tf_train_environment, random_policy, replay_buffer)
    logger.info("Initial data collection completed.")
    
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=64,
        num_steps=2
    ).prefetch(3)
    
    iterator = iter(dataset)
    
    
    num_iterations = 100  ## CAHNGE for testing
    log_interval = 10
    eval_interval = 4
    num_episodes = 10

    average_return = compute_average_return(tf_test_environment, tf_agent.policy, num_episodes)

    cumulative_returns = [average_return]
    iterations = [0]
    
    logger.info("Starting training...")
    for iteration in range(1, num_iterations + 1):
        collect_step(tf_train_environment, tf_agent.collect_policy, replay_buffer)
    
        # Sample a batch of data from the buffer and update the agent
        try:
            experience, unused_info = next(iterator)
        except StopIteration:
            iterator = iter(dataset)
            experience, unused_info = next(iterator)
        
        train_loss = tf_agent.train(experience).loss
    
        # Log progress at specified intervals
        if (iteration % log_interval) == 0:
            logger.info(f"Iteration {iteration}: Loss = {train_loss.numpy()}")
            
    
        # eval on test
        if (iteration % eval_interval) == 0:
            average_return = compute_average_return(tf_test_environment, tf_agent.policy, num_episodes)
            logger.info(f"Iteration {iteration}: Loss = {train_loss.numpy()}, Avg Return on test dataset = {average_return}")
            cumulative_returns.append(average_return)
            iterations.append(iteration)
    
    # =========================
    # Save the Final Policy
    # =========================

    saver = policy_saver.PolicySaver(tf_agent.collect_policy)
    saver.save('policy_final.h5')
    print("Saved final policy.")
    
    # =========================
    # Plot the Cumulative Returns
    # =========================

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, cumulative_returns, label='Test Average Return')
    plt.xlabel('Training Iterations')
    plt.ylabel('Average Return on Test Set')
    plt.title('Test Set Performance Over Training Iterations')
    plt.legend()
    plt.grid(True)
    plt.savefig('test_performance.png')
    plt.show()

if __name__ == "__main__":
    main()