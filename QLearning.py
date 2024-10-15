import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete

import numpy as np
import time

import stable_baselines3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.type_aliases import GymEnv


class Q_Learning():
    def __init__(self, env: GymEnv):
        
        self.env = env
        self.obs_size = [] # size of the overall observation of the environment
        self.actions = [] # this is the size of the action space

        
        if isinstance(self.env.action_space, Discrete):
            self.actions = [self.env.action_space.n]
        elif isinstance(env.action_space, MultiDiscrete):
            self.actions = env.action_space.nvec

        
        # adding each observation component to the size of the obs_size
        for key in self.env.observation_space:
            if isinstance(self.env.observation_space[key], Discrete):
                self.obs_size += [self.env.observation_space[key].n]
            elif isinstance(self.env.observation_space[key], MultiDiscrete):
                self.obs_size = np.append(self.obs_size, env.observation_space[key].nvec)
                
        self.q_values = np.zeros(np.append(self.obs_size, self.actions), dtype= np.float16)


    # flattens nested lists with multiple values and lists inside, into a single tuple
    # example:    dict_values([0, array([5, 5]), array([9, 9])])         into         (0,5,5,9,9) 
    def list_to_tuple(self, lst):
        flattened_list = []
        for item in lst:
            if isinstance(item, list) or isinstance(item, np.ndarray):
                flattened_list = np.append(flattened_list, item)
            else:
                flattened_list.append(item)
                
        return tuple(flattened_list)
            
        
    
    def get_next_action(self, obs_idx: tuple, epsilon: float):
        if np.random.random() < epsilon:
            action = [np.random.randint(x) for x in self.actions]
            return action
        else:
            max_q_value = np.max(self.q_values[obs_idx])
            action = [x[0] for x in np.where(self.q_values[obs_idx] == max_q_value)]
            return action

    
    def learn(self, total_epochs, epsilon=0.9,epsilon_decay=1, epsilon_min=0.1, lr=0.9, lr_decay=1, lr_min=0.1, discount_factor=0.9, silent=False):
        intervals = total_epochs // 20 # set intervals to 5% of total epochs        
        
        train_start = time.time()
        for epoch in range(total_epochs+1): 
            epoch_reward = 0
            obs, _ = self.env.reset()
            obs_idx = self.list_to_tuple(obs.values())  # transforms observations into single tuple for indexing 
            done = False 
        
            while not done:
                action = self.get_next_action(obs_idx, epsilon)

                old_idx = obs_idx
                obs, reward, done, _, info = self.env.step(np.array(action, dtype = np.int8))
                obs_idx = self.list_to_tuple(obs.values())

                old_q_value = self.q_values[old_idx + tuple(action)]
                temporal_difference = reward + discount_factor * np.max(self.q_values[obs_idx]) * (1- done) - old_q_value # temporal difference equation

                new_q_value = old_q_value + lr * temporal_difference # bellman equation
                self.q_values[old_idx + tuple(action)] = new_q_value

                epoch_reward += reward

            # Print training log at intervals
            if( epoch%intervals == 0 and not silent ):
                print("----- epoch:", epoch,
                      "----- epsilon:", round(epsilon,4), 
                      "----- lr:", round(lr,4),
                      "----- last epoch reward:", round(epoch_reward,2),
                      "-----", round((epoch/total_epochs)*100), 
                      "% complete ----- time elapsed:", round(time.time()-train_start, 5))
                
            epsilon = max(epsilon_min, epsilon*epsilon_decay)
            lr = max(lr_min, lr*lr_decay)
            

    
    def predict():
        pass