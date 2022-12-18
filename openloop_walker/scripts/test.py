#!/usr/bin/env python3

# from openloop_walker.eval_envs.gym.walk_env import RexWalkEnv
from openloop_walker.envs.gym.walk_env import RexWalkEnv
from openloop_walker.eval_envs import rex_gym_env
from stable_baselines3.common.env_util import make_vec_env

import numpy as np
from stable_baselines3 import PPO, SAC
import random


# env = RexWalkEnv(render=True, terrain_id='plane', signal_type='ol',terrain_type='plane', control_time_step=0.02, action_repeat=20,)

env = RexWalkEnv(render=True, terrain_id='plane', signal_type='ol',terrain_type='plane', control_time_step=0.005, action_repeat=5, target_position=10, debug=True)



# env = RexWalkEnv(render=True, terrain_id='random', signal_type='ol',terrain_type='random', control_time_step=0.005, action_repeat=5, target_position=5, debug=True)
# env = EvalRexWalkEnv(render=True, terrain_id='hills', signal_type='ol',terrain_type='plane', target_position=4)
# print(env.action_space)
# print(env.observation_space)

# model = PPO.load('/home/ros/rl_ws/src/openloop_walker/modelz/700k_PPO_model.zip')
model = SAC.load('/home/ros/rl_ws/src/openloop_walker/model_final/10m_PPO_model.zip')
# model = SAC.load('/home/ros/rl_ws/src/openloop_walker/modelz/8m_SAC_model.zip')

# env = make_vec_env(RexWalkEnv, n_envs=1,)

for ep in range(100):
    # env._target_position = 20
    # distance = random.randint(1, 20)
    # env.set_attr('_target_position', distance)
    # if distance <= 6:
    #     env.set_attr("max_timesteps_allowed", value= (5500/20)*env._target_position + 350)
    
    # else:
    #     env.set_attr("max_timesteps_allowed", value= (5500/20)*env._target_position)  # for 20 mtrs time steps required are 5500 if control_time_step is 0.005 and action repeat is 5
    # final_obs = [] 
    # for _ in range(4):
    #     final_obs.extend(obs.tolist())
    obs = env.reset()
    # print(env.max_timesteps_allowed)
    # for i in range(6000):
    while True:
        # print(i)
        # actions = env.action_space.sample()
        # print(obs)
        # obs[1] = 15
        # print(obs[1])
        

        actions, _states = model.predict(obs,)
        # actions, _states = model.predict(obs, deterministic=True)
        # actions = np.array([0.0,0,0,0,0,0,0,0])
        # actions = np.array([0.01,1,1,1,1,1,1,1])
        print(actions)
        # actions = actions*1.5
        obs, reward, done, info = env.step(action=actions)
        # print(m[0][25:], flush=True, end='\r')
        # print(obs[0])
        # print(round(obs[0], 5), round(obs[1], 2))
        print(reward)
        if done:
            break