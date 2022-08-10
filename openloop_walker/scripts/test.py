

from openloop_walker.eval_envs.gym.walk_env import RexWalkEnv
# from openloop_walker.envs.gym.walk_env import RexWalkEnv
import numpy as np
from stable_baselines3 import PPO, SAC


env = RexWalkEnv(render=True, terrain_id='plane', signal_type='ol',terrain_type='plane', control_time_step=0.02, action_repeat=20,)
# env = EvalRexWalkEnv(render=True, terrain_id='hills', signal_type='ol',terrain_type='plane', target_position=4)
print(env.action_space)
print(env.observation_space)

# model = PPO.load('/home/ros/rl_ws/src/openloop_walker/modelz/700k_PPO_model.zip')
model = SAC.load('/home/ros/rl_ws/src/openloop_walker/tl_models/24k_tl_SAC_model.zip')
# model = SAC.load('/home/ros/rl_ws/src/openloop_walker/modelz/21k_SAC_model.zip')

for ep in range(100):
    obs = env.reset()
    for i in range(2000):
        # actions = env.action_space.sample()
        # print(obs)
        obs[1] = 2
        # actions, _states = model.predict(obs,)
        actions, _states = model.predict(obs, deterministic=True)
        # actions = np.array([0.1,0,0,0,0,0,0,0])
        # actions = np.array([0.01,1,1,1,1,1,1,1])
        # print(actions)
        # actions = actions*0.001
        obs, reward, done, info = env.step(action=actions)
        # print(m[0][25:], flush=True, end='\r')
        # print(obs[0])
        # print(round(obs[0], 5), round(obs[1], 2))
        # print(reward)
        if done:
            break