#!/usr/bin/python3


from openloop_walker.envs.gym.walk_env import RexWalkEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecFrameStack
# from stable_baselines3.common.vec_env import SubprocVecEnv

import numpy as np
import rospkg
import torch
import rospkg
rospack = rospkg.RosPack()

pkg = rospkg.RosPack()
path = pkg.get_path('openloop_walker')



HEIGHT_FIELD = False
CONTACTS = True





policy_kwargs = dict(
    # squash_output=True,
    optimizer_class = torch.optim.Adam,
    activation_fn=torch.nn.LeakyReLU, 
    net_arch=dict(pi=[128, 128, 128, 128], qf=[128, 128, 128, 128])
    )

# policy_kwargs = dict(activation_fn=th.nn.ReLU,
#                      net_arch=[dict(pi=[32, 32], vf=[32, 32])])

if __name__ == "__main__":

    env = make_vec_env(RexWalkEnv, n_envs=4,)
    # env = RexWalkEnv()

    # env = VecFrameStack(env, n_stack=4)

    log_path =  path + '/log_path/'
    outdir_2 = path + '/model_final/{}m_PPO_model'


    # model = SAC('MlpPolicy', env, learning_starts=1, verbose=2, learning_rate=0.0001, policy_kwargs=policy_kwargs, batch_size=1400, tensorboard_log=log_path)
    model = SAC('MlpPolicy', env, learning_starts=1,verbose=2, learning_rate=0.00001, policy_kwargs=policy_kwargs, batch_size=256, tensorboard_log=log_path)
    # model.get_parameters()
    # model = SAC.load('/home/ros/rl_ws/src/openloop_walker/modelz/12.0m_SAC_model', env = env)
    # model.load_replay_buffer('/home/ros/rl_ws/src/openloop_walker/modelz/replaybufferm_SAC_model.pkl')
    
    for i in range(1,1000):
        model.learn(total_timesteps=100000, log_interval=1,)
        model.save(outdir_2.format(i))
        model.save_replay_buffer(outdir_2.format('replaybuffer'))

    
