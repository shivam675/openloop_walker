#!/usr/bin/python3

from openloop_walker.envs.gym.walk_env import RexWalkEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv

import numpy as np
import rospkg
import torch
from itertools import cycle
import rospkg
rospack = rospkg.RosPack()

pkg = rospkg.RosPack()
path = pkg.get_path('openloop_walker')

# save_path = path + '/torch_models'


HEIGHT_FIELD = False
CONTACTS = True


# env = make_vec_env(RexWalkEnv, n_envs=4,)

# env = RexWalkEnv(render=True, debug=False, terrain_id='plane', signal_type='ol', control_time_step=0.005, terrain_type='plane')
env = RexWalkEnv(render=False, debug=False, terrain_id='plane', signal_type='ol', control_time_step=0.005, terrain_type='plane')

# eval_env = RexWalkEnv(render=False, debug=False, terrain_id='plane', signal_type='ol', control_time_step=0.005, terrain_type='plane')


policy_kwargs = dict( 
    optimizer_class = torch.optim.Adam, 
    activation_fn=torch.nn.LeakyReLU, 
    net_arch=dict(pi=[256, 128, 128], qf=[128, 128, 256])
    )


# policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[400, 300]))


if __name__ == "__main__":

    log_path =  path + '/log_path/'
    outdir_2 = path + '/modelz/{}m_SAC_model'


    model = SAC('MlpPolicy', env, verbose=2,learning_rate=0.0001, policy_kwargs=policy_kwargs, batch_size=64, tensorboard_log=log_path)

    # model = SAC.load('/home/ros/rl_ws/src/openloop_walker/modelz/SAC_working_models/500k_SAC_model.zip', env)
    model.get_parameters()
    for i in range(1,100):
        model.learn(total_timesteps=100000)
        model.save_replay_buffer(outdir_2.format('replaybuffer'))
    # model.learn(total_timesteps=1)
        model.save(outdir_2.format(i/2))