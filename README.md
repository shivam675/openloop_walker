# sac_openloop_walker
Successful Implementation of Soft Actor Critic with openloop to induce action values to generate gait

## Getting started
 - start with openloop_walker package
    - It contains all the reinforcement learning scripts and stable models
    - 9m and 8m models in final_models folder in this package are the most stable ones

 - About models
    - model is based on stable baselines3 and is a true pytorch model
    - model arch --> pi = [128, 128, 128, 128] & q = [128, 128, 128, 128]
    - model size is approx 3 mb
    - model has a replaybuffer.pkl

 - observation space
     - dims: numpy array of size 30
     - f > front, r > rear, r > right, l > left, s > shoulder, e > elbow, w > wrist ( for reference below)
     - body_x_vel, body_y_vel, 
     - roll, pitch, roll_rate, pitch_rate
     - currnet_fls_pos, currnet_fle_pos, currnet_flw_pos 
     - currnet_frs_pos, currnet_fre_pos, currnet_frw_pos 
     - currnet_rls_pos, currnet_rle_pos, currnet_rlw_pos 
     - currnet_rrs_pos, currnet_rre_pos, currnet_rrw_pos
     - currnet_fls_vel, currnet_fle_vel, currnet_flw_vel 
     - currnet_frs_vel, currnet_fre_vel, currnet_frw_vel 
     - currnet_rls_vel, currnet_rle_vel, currnet_rlw_vel 
     - currnet_rrs_vel, currnet_rre_vel, currnet_rrw_vel
 - action space
     - action_dims  numpy array of size 12 for 12 motors 
     - action_high -> 0.2 for every motor 
     - action_low -> -0.2 for every motor 
     - fls_pos, fle_pos, flw_pos
     - frs_pos, fre_pos, frw_pos
     - rls_pos, rle_pos, rlw_pos
     - rrs_pos, rre_pos, rrw_pos
     
 - reward function
    - 1.0 x forward reward (current_x / target_postion)*5
    - 2.0 x drift_reward = -abs(current_base_position_y)
    - 0.005 x shake_reward
    - 0.0005 x energy_consumption_reward

 

