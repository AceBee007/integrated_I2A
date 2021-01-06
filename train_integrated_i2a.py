import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

import gym
import gym_minipacman

from common.environment_model import EnvModel, EnvModelRolloutStorage
from common.i_i2a import Integrated_I2A, RolloutStorage

from tqdm import tqdm

from common.utils import *

if __name__ == '__main__':
    arg = parser.parse_args()
    paramlogger = params_logger()
    paramlogger.log(20, now_time()+' '+argstr(arg))
    paramlogger.log(20, now_time()+' '+hash_arg(arg))
    print(arg)
    print('UES_CUDA = ', USE_CUDA)
    LABEL = 'i_i2a({}_{}_{})'.format(arg.mode,arg.rollout_method,arg.global_seed)                  
    num_envs = arg.num_envs          # 環境の数
    mode = arg.mode
    env_id = "{}MiniPacmanNoFrameskip-v0".format(arg.mode.capitalize())
    global_seed = arg.global_seed
    set_random_seed(global_seed)     # グローバルのシード値
    # a2c hyperparams:
    gamma = 0.99                     # 割引率
    entropy_coef = 0.01              # エントロピーの係数
    value_loss_coef = 0.5            # Value loss の係数
    max_grad_norm = 0.5              # 最大傾き更新のノーマライゼーション
    num_steps = arg.num_steps        # ステップ数
    num_frames = int(arg.num_frames) # 訓練フレーム(batch)数
    rollout_depth = arg.rollout_depth
    rollout_breadth = arg.rollout_breadth
    rollout_method = arg.rollout_method

    #rmsprop hyperparams:
    #lr    = 7e-4 # default
    lr    = arg.learning_rate
    eps   = 1e-5
    alpha = 0.99

    envs = SubprocVecEnv([make_env(env_id, i) for i in range(num_envs)])
    state = envs.reset()
    state = FloatTensor(state)
    state_shape = envs.observation_space.shape
    num_pixels = arg.env_pixel_model # default is 8
    env_model    = EnvModel(envs.observation_space.shape, num_pixels,num_rewards=5)
    
    #Init i_i2a and rmsprop
    i_i2a = Integrated_I2A(
        in_shape=envs.observation_space.shape,
        num_actions=envs.action_space.n,
        env_model=env_model,
        mode_reward=MODE_REWARDS[mode],
        rollout_method=rollout_method,
        rollout_depth=rollout_depth,
        rollout_breadth=rollout_breadth
    )
    optimizer = optim.RMSprop(i_i2a.parameters(), lr, eps=eps, alpha=alpha)
    
    rollout = RolloutStorage(num_steps, num_envs, envs.observation_space.shape)

    if USE_CUDA:
        i_i2a = i_i2a.cuda()
        rollout.cuda()
        env_model.cuda()


    state = envs.reset()
    state = FloatTensor(state)

    rollout.reset_first(state)

    episode_rewards = torch.zeros(num_envs, 1).cuda()
    final_rewards   = torch.zeros(num_envs, 1).cuda()
    
    writer = new_writer(LABEL, arg)

    for i_update in tqdm(range(num_frames)):

        for step in range(num_steps):
            action = i_i2a.get_planned_action(state)

            next_state, raw_reward, done, _ = envs.step(action.squeeze(1).cpu().data.numpy()) # got numpy value
            next_state = FloatTensor(next_state)
            i_i2a.store_trajectory(state, action, next_state, FloatTensor(raw_reward))
            reward = prosses_reward(raw_reward)

            reward = FloatTensor(reward).unsqueeze(1)
            episode_rewards += reward
            masks = FloatTensor(1-np.array(done)).unsqueeze(1)
            final_rewards *= masks
            final_rewards += (1-masks) * episode_rewards
            episode_rewards *= masks

            state = next_state
            rollout.insert(step, state, action.data, reward, masks)
            

        with torch.no_grad():
            _, next_value = i_i2a(rollout.states[-1])
        next_value = next_value.data

        returns = rollout.compute_returns(next_value, gamma)

        logit, action_log_probs, values, entropy = i_i2a.evaluate_actions(
            rollout.states[:-1].view(-1, *state_shape),
            rollout.actions.view(-1, 1)
        )

        values = values.view(num_steps, num_envs, 1)
        action_log_probs = action_log_probs.view(num_steps, num_envs, 1)
        advantages = returns - values

        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.data * action_log_probs).mean()
        optimizer.zero_grad()
        loss = value_loss * value_loss_coef + action_loss - entropy * entropy_coef
        loss.backward()
        nn.utils.clip_grad_norm_(i_i2a.parameters(), max_grad_norm)
        optimizer.step()
        
        rollout.after_update()
        
        if i_update % arg.log_interval == 0:
            writer.add_scalars('final_rewards', {
                'max':max(final_rewards),
                'min':min(final_rewards),
                'mean':final_rewards.mean()
                }, i_update)
            writer.add_scalars('loss', {
                'value_loss':value_loss,
                'action_loss':action_loss,
                'loss':float(loss.item())
                    }, i_update)
            writer.add_scalars('env_loss', {
                'env_loss_SMA':i_i2a.env_losses.mean(),
                'env_loss':i_i2a.current_env_loss:
                }, i_update)
            if i_update % 200 == 0:
                write_histograms(i_i2a, writer, i_update)

            # save models
            if i_update != 0 and i_update%arg.save_model_interval == 0:
                save_model(i_i2a, '{}_{}'.format(LABEL, i_update), arg)
                save_model(i_i2a.env_model, '{}_envmodel_{}'.format(LABEL, i_update), arg)
    writer.close()
    save_model(i_i2a, '{}_{}'.format(LABEL, 'final'), arg)
    save_model(i_i2a.env_model, '{}_envmodel_{}'.format(LABEL, 'final'), arg)



