import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

import gym
import gym_minipacman

from tqdm import tqdm # for progress bar
import os

from common.actor_critic import  OnPolicy, ActorCritic, RolloutStorage
from common.environment_model import EnvModel
from common.ei_i2a import RolloutEncoder, EnvIntegrated_I2A, ImaginationCore
from common.utils import *



if __name__ == '__main__':
    arg = parser.parse_args()
    paramlogger = params_logger()
    paramlogger.log(20, now_time()+' '+argstr(arg))
    paramlogger.log(20, now_time()+' '+hash_arg(arg))
    print(arg)
    print('UES_CUDA = ', USE_CUDA)
    LABEL = 'ei_i2a_{}_{}'.format(arg.mode, arg.global_seed)
    num_envs = arg.num_envs
    mode = arg.mode
    env_id = '{}MiniPacmanNoFrameskip-v0'.format(arg.mode.capitalize())
    global_seed = arg.global_seed
    # a2c hyperparams:
    gamma = 0.99                     # 割引率
    entropy_coef = 0.01              # エントロピーの係数
    value_loss_coef = 0.5            # Value loss の係数
    max_grad_norm = 0.5              # 最大傾き更新のノーマライゼーション
    num_steps = arg.num_steps        # ステップ数
    num_frames = int(arg.num_frames) # 訓練フレーム数
    set_random_seed(global_seed)     # グローバルのシード値
    #rmsprop hyperparams:
    #lr    = 7e-4 # default
    lr    = arg.learning_rate
    eps   = 1e-5
    alpha = 0.99

    num_pixels = arg.env_pixel_model # default is 8
    envs = SubprocVecEnv([make_env(env_id, i) for i in range(num_envs)])
    state = envs.reset()
    state = FloatTensor(state)
    state_shape = envs.observation_space.shape


    writer = new_writer(LABEL, arg)
    
    state_shape = envs.observation_space.shape
    num_actions = envs.action_space.n
    num_rewards = len(MODE_REWARDS[arg.mode])


    env_model    = EnvModel(envs.observation_space.shape, num_pixels,num_rewards=5)
    if USE_CUDA:
        env_model.cuda()
    distill_policy = ActorCritic(envs.observation_space.shape, envs.action_space.n)
    distill_optimizer = optim.Adam(distill_policy.parameters())

    ei_i2a = EnvIntegrated_I2A(state_shape, num_actions, hidden_size=256, full_rollout=True, env_model=env_model, mode_reward=MODE_REWARDS[mode])
    imagination = ImaginationCore(arg.rollout_depth, state_shape, num_actions, num_rewards, ei_i2a.env_model, distill_policy, full_rollout=True)
    ei_i2a.set_imagination(imagination)
    optimizer = optim.RMSprop(ei_i2a.parameters(), lr, eps=eps, alpha=alpha)
     
    rollout = RolloutStorage(num_steps, num_envs, envs.observation_space.shape)
    if USE_CUDA:
        distill_policy.cuda()
        ei_i2a.cuda()
        rollout.cuda()

    state = envs.reset()
    current_state = FloatTensor(state)

    rollout.states[0].copy_(current_state)

    episode_rewards = torch.zeros(num_envs, 1)
    final_rewards   = torch.zeros(num_envs, 1)
    if USE_CUDA:
        episode_rewards = episode_rewards.cuda()
        final_rewards = final_rewards.cuda()

    print('Start training EI_I2A')
    for i_update in tqdm(range(num_frames)):

        for step in range(num_steps):
            action = ei_i2a.act(current_state)

            next_state, raw_reward, done, _ = envs.step(action.squeeze(1).cpu().data.numpy())
            next_state = FloatTensor(next_state)
            ei_i2a.store_trajectory(current_state, action, next_state, FloatTensor(raw_reward))
            reward = process_reward(raw_reward, MODE_REWARDS[mode])

            reward = FloatTensor(reward).unsqueeze(1)
            episode_rewards += reward
            masks = FloatTensor(1-np.array(done)).unsqueeze(1)
            final_rewards *= masks
            final_rewards += (1-masks) * episode_rewards
            episode_rewards *= masks

            current_state = next_state
            rollout.insert(step, current_state, action.data, reward, masks)


        with torch.no_grad():
            _, next_value = ei_i2a(rollout.states[-1])
        next_value = next_value.data

        returns = rollout.compute_returns(next_value, gamma)

        logit, action_log_probs, values, entropy = ei_i2a.evaluate_actions(
            rollout.states[:-1].view(-1, *state_shape),
            rollout.actions.view(-1, 1)
        )
        
        distil_logit, _, _, _ = distill_policy.evaluate_actions(
            rollout.states[:-1].view(-1, *state_shape),
            rollout.actions.view(-1, 1)
        )
            
        distil_loss = 0.01 * (F.softmax(logit, dim=1).detach() * F.log_softmax(distil_logit, dim=1)).sum(1).mean()

        values = values.view(num_steps, num_envs, 1)
        action_log_probs = action_log_probs.view(num_steps, num_envs, 1)
        advantages = returns - values

        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.data * action_log_probs).mean()

        optimizer.zero_grad()
        loss = value_loss * value_loss_coef + action_loss - entropy * entropy_coef
        loss.backward()
        nn.utils.clip_grad_norm_(ei_i2a.parameters(), max_grad_norm)
        optimizer.step()
        
        distill_optimizer.zero_grad()
        distil_loss.backward()
        distill_optimizer.step()
        
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
                'env_loss_SMA':ei_i2a.env_losses.mean(),
                'env_loss':ei_i2a.current_env_loss,
                    }, i_update)
            if i_update != 0 and i_update % arg.save_model_interval == 0:
                write_histograms(ei_i2a, writer, i_update)
                if i_update % 1000 == 0:
                    save_model(ei_i2a, '{}_{}'.format(LABEL, i_update), arg)
            
        rollout.after_update()
    save_model(ei_i2a, '{}_{}'.format(LABEL, i_update), arg)
    writer.close()



