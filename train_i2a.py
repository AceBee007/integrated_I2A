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
from common.i2a import RolloutEncoder, I2A, ImaginationCore
from common.utils import *



if __name__ == '__main__':
    arg = parser.parse_args()
    paramlogger = params_logger()
    paramlogger.log(20, now_time()+' '+argstr(arg))
    paramlogger.log(20, now_time()+' '+hash_arg(arg))
    print(arg)
    print('UES_CUDA = ', USE_CUDA)
    LABEL = 'i2a_{}_{}'.format(arg.mode, arg.global_seed)
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

    envs = SubprocVecEnv([make_env(env_id, i) for i in range(num_envs)])
    state = envs.reset()
    state = torch.FloatTensor(np.float32(state)).cuda()
    state_shape = envs.observation_space.shape

    #Init a2c and rmsprop
    actor_critic = ActorCritic(envs.observation_space.shape, envs.action_space.n)
    optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha)
    # Init rollout storage
    rollout = RolloutStorage(num_steps, num_envs, envs.observation_space.shape)
    if USE_CUDA:
        actor_critic = actor_critic.cuda()
        rollout.cuda()
    
    rollout.states[0].copy_(state)
    
    episode_rewards = torch.zeros(num_envs, 1)
    final_rewards   = torch.zeros(num_envs, 1)

    writer = new_writer(LABEL, arg)

    a2c_model_path =  './trained_models/tmp_a2c_{}'.format(arg.global_seed)
    if os.path.exists(a2c_model_path):
        print('Load A2C model from ', a2c_model_path)
        actor_critic.load_state_dict(torch.load(a2c_model_path))
    else:
        print('Start training A2C model')
        for i_update in tqdm(range(num_frames)):

            for step in range(num_steps):
                action = actor_critic.act(state.cuda())

                next_state, reward, done, _ = envs.step(action.squeeze(1).cpu().data.numpy())
                
                reward = process_reward(reward, MODE_REWARDS[mode])
                reward = torch.FloatTensor(reward).unsqueeze(1)
                episode_rewards += reward
                masks = torch.FloatTensor(1-np.array(done)).unsqueeze(1)
                final_rewards *= masks
                final_rewards += (1-masks) * episode_rewards
                episode_rewards *= masks

                if USE_CUDA:
                    masks = masks.cuda()

                state = torch.FloatTensor(np.float32(next_state))
                rollout.insert(step, state, action.data, reward, masks)

            with torch.no_grad():
                _, next_value = actor_critic(rollout.states[-1])
            next_value = next_value.data

            returns = rollout.compute_returns(next_value, gamma)

            logit, action_log_probs, values, entropy = actor_critic.evaluate_actions(
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
            nn.utils.clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
            optimizer.step()
            rollout.after_update
        torch.save(actor_critic.state_dict(), a2c_model_path)
    print('Finished training A2C')
    
    def get_action(state):
        if state.ndim == 4:
            state = FloatTensor(np.float32(state))
        else:
            state = FloatTensor(np.float32(state)).unsqueeze(0)
        with torch.no_grad():
            action = actor_critic.act(state)
        action = action.data.cpu().squeeze(1).numpy()
        return action

    def play_games(envs, frames):
        states = envs.reset()
        
        for frame_idx in tqdm(range(frames)):
            actions = get_action(states)
            next_states, rewards, dones, _ = envs.step(actions)
            
            yield frame_idx, states, actions, rewards, next_states, dones
            
            states = next_states

    env_model = EnvModel(envs.observation_space.shape, num_pixels=arg.env_pixel_model, num_envs=arg.num_envs)
    env_model.cuda()
    env_model_path = './trained_models/env_model_a2c_{}'.format(arg.global_seed)
    if os.path.exists(env_model_path):
        print('Load Env Model')
        env_model.load_state_dict(torch.load(env_model_path))
    else:
        print('Start training env_model')
        reward_coef = 0.1
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(env_model.parameters())
   	    
        envs = SubprocVecEnv([make_env(env_id, i) for i in range(num_envs)])
        state = envs.reset()
        state = FloatTensor(state)

        for frame_idx, states, actions, rewards, next_states, dones in play_games(envs, arg.num_frames):
            states      = FloatTensor(states)
            actions     = LongTensor(actions)

            inputs = env_model.get_inputs(states, actions)

            imagined_state, imagined_reward = env_model(inputs)

            target_state = LongTensor(pix_to_target(next_states))
            
            target_reward = FloatTensor(rewards)

            optimizer.zero_grad()
            image_loss  = criterion(imagined_state, target_state)
            reward_loss = nn.functional.mse_loss(imagined_reward, target_reward)
            loss = image_loss + reward_coef * reward_loss
            loss.backward()
            optimizer.step()

        torch.save(env_model.state_dict(), env_model_path)
        print('Finished training env_model')
    
    state_shape = envs.observation_space.shape
    num_actions = envs.action_space.n
    num_rewards = len(MODE_REWARDS[arg.mode])


    distill_policy = actor_critic
    distill_optimizer = optim.Adam(distill_policy.parameters())
    imagination = ImaginationCore(arg.rollout_depth, state_shape, num_actions, num_rewards, env_model, distill_policy, full_rollout=True)


    i2a = I2A(state_shape, num_actions, num_rewards,  256, imagination, full_rollout=True)
    optimizer = optim.RMSprop(i2a.parameters(), lr, eps=eps, alpha=alpha)
     
    rollout = RolloutStorage(num_steps, num_envs, envs.observation_space.shape)
    if USE_CUDA:
        i2a.cuda()
        rollout.cuda()

    state = envs.reset()
    current_state = FloatTensor(np.float32(state))

    rollout.states[0].copy_(current_state)

    episode_rewards = torch.zeros(num_envs, 1)
    final_rewards   = torch.zeros(num_envs, 1)
    if USE_CUDA:
        episode_rewards = episode_rewards.cuda()
        final_rewards = final_rewards.cuda()

    print('Start training I2A')
    for i_update in tqdm(range(num_frames)):

        for step in range(num_steps):
            if USE_CUDA:
                current_state = current_state.cuda()
            action = i2a.act(current_state)

            next_state, raw_reward, done, _ = envs.step(action.squeeze(1).cpu().data.numpy())
            reward = process_reward(raw_reward, MODE_REWARDS[mode])

            reward = FloatTensor(reward).unsqueeze(1)
            episode_rewards += reward
            masks = FloatTensor(1-np.array(done)).unsqueeze(1)
            final_rewards *= masks
            final_rewards += (1-masks) * episode_rewards
            episode_rewards *= masks

            if USE_CUDA:
                masks = masks.cuda()

            current_state = FloatTensor(np.float32(next_state))
            rollout.insert(step, current_state, action.data, reward, masks)


        with torch.no_grad():
            _, next_value = i2a(rollout.states[-1])
        next_value = next_value.data

        returns = rollout.compute_returns(next_value, gamma)

        logit, action_log_probs, values, entropy = actor_critic.evaluate_actions(
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
        nn.utils.clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
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
            if i_update != 0 and i_update % arg.save_model_interval == 0:
                write_histograms(i2a, writer, i_update)
                if i_update % 1000 == 0:
                    save_model(i2a, '{}_{}'.format(LABEL, i_update), arg)
            
        rollout.after_update()
    save_model(i2a, '{}_{}'.format(LABEL, i_update), arg)
    writer.close()



