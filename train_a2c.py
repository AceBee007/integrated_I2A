import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

import gym
import gym_minipacman

from tqdm import tqdm # for progress bar

from common.actor_critic import  OnPolicy, ActorCritic, RolloutStorage
from common.utils import *



if __name__ == '__main__':
    arg = parser.parse_args()
    paramlogger = params_logger()
    paramlogger.log(20, now_time()+' '+argstr(arg))
    paramlogger.log(20, now_time()+' '+hash_arg(arg))
    print(arg)
    print('UES_CUDA = ', USE_CUDA)
    LABEL = 'test_a2c'
    num_envs = arg.num_envs
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

    for i_update in tqdm(range(num_frames)):

        for step in range(num_steps):
            action = actor_critic.act(state.cuda())

            next_state, reward, done, _ = envs.step(action.squeeze(1).cpu().data.numpy())

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
            if i_update % arg.save_model_interval == 0:
                write_histograms(actor_critic, writer, i_update)
                if i_update % 1000 == 0:
                    save_model(actor_critic, '{}_{}'.format(LABEL, i_update), arg)
            
            #clear_output(True)
            #plt.figure(figsize=(20,5))
            #plt.subplot(131)
            #plt.title('epoch %s. reward: %s' % (i_update, np.mean(all_rewards[-10:])))
            #plt.plot(all_rewards)
            #plt.subplot(132)
            #plt.title('loss %s' % all_losses[-1])
            #plt.plot(all_losses)
            #plt.show()
            
        rollout.after_update()
    writer.close()
    save_model(actor_critic, '{}_final'.format(LABEL), arg)
