import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from .actor_critic import OnPolicy, RolloutStorage
from .environment_model import EnvModelRolloutStorage, EnvModel
from .utils import process_reward, np_softmax, np_deque_append, target_to_pix

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor 

class RolloutEncoder(nn.Module):
    def __init__(self, in_shape, num_rewards, hidden_size):
        super(RolloutEncoder, self).__init__()
        
        self.in_shape = in_shape
        
        self.features = nn.Sequential(
            nn.Conv2d(in_shape[0], 16, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.LeakyReLU(),
        )
        
        self.gru = nn.GRU(self.feature_size() + num_rewards, hidden_size)
        
    def forward(self, state, reward):
        num_steps  = state.size(0)
        batch_size = state.size(1)
        
        state = state.view(-1, *self.in_shape)
        state = self.features(state)
        state = state.view(num_steps, batch_size, -1)
        rnn_input = torch.cat([state, reward], 2)
        _, hidden = self.gru(rnn_input)
        return hidden.squeeze(0)
    
    
    def feature_size(self):
        return self.features(torch.zeros(1, *self.in_shape)).view(1, -1).size(1)


class EnvIntegrated_I2A(OnPolicy):
    
    def __init__(self, in_shape, num_actions, hidden_size, 
                mode_reward,
                env_model,
                full_rollout=True,
                 mode='RegularMiniPacmanNoFrameskip-v0',
                 num_rewards = 5,
                 env_storage_capa=10, 
                 num_envs=16, 
                 reset_env_model=False, 
                 rollout_depth=3, 
                 rollout_breadth=3, 
                 rollout_method='MonteCarlo'):
        super().__init__()
        
        self.in_shape      = in_shape
        self.num_actions   = num_actions
        self.num_rewards   = num_rewards
        self.avaliable_actions = np.array(list(range(num_actions)))
        
        self.env_model = env_model
        
        self.mode = mode
        self.mode_reward = mode_reward
        
        self.env_storage_capa = env_storage_capa
        self.env_storage = EnvModelRolloutStorage(env_storage_capa, num_envs, self.in_shape)
        self.current_storage_len = 0

        # # used in method 1 (interval sampling method)
        # self.env_learn_interval = 1
        # self.env_loss = 5 # a big number to calculate next learn_interval
        
        # # used in method 2
        self.env_losses = np.array([0.0 for i in range(self.env_storage_capa)])
        self.current_env_loss = 0
        self.current_frame = 0

        self.env_reward_coef = 0.1
        self.env_criterion = nn.MSELoss()
        self.tmp_criterion = nn.MSELoss()
        self.env_optimizer = optim.Adam(self.env_model.parameters())
        
        
        self.features = nn.Sequential(
            nn.Conv2d(in_shape[0], 16, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.LeakyReLU(),
        )
        
        self.encoder = RolloutEncoder(in_shape, num_rewards, hidden_size)
        
        if full_rollout:
            self.fc = nn.Sequential(
                nn.Linear(self.feature_size() + num_actions * hidden_size, 256),
                nn.LeakyReLU(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.feature_size() + hidden_size, 256),
                nn.LeakyReLU(),
            )
        
        self.critic  = nn.Linear(256, 1)
        self.actor   = nn.Linear(256, num_actions)

    def set_imagination(self, imagination):
        self.imagination = imagination
        
    def forward(self, state):
        batch_size = state.size(0)
        
        imagined_state, imagined_reward = self.imagination(state.data)
        hidden = self.encoder(imagined_state, imagined_reward)
        hidden = hidden.view(batch_size, -1)
        
        state = self.features(state)
        state = state.view(state.size(0), -1)
        
        x = torch.cat([state, hidden], 1)
        x = self.fc(x)
        
        logit = self.actor(x)
        value = self.critic(x)
        
        return logit, value
        
    def feature_size(self):
        return self.features(torch.zeros(1, *self.in_shape)).view(1, -1).size(1)
    
    
    def store_trajectory(self, state, action, next_state,  reward):
        # 毎回、env_model(state+action)のimaginedとlnext_state, reward比較し、lossがSMA(loss)より大きいものについて学習
        with torch.no_grad():
            tmp_env_inputs = self.env_model.get_inputs(state, action)
            tmp_imagined_state, tmp_imagined_reward = self.env_model(tmp_env_inputs)
            tmp_target_state = self.env_model.state_to_target(next_state)
            tmp_target_reward = reward
            image_loss  = self.tmp_criterion(tmp_imagined_state.detach().clone(), tmp_target_state.detach().clone())
            reward_loss = self.tmp_criterion(tmp_imagined_reward.detach().clone(), tmp_target_reward.detach().clone())
            tmp_loss = (image_loss + self.env_reward_coef * reward_loss).data.cpu().numpy()
        self.current_env_loss = tmp_loss
        env_loss_threashold = self.env_losses.mean()
        if tmp_loss > env_loss_threashold:
            self.env_storage.insert(self.current_storage_len, state, action, next_state, reward)
            self.current_storage_len += 1 # 履歴の長さを更新
        self.env_losses = np_deque_append(self.env_losses, tmp_loss)
        # 履歴がenv_storage_capaに達したら、学習
        if self.current_storage_len+1 == self.env_storage_capa:
            self.train_env()
            self.current_storage_len = 0 # 最後で履歴の長さをリセット
            
    def train_env(self):
        # 毎回、store_trajectoryの後、len(env_storage)がmaxに達したかを確認、達したら、これを実行
        # losses = [] # used in  interval sampling method
        for step in range(len(self.env_storage)):
            states = FloatTensor(self.env_storage.states[step])
            actions = LongTensor(self.env_storage.actions[step])
            next_states = FloatTensor(self.env_storage.next_states[step])
            rewards = FloatTensor(self.env_storage.rewards[step])
            batch_size = states.size(0)
            env_inputs = self.env_model.get_inputs(states, actions)
            
            imagined_state, imagined_reward = self.env_model(env_inputs)
            
            target_state = self.env_model.state_to_target(next_states.detach().clone())
            target_reward = rewards

            self.env_optimizer.zero_grad()
            image_loss  = self.env_criterion(imagined_state, target_state)
            reward_loss = self.env_criterion(imagined_reward, target_reward)
            loss = image_loss + self.env_reward_coef * reward_loss
            loss.backward()
            self.env_optimizer.step()
            
            # losses.append(loss.item()) # used in interval sampling method
        # # vvvvvvvvvvvv interval sampling method  vvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # # aとbが近い時、intervalが大きく環境からのサンプルが遅くなる
        # # aとbの差が大きい時、intervalが小さく、現在の環境から多くサンプリングする
        # a = self.env_loss
        # b = sum(losses)/len(losses)
        # self.env_learn_interval = round(abs(a//(a-b)))
        # self.env_loss = b # env_lossを更新する
        # # ^^^^^^^^^^^ interval sampling method   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
class ImaginationCore():
    def __init__(self, num_rollouts, in_shape, num_actions, num_rewards, env_model, distill_policy, full_rollout=True):
        self.num_rollouts  = num_rollouts
        self.in_shape      = in_shape
        self.num_actions   = num_actions
        self.num_rewards   = num_rewards
        self.env_model     = env_model
        self.distill_policy = distill_policy
        self.full_rollout  = full_rollout
        
    def __call__(self, state):
        state      = state
        batch_size = state.size(0)

        rollout_states  = []
        rollout_rewards = []

        if self.full_rollout:
            state = state.unsqueeze(0).repeat(self.num_actions, 1, 1, 1, 1).view(-1, *self.in_shape)
            action = LongTensor([[i] for i in range(self.num_actions)]*batch_size)
            rollout_batch_size = batch_size * self.num_actions
        else:
            with torch.no_grad():
                action = self.distill_policy.act(state)
            action = action.data.cpu()
            rollout_batch_size = batch_size

        for step in range(self.num_rollouts):
            inputs = self.env_model.get_inputs(state, action)
            imagined_state, imagined_reward = self.env_model.get_imagined(inputs)
            
            imagined_reward = FloatTensor(imagined_reward)
            imagined_state = FloatTensor(imagined_state).reshape(-1,3,15,19)
            
            rollout_states.append(imagined_state.unsqueeze(0))
            rollout_rewards.append(imagined_reward.unsqueeze(0))

            state  = imagined_state
            with torch.no_grad():
                action = self.distill_policy.act(state)
        
        return torch.cat(rollout_states), torch.cat(rollout_rewards)


