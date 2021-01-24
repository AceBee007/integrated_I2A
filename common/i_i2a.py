import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from .actor_critic import OnPolicy, RolloutStorage
from .environment_model import EnvModelRolloutStorage
from .utils import process_reward, np_softmax, np_deque_append

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor 

class Integrated_I2A(OnPolicy):
    
    def __init__(self, in_shape, num_actions, env_model,  
                 mode_reward,
                 mode='RegularMiniPacmanNoFrameskip-v0',
                 num_rewards = 5,
                 env_storage_capa=10, 
                 num_envs=16, 
                 reset_env_model=False, 
                 rollout_depth=3, 
                 rollout_breadth=3, 
                 rollout_method='MonteCarlo'):
        # have method: act
        # delete hidden_size, hidden_sizeはGRU(LSTM)用のパラメータなので不要？
        # rollout_method: 'MonteCarlo':モンテカルロ法、全部は自分のmodelfreeのポリシーで決定する
        # 'deterministic': 常に自分のmodelfreeでそのstateの上位{breadth}を探索
        # 'random': ランダムで探索
        super().__init__()
        
        self.in_shape      = in_shape
        self.num_actions   = num_actions
        self.num_rewards   = num_rewards
        self.num_envs = num_envs
        self.avaliable_actions = np.array(list(range(num_actions)))
        
        self.env_model = env_model
        self.rollout_depth = rollout_depth
        self.rollout_breadth = rollout_breadth
        self.rollout_method = rollout_method
        if self.rollout_method not in {'MonteCarlo', 'random', 'deterministic'}:
            raise ValueError
        self.planning_memo = {}
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
        
        
        # self.features = nn.Sequential(
        #     nn.Conv2d(in_shape[0], 16, kernel_size=3, stride=1),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=2),
        #     nn.LeakyReLU(),
        # )
        # self.fc = nn.Sequential(
        #     nn.Linear(self.feature_size(), 256),
        #     nn.LeakyReLU(),
        # )
        # 
        # self.critic  = nn.Linear(256, 1)
        # self.actor   = nn.Linear(256, num_actions)
        
        self.features = torch.jit.script(nn.Sequential(
            nn.Conv2d(in_shape[0], 16, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.LeakyReLU(),
        ))
        self.fc = torch.jit.script(nn.Sequential(
            nn.Linear(self.feature_size(), 256),
            nn.LeakyReLU(),
        ))
      
        self.critic  = torch.jit.script(nn.Linear(256, 1))
        self.actor   = torch.jit.script(nn.Linear(256, num_actions))
    
    def forward(self, state):
        state = self.features(state)
        state = state.view(state.size(0), -1)
        
        x = self.fc(state)
        
        logit = self.actor(x)
        value = self.critic(x)
        
        return logit, value
    
    def act(self, state, deterministic=False):
        logit, value = self.forward(state)
        probs = F.softmax(logit, dim=1)
        
        if deterministic:
            action = probs.max(1)[1]
        else:
            action = probs.multinomial(1)
        
        return action
    
    def get_planned_action(self, state):
        if state.ndim == 4:
            planned_action = LongTensor([self.plan(state[i]) for i in range(self.num_envs)]).view(-1,1)
        else:
            planned_action = self.plan(state)
        return planned_action

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.in_shape))).view(1, -1).size(1)
    
    def get_action(self, state, deterministic=False):
        if type(state)==torch.Tensor:
            if state.ndim == 4:
                state = state
            else:
                state = state.unsqueeze(0)
        else:
            if state.ndim == 4:
                state = FloatTensor(np.float32(state))
            else:
                state = FloatTensor(np.float32(state)).unsqueeze(0)
        with torch.no_grad():
            action = self.act(state, deterministic)
        return action.data.cpu().squeeze(1).numpy()

    def rollout(self, state):
        # 一つのstateについて、{rollout_breadth}回分のactionを生成
        if self.rollout_method == 'random':
            return [np.array([np.random.choice(self.avaliable_actions)]) for _ in range(self.rollout_breadth)]
        elif self.rollout_method == 'MonteCarlo':
            return [self.get_action(state, deterministic=False) for _ in range(self.rollout_breadth)]
        else: # method == 'deterministic'
            with torch.no_grad():
                logit, value = self.forward(FloatTensor(state).unsqueeze(0))
                probs = F.softmax(logit, dim=1).data.cpu().numpy()[0]
                sorted_probs = sorted([(i, probs[i]) for i in range(len(probs))], key=lambda x:x[1], reverse=True)
                actions = [np.array([i[0]]) for i in sorted_probs[:self.rollout_breadth]]
            return actions
    
    def plan(self, state):
        self.planning_memo = {}
        rewards, actions = self.imagine(state,'',0)
        if self.rollout_method == 'deterministic':
            return actions[rewards.argmax()][0]
        else:
            return np.random.choice(np.array(actions).flatten(), p=np_softmax(np.array(rewards)))
    
    def imagine(self, state, route='', depth=0):
        # 再帰的探索していく
        # return rewards list and actions list
        if route in self.planning_memo:
            return self.planning_memo[route]
        if depth == self.rollout_depth:
            return np.array([0]), np.array([0])
        rewards = np.array([])
        actions = self.rollout(state)
        for action in actions:
            with torch.no_grad():
                inputs = self.env_model.get_inputs(state, action)
                imagined_state, raw_imagined_reward = self.env_model.get_imagined(inputs)
                imagined_state = imagined_state.transpose(2,0,1)
                imagined_reward = process_reward(raw_imagined_reward, self.mode_reward)
            good_reward, good_action = self.imagine(imagined_state, route+str(action), depth=depth+1)
            rewards = np.append(rewards, imagined_reward+good_reward.sum())
        self.planning_memo[route]=(rewards, actions)
        return self.planning_memo[route]
    
    
    # # for method interval sampling method
    # def store_trajectory(self, state, action, next_state,  reward):
    #     # 毎回、env.step(action)のあと、これを呼び出して、intervalの間隔でenv_storageに履歴を備蓄する
    #     if self.current_frame % self.env_learn_interval == 0:
    #         self.env_storage.insert(self.current_storage_len, state, action, next_state, reward)
    #         self.current_storage_len += 1 # 履歴の長さを更新
    #     self.current_frame = (self.current_frame+1)%self.env_learn_interval
    #     # 履歴がenv_storage_capaに達したら、学習
    #     if self.current_storage_len+1 == self.env_storage_capa:
    #         self.train_env()
            
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
        


