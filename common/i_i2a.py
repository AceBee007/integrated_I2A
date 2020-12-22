import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from actor_critic import OnPolicy, RolloutStorage
from i2a import RolloutEncoder, ImaginationCore

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor

class integrated_I2A(OnPolicy):
    def __init__(self, in_shape, num_actions, num_rewards, mode='regular', env_model, env_storage_capa=100, num_envs=16, reset_env_model=False, rollout_depth=3, rollout_breadth=3, rollout_method='MonteCarlo'):
        # have method: act
        # delete hidden_size, hidden_sizeはGRU(LSTM)用のパラメータなので不要？
        # rollout_method: 'MonteCarlo':モンテカルロ法、全部は自分のmodelfreeのポリシーで決定する
        # 'deterministic': 常に自分のmodelfreeでそのstateの上位{breadth}を探索
        # 'random': ランダムで探索
        super().__init__()
        
        self.in_shape      = in_shape
        self.num_actions   = num_actions
        self.num_rewards   = num_rewards
        self.avaliable_actions = np.array(list(range(num_actions)))
        
        self.env_model = env_model
        self.rollout_method = rollout_method
        self.planning_memo = {}
        self.mode = mode
        
        self.env_storage = RolloutStorage(env_storage_capa, num_envs, self.in_shape)
        self.env_learn_interval = 1
        self.env_loss = float('inf')
        self.current_frame = 0
        self.criterion = nn.CrossEntropyLoss()
        self.env_optimizer = optim.Adam(self.env_model.parameters())
        
        
        self.features = nn.Sequential(
            nn.Conv2d(in_shape[0], 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        
        # self.encoder = RolloutEncoder(in_shape, num_rewards, hidden_size)
        # 
        # if full_rollout:
        #     self.fc = nn.Sequential(
        #         nn.Linear(self.feature_size() + num_actions * hidden_size, 256),
        #         nn.ReLU(),
        #     )
        # else:
        #     self.fc = nn.Sequential(
        #         nn.Linear(self.feature_size() + hidden_size, 256),
        #         nn.ReLU(),
        #     )
        
        self.critic  = nn.Linear(256, 1)
        self.actor   = nn.Linear(256, num_actions)
        
    def forward(self, state):
        batch_size = state.size(0)

        planned_action = self.plan(state)
        #imagined_state, imagined_reward = self.imagination(state.data)
        #hidden = self.encoder(Variable(imagined_state), Variable(imagined_reward))
        #hidden = hidden.view(batch_size, -1)
        
        state = self.features(state)
        state = state.view(state.size(0), -1)
        
        #x = torch.cat([state, hidden], 1)
        x = self.fc(state)
        
        logit = self.actor(x)
        value = self.critic(x)
        
        return logit, value


    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.in_shape))).view(1, -1).size(1)

    def get_action(self, state, deterministic=False):
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
            return [np.random.choice(self.avaliable_actions) for _ in range(self.rollout_breadth)]
        elif self.rollout_method == 'MonteCarlo':
            return [self.get_action(state, deterministic=False) for _ in range(self.rollout_breadth)]
        else: # method == 'deterministic'
            with torch.no_grad():
                logit, value = self.forward(state)
                probs = F.softmax(logit, dim=1)
                sorted_probs = sorted([(i, probs[i]) for i in range(probs)], key=lambda x:x[1], reversed=True)
                actions = [i[0] for i in sorted_probs[:self.rollout_breadth]]
            return actions
    
    def plan(self, state):
        self.planning_memo = {}
        rewards, actions = self.imagine(state)
        if self.rollout_method == 'deterministic':
            return actions[rewards.argmax()]
        else:
            return np.random.choice(actions, p=rewards)
        
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
                imagined_state, imagined_reward = self.env_model(state, action)
            good_reward, good_action = self.imagine(imagined_state, route+str(action), depth+1)
            np.append(rewards, imagined_reward+good_reward.sum())
        self.planning_memo[route]=(rewards, actions)
        return self.planning_memo[route]
            
    def store_trajectory(self, state, action, reward, mask):
        # 毎回、env.step(action)のあと、これを呼び出して、intervalの間隔でenv_storageに履歴を備蓄する
        if self.current_frame % self.env_learn_interval == 0:
            self.env_storage.insert(self.current_frame, state, action_data, reward, masks)
        self.current_frame += 1
    
    def train_env(self):
        # 毎回、store_trajectoryの後、len(env_storage)がmaxに達したかを確認、達したら、これを実行
        losses = []
        for step in range(len(self.env_storage)):
            state = FloatTensor(self.env_storage.states[step])
            next_states = FloatTensor(self.env_storage.states[step+1])
            actions = LongTensor(self.env_storage.actions[step])
            rewards = FloatTensor(self.env_storage.rewards[step])
            batch_size = state.size(0)
            onehot_actions = torch.zeros(batch_size, self.num_actions, *self.in_shape)
            onehot_actions[range(batch_size), actions] = 1
            env_inputs = torch.act([states, onehot_actions], 1)
            if USE_CUDA:
                env_inputs = env_inputs.cuda()
            
            imagined_state, imagined_reward = env_model(env_inputs)
            
            target_state = LongTensor(pix_to_target(next_states))
            target_reward = LongTensor(reward_to_target(self.mode, rewards))

            self.env_optimizer.zero_grad()
            image_loss  = self.criterion(imagined_state, target_state)
            reward_loss = self.criterion(imagined_reward, target_reward)
            loss = image_loss + reward_coef * reward_loss
            loss.backward()
            self.env_optimizer.step()
                                    
            losses.append(loss.item())
        # aとbが近い時、intervalが大きく環境からのサンプルが遅くなる
        # aとbの差が大きい時、intervalが小さく、現在の環境から多くサンプリングする
        a = self.env_loss
        b = sum(losses)/len(losses)
        self.env_learn_interval = round(abs(a//(a-b)))
        self.env_loss = b # env_lossを更新する
        self.env_storage.after_update()




