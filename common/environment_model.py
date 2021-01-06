import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor

class BasicBlock(nn.Module):
    def __init__(self, in_shape, n1, n2, n3):
        super().__init__()
        
        self.in_shape = in_shape
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        
        self.maxpool = nn.MaxPool2d(kernel_size=in_shape[1:])
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_shape[0] * 2, n1, kernel_size=1, stride=2, padding=6),
            nn.LeakyReLU(),
            nn.Conv2d(n1, n1, kernel_size=10, stride=1, padding=(5, 6)),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_shape[0] * 2, n2, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(n2, n2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(n1 + n2,  n3, kernel_size=1),
            nn.LeakyReLU()
        )
        
    def forward(self, inputs):
        x = self.pool_and_inject(inputs)
        x = torch.cat([self.conv1(x), self.conv2(x)], 1)
        x = self.conv3(x)
        x = torch.cat([x, inputs], 1)
        return x
    
    def pool_and_inject(self, x):
        pooled     = self.maxpool(x)
        tiled      = pooled.expand((x.size(0),) + self.in_shape)
        out        = torch.cat([tiled, x], 1)
        return out

# in "Imagination Augmented Agents" they NO conv layer at the beginning of this module
class PoolAndInject(torch.nn.Module):
    def __init__(self,W,H,use_cuda):
        super(PoolAndInject, self).__init__()
        self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.W = W
        self.H = H
        self.pool = nn.MaxPool2d(kernel_size=(W,H))

    def forward(self,input):
        x = F.leaky_relu_(self.pool(input))   #max-pool
        x = x.repeat(1, 1, self.W, self.H)  #tile
        return x + input


class BasicBlock2(torch.nn.Module):
    def __init__(self,num_inputs,n1,n2,n3,W,H,use_cuda):
        super(BasicBlock, self).__init__()
        self.pool_and_inject = PoolAndInject(W,H,use_cuda)

        input_channels = num_inputs
        # pool-and-inject layer is size-preserving therefore num_inputs is the input to conv1
        self.left_conv1 = nn.Conv2d(input_channels, n1, kernel_size=1)
        self.left_conv2 = nn.Conv2d(n1, n1, kernel_size=9, padding=4)
        self.right_conv1 = nn.Conv2d(input_channels, n2, kernel_size=1)
        self.right_conv2 = nn.Conv2d(n2, n2, kernel_size=3, padding=1)
        # input after cat is output size of left side + output size of right side = n1 + n2
        self.conv3 = nn.Conv2d(n1+n2, n3, kernel_size=1)
        self.apply(xavier_weights_init_relu)

    def forward(self,input):
        x = self.pool_and_inject(input)
        left_side = F.leaky_relu_(self.left_conv2(F.leaky_relu_(self.left_conv1(x))))
        right_side = F.leaky_relu_(self.right_conv2(F.leaky_relu_(self.right_conv1(x))))
        x = torch.cat((left_side,right_side),1)
        x = F.leaky_relu_(self.conv3(x))
        return x + input


class EnvModel(nn.Module):
    def __init__(self, in_shape, num_pixels, num_envs=16, num_rewards=5, num_actions=5):
        super(EnvModel, self).__init__()
        self.num_pixels = num_pixels
        self.num_rewards = num_rewards
        self.num_actions = num_actions
        if self.num_pixels == 8:
            self.mode = 'onehot'
        elif self.num_pixels == 7:
            self.mode = 'old_onehot'
        elif self.num_pixels == 3:# also supports "rgb" with num_pixles == 3
            self.mode = 'rgb'
        else:
            self.mode='unknown'
            
        self.pixels = np.array([
            (0.0, 1.0, 1.0), # bluegreen/ cyan PowerPill
            (0.0, 1.0, 0.0), #green Pillman
            (0.0, 0.0, 1.0), #Blue Food/Pill
            (1.0, 1.0, 1.0), # White Walls
            (1.0, 1.0, 0.0),  # Yellow   edible Ghost
            (0.0, 0.0, 0.0),  # Black, empty route
            (1.0, 0.0, 0.0),  # Red  Ghost
        ])
        self.bin_pixels = np.array([
            [0,0,0], # 0, 黒, route
            [0,0,1], # 1, 青, food
            [0,1,0], # 2, 緑, Pacman
            [0,1,1], # 3, cyan,青緑, PowerPill
            [1,0,0], # 4, 赤, ghost
            [1,0,1], # 5, 紫, 対応なし
            [1,1,0], # 6, 黃, 食用Ghost
            [1,1,1]  # 7, 白、壁
        ]).astype(float)
        self.bin_factor = np.array([4,2,1])
        self.tensor_bin_factor = LongTensor([4,2,1])
        self.in_shape = in_shape
        width  = in_shape[1]
        height = in_shape[2]
        
        # self.conv = nn.Sequential(
        #     nn.Conv2d(8, 64, kernel_size=1),
        #     nn.ReLU()
        # )
        # 
        # self.basic_block1 = BasicBlock((64, width, height), 16, 32, 64)
        # self.basic_block2 = BasicBlock((128, width, height), 16, 32, 64)
        # 
        # self.image_conv = nn.Sequential(
        #     nn.Conv2d(192, 256, kernel_size=1),
        #     nn.ReLU()
        # )
        # self.image_fc = nn.Linear(256, num_pixels)
        # 
        # self.reward_conv = nn.Sequential(
        #     nn.Conv2d(192, 64, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=1),
        #     nn.ReLU()
        # )
        # self.reward_fc    = nn.Linear(64 * width * height, num_rewards)
        
        self.conv = torch.jit.script(nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=1),
            nn.ReLU()
        ))
      
        self.basic_block1 = torch.jit.script(BasicBlock((64, width, height), 16, 32, 64))
        self.basic_block2 = torch.jit.script(BasicBlock((128, width, height), 16, 32, 64))
      
        self.image_conv = torch.jit.script(nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=1),
            nn.ReLU()
        ))
        self.image_fc = torch.jit.script(nn.Linear(256, num_pixels))
      
        self.reward_conv = torch.jit.script(nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU()
        ))
        self.reward_fc    = torch.jit.script(nn.Linear(64 * width * height, num_rewards))
        

    def forward(self, inputs):
        batch_size = inputs.size(0)
        
        x = self.conv(inputs)
        x = self.basic_block1(x)
        x = self.basic_block2(x)
        
        image = self.image_conv(x)
        image = image.permute(0, 2, 3, 1).contiguous().view(-1, 256)
        image = self.image_fc(image)

        reward = self.reward_conv(x)
        reward = reward.view(batch_size, -1)
        reward = self.reward_fc(reward)
        
        return image, reward
    
    def get_imagined(self, inputs):
        '''
        Input:  inputs: Tensor (8,15,19)
        Output: ndarray(15,19,3), ndarray(5,)
        get inputs to calculate imagined image and rewards
        '''
        with torch.no_grad():
            tensor_image, tensor_reward = self.forward(inputs)
            np_reward = tensor_reward.data.cpu().numpy()
            if self.mode == 'onehot':
                np_image = np.array([self.bin_pixels[i] for i in torch.argmax(tensor_image, dim=1).data.cpu().numpy()]).reshape(15,19,3)
            elif self.mode == 'old_onehot':
                np_image = np.array([self.pixels[i] for i in torch.argmax(tensor_image, dim=1).data.cpu().numpy()]).reshape(15,19,3)
            elif self.mode == 'rgb':
                np_image = np.clip(tensor_image.reshape(15,19,3).data.cpu().numpy(), a_min=0.0, a_max=1.0)
        return np_image, np_reward
    
    def get_inputs(self, states, actions):
        '''
        Input:  states:  Tensor or ndarray (num_envs, 15, 19, 3)
                actions: Tensor or ndarray (num_envs, 5)
        Output: Tensor (8, 15, 19)
        get states and actions, return processed inputs for environment model
        '''
        actions     = LongTensor(actions)
        if states.ndim == 4: # serial states
            states = FloatTensor(states)
            batch_size = states.size(0)
        else: # only one state
            states = FloatTensor(states).unsqueeze(0)
            batch_size = 1
        if USE_CUDA:
            onehot_actions = torch.zeros(batch_size, self.num_actions, *self.in_shape[1:]).cuda()
        else:
            onehot_actions = torch.zeros(batch_size, self.num_actions, *self.in_shape[1:])
        onehot_actions[range(batch_size), actions] = 1
        ret = torch.cat([states, onehot_actions], 1)
        if USE_CUDA:
            ret = ret.cuda()
        return ret
    
    
    def process_reward(self, reward, mode_reward):
        '''
        Input:  reward: ndarray/Tensor (5, )
                mode_reward: ndarray   (5, )
        Output: float32
        '''
        if type(reward) ==torch.Tensor:
            return (reward.data.cpu().numpy()*mode_reward).sum(axis=1)
        else:
            return (reward*mode_reward).sum(axis=1)
        
    def states_to_onehot_tensor(self, states, cate):
        # cateはonehotで変換する時のカテゴリ数今回、3ビットの01情報なので、8、
        # old_onehotの時、cate=7になる
        if type(states) ==torch.Tensor:
            want = (states.transpose(1,2).transpose(2,3).reshape(-1,3)*self.tensor_bin_factor).sum(dim=1).long()
            res = torch.eye(8)[want]
            if USE_CUDA:
                res = res.cuda()
            return res
        else:
            want = states.transpose(0,2,3,1).reshape(-1,3)
            res = (want*self.bin_factor).sum(axis=1)
            return np.eye(cate)[res.astype(int)]
        
    def state_to_target(self, env_states):
        '''
        Input:  ndarray (num_envs, 15,19,3)
        Output: Tensor (num_envs, [8/7/3],15,19)
        8 and 7 means onehot target tensor
        3 means rgb target tensor
        '''
        if self.mode == 'onehot':
            target_state = self.states_to_onehot_tensor(env_states, cate=8)
        elif self.mode == 'rgb':
            target_state = env_states.transpose(0,2,3,1).reshape(self.num_envs*15*19, 3)
        elif self.mode == 'old_onehot':
            target_state = self.states_to_onehot_tensor(env_states, cate=7)
        return FloatTensor(target_state)


class EnvModelRolloutStorage(object):
    def __init__(self, storage_capa, num_envs, state_shape, reward_shape=5):
        self.storage_capa = storage_capa
        self.num_envs  = num_envs
        self.reward_shape = reward_shape
        self.states       = torch.zeros(self.storage_capa, num_envs, *state_shape)
        self.actions      = torch.zeros(self.storage_capa, num_envs, 1).long()
        self.next_states  = torch.zeros(self.storage_capa, num_envs, *state_shape)
        self.rewards      = torch.zeros(self.storage_capa, num_envs, self.reward_shape)
        
        self.use_cuda = False
        if USE_CUDA:
            self.cuda()
            
    def cuda(self):
        self.use_cuda  = True
        self.states    = self.states.cuda()
        self.actions   = self.actions.cuda()
        self.next_states    = self.next_states.cuda()
        self.rewards   = self.rewards.cuda()
        
    def insert(self,step, state, action, next_state, reward):
        self.states[step].copy_(state)
        self.actions[step].copy_(action)
        self.next_states[step].copy_(next_state)
        self.rewards[step].copy_(reward)
    
    def __len__(self):
        return len(self.rewards)
