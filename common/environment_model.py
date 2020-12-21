import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_shape, n1, n2, n3):
        super(BasicBlock, self).__init__()
        
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
    def __init__(self, in_shape, num_pixels, num_rewardsi=5):
        super(EnvModel, self).__init__()
        
        width  = in_shape[1]
        height = in_shape[2]
        
        self.conv = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=1),
            nn.LeakyReLU()
        )
        
        self.basic_block1 = BasicBlock((64, width, height), 16, 32, 64)
        self.basic_block2 = BasicBlock((128, width, height), 16, 32, 64)
        
        self.image_conv = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=1),
            nn.LeakyReLU()
        )
        self.image_fc = nn.Linear(256, num_pixels)
        
        self.reward_conv = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.LeakyReLU()
        )
        self.reward_fc    = nn.Linear(64 * width * height, num_rewards)
        
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

