# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch (tested in PyTorch 0.2.0 and 0.3.0)

@author: Junxiao Song
""" 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#定义网络模型类
class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()
        #棋盘的长宽
        self.board_width = board_width
        self.board_height = board_height
        # common layers 首先是三层的公共dd额卷积神经网络层
        #第一层使用32个卷积滤波器，滤波器大小为3，stride为1，padding为1
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        #第二层使用64个卷积滤波器，滤波器大小为3，stride为1，padding为1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) 
        #第三层使用128个卷积滤波器，滤波器大小为3，stride为1，padding为1
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        #接着分为两部分：走子概率p和局面评估值v
        # action policy layers
        #走子概率部分 策略层 先经过一个卷积神经网络层，再经过一个全连接层，最后输出走子概率p
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height, board_width*board_height)
        # state value layers
        #局面评估值部分 价值层 先经过一个卷积神经网络层，再经过两个全连接层，最后输出走子概率p
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, board_width*board_height)
        self.val_fc2 = nn.Linear(board_width*board_height, 1)

        #原先的
        # self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        # self.val_fc2 = nn.Linear(64, 1)
    
    def forward(self, state_input):
        #函数激励层，把卷积层的输出结果做非线性映射
        #每层卷积神经网络的输出都要经过RELU激活函数进行非线性转换，从而增强对棋盘局面的表达能力
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers（走子概率部分）
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)#重构张量的维度
        x_act = F.log_softmax(self.act_fc1(x_act),dim=1)  # x_act = F.log_softmax(self.act_fc1(x_act))  modified by haward 2018/03/12
        # state value layers（局面评估部分）
        x_val = F.relu(self.val_conv1(x))
        #view：重构张量的维度，全连接层时需要将高维度数据平铺变为低拉数据
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        #最后将全连接层的输出用激活函数tanh映射到[-1，1]之间
        #改值代表着在当前局面下最终取得的胜率
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height, net_params=None, use_gpu=False):        #use_gpu=False
        #是否使用GPU
        self.use_gpu = use_gpu
        #棋盘的宽
        self.board_width = board_width
        #棋盘的高度
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty 
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()     # 这里用了上面的Net()
        else:
            self.policy_value_net = Net(board_width, board_height)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

        if net_params:
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values 
        """
        #判断是否使用GPU进行训练
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())            
            return act_probs, value.data.numpy()
        

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, 4, self.board_width, self.board_height))
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        #判断是否用GPU进行训练
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)
        
        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2 (Note: the L2 penalty is incorporated in optimizer)
        value_loss = F.mse_loss(value.view(-1), winner_batch)  #nn.mes_loss(),均方根损失函数 MSELoss
        #value.view(-1)转成一维的v（策略价值网络输出的局面评估值）
        #winner_batch：z，蒙特卡洛树输出的局面评估值
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        #使用loss.backward()进行误差反向传播，计算得到loss后就要传回损失
        #需要清除已经存在的梯度值，否则梯度会积累到现有的梯度上
        loss.backward()
        #执行单个优化步骤（参数更新）。回传损失过程中会计算梯度，然后需要根据这些梯度更新参数
        #optimizer.step()就是用来更新参数的
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.item() , entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params
