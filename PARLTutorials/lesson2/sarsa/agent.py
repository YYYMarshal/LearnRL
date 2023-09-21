#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-

import numpy as np


class SarsaAgent(object):
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        self.act_n = act_n  # 动作维度，有几个动作可选
        self.lr = learning_rate  # 学习率
        self.gamma = gamma  # reward的衰减率
        self.epsilon = e_greed  # 按一定概率随机选动作
        self.Q = np.zeros((obs_n, act_n))  # Q表格

    def sample(self, obs):
        """
        采样：epsilon-greed 算法\n
        根据输入观察值，采样输出的动作值，带探索。\n
        :param obs: observation
        :return: action
        """
        # argmax_a Q(a): 概率 = 1 - epsilon，利用
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):  # 根据table的Q值选动作
            action = self.predict(obs)
        # 随机选择一个行动：概率 = epsilon，探索
        else:
            action = np.random.choice(self.act_n)  # 有一定概率随机探索选取一个动作
        return action

    # 根据输入观察值，预测输出的动作值
    def predict(self, obs):
        q_list = self.Q[obs, :]
        max_q = np.max(q_list)
        action_list = np.where(q_list == max_q)[0]  # maxQ可能对应多个action
        action = np.random.choice(action_list)
        return action

    def learn(self, obs, action, reward, next_obs, next_action, done):
        """
        学习方法，也就是更新Q-table的方法 \n
        on-policy \n
        obs: 交互前的obs, s_t \n
        action: 本次交互选择的action, a_t \n
        reward: 本次动作获得的奖励r \n
        next_obs: 本次交互后的obs, s_t+1 \n
        next_action: 根据当前Q表格, 针对next_obs会选择的动作, a_t+1 \n
        done: episode是否结束 \n
        """
        predict_q = self.Q[obs, action]
        if done:
            target_q = reward  # 没有下一个状态了
        else:
            target_q = reward + self.gamma * self.Q[next_obs, next_action]  # Sarsa
        self.Q[obs, action] += self.lr * (target_q - predict_q)  # 修正q

    def save(self):
        npy_file = './q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')
