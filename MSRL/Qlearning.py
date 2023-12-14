import gym
from matplotlib import pyplot as plt
import numpy as np
from IPython import display
import random


# import time


# 定义环境
class MyWrapper(gym.Wrapper):

    def __init__(self):
        # is_slippery控制会不会滑
        env = gym.make('FrozenLake-v1', render_mode='rgb_array', is_slippery=False)
        # env = gym.make('FrozenLake-v1', render_mode="human", is_slippery=False)
        super().__init__(env)
        self.env = env

    def reset(self):
        state, _ = self.env.reset()
        return state

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        over = terminated or truncated
        # 走一步扣一分,逼迫机器人尽快结束游戏
        if not over:
            reward = -1
        # 掉坑扣100分
        if over and reward == 0:
            reward = -100
        return state, reward, over

    # 打印游戏图像
    def show(self):
        plt.figure(figsize=(6, 6))
        plt.imshow(self.env.render())
        plt.show(block=False)

        plt.pause(0.5)
        plt.close()


# 玩一局游戏并记录数据
def play(env: MyWrapper, q_table, is_show=False):
    data = []
    reward_sum = 0

    state = env.reset()
    over = False
    step = 0
    epsilon = 0.1
    while not over:
        action = q_table[state].argmax()
        if random.random() < epsilon:
            action = env.action_space.sample()
        step += 1
        # str_action = ["←", "↑", "↓", "→"]
        # print(f"action {step} = {action}, {str_action[action]}")
        next_state, reward, over = env.step(action)

        data.append((state, action, reward, next_state, over))
        reward_sum += reward

        state = next_state

        if is_show:
            display.clear_output(wait=True)
            env.show()
            # env.render()
    return data, reward_sum


# 数据池
class Pool:
    def __init__(self):
        self.pool_list = []

    def __len__(self):
        return len(self.pool_list)

    def __getitem__(self, i):
        return self.pool_list[i]

    # 更新动作池
    def update(self, env, q_table, is_show=False):
        # 每次更新不少于N条新数据
        # old_len：记录一下进入 update 函数时，pool_list 的长度
        old_len = len(self.pool_list)
        while len(self.pool_list) - old_len < 200:
            # print(f"len(self.pool_list) = {len(self.pool_list)}, old_len = {old_len}")
            data, reward_sum = play(env, q_table, is_show)
            # print(f"len(data) = {len(data)}")
            self.pool_list.extend(data)
            # print(data)
            # print(self.pool_list)
            # print("--------")
        # 只保留最新的N条数据
        # Python 3.6 及之后，1_0000 和 10000 是一样的。
        self.pool_list = self.pool_list[-1_0000:]

    # 获取一批数据样本
    def sample(self):
        return random.choice(self.pool_list)


# 训练
def train(pool, env, q_table):
    gamma = 0.9
    lr = 0.1
    # 共更新N轮数据
    for epoch in range(1000):
        pool.update(env, q_table)

        # 每次更新数据后,训练N次
        for i in range(200):
            # 随机抽一条数据
            state, action, reward, next_state, over = pool.sample()

            # Q矩阵当前估计的state下action的价值
            value = q_table[state, action]

            # 实际玩了之后得到的reward + 下一个状态的价值 * gamma
            target = reward + q_table[next_state].max() * gamma

            # value和target应该是相等的,说明Q矩阵的评估准确
            # 如果有误差,则应该以target为准更新Q表,修正它的偏差
            # 这就是TD误差,指评估值之间的偏差,以实际成分高的评估为准进行修正
            update = (target - value) * lr

            # 更新Q表
            q_table[state, action] += update

        if epoch % 100 == 0:
            print(epoch, len(pool), play(env, q_table)[-1])


def main():
    env = MyWrapper()
    env.reset()
    # env.show()
    # 初始化Q表,定义了每个状态下每个动作的价值
    # 地图：4*4=16，动作：上下左右4个
    q_table = np.zeros((16, 4))
    print(q_table)
    pool = Pool()
    pool.update(env, q_table)
    print(f"len(pool) = {len(pool)}, pool[0] = {pool[0]}")
    train(pool, env, q_table)
    print(q_table)
    data, reward_sum = play(env, q_table, True)
    print(reward_sum)


if __name__ == "__main__":
    main()
