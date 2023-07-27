"""
时序差分算法：无模型，在线学习
Sarsa算法：根据S,A,R,S,A计算更新量
"""
import numpy as np
import random
from IPython import display
import time


# 获取一个格子的状态
def get_state(row, col):
    if row != 3:
        return 'ground'
    if row == 3 and col == 0:
        return 'ground'
    if row == 3 and col == 11:
        return 'terminal'
    return 'trap'


# 在一个格子里做一个动作
def move(row, col, action):
    # 如果当前已经在陷阱或者终点，则不能执行任何动作
    if get_state(row, col) in ['trap', 'terminal']:
        return row, col, 0
    # ↑
    if action == 0:
        row -= 1
    # ↓
    if action == 1:
        row += 1
    # ←
    if action == 2:
        col -= 1
    # →
    if action == 3:
        col += 1

    # 不允许走到地图外面去
    row = max(0, row)
    row = min(3, row)
    col = max(0, col)
    col = min(11, col)
    # 是陷阱的话，奖励是-100，否则都是-1
    reward = -1
    if get_state(row, col) == 'trap':
        reward = -100
    return row, col, reward


# 根据状态选择一个动作
def get_action(row, col):
    # 有小概率选择随机动作
    if random.random() < 0.1:
        return random.choice(range(4))
    # 否则选择分数最高的动作
    return Q[row, col].argmax()


# 更新分数，每次更新取决于当前的格子，当前的动作，下个格子，和下个格子的动作
def get_update(row, col, action, reward, next_row, next_col, next_action):
    # 计算target
    target = 0.9 * Q[next_row, next_col, next_action]
    target += reward
    # 计算value
    value = Q[row, col, action]
    # 根据时序差分算法,当前state,action的分数 = 下一个state,action的分数*gamma + reward
    # 此处是求两者的差,越接近0越好
    update = target - value
    # 这个0.1相当于lr
    update *= 0.1
    # 更新当前状态和动作的分数
    return update


# 训练
# 在线学习，走一步算一步
def train():
    for epoch in range(1500):
        # 初始化当前位置
        row = random.choice(range(4))
        col = 0
        # 初始化第一个动作
        action = get_action(row, col)
        # 计算反馈的和，这个数字应该越来越小
        reward_sum = 0
        # 循环直到到达终点或者掉进陷阱
        while get_state(row, col) not in ['terminal', 'trap']:
            # 执行动作
            next_row, next_col, reward = move(row, col, action)
            reward_sum += reward
            # 求新位置的动作
            next_action = get_action(next_row, next_col)
            # 更新分数
            update = get_update(row, col, action, reward, next_row, next_col, next_action)
            Q[row, col, action] += update
            # 更新当前位置
            row = next_row
            col = next_col
            action = next_action
        if epoch % 150 == 0:
            print(epoch, reward_sum)


# 打印游戏，方便测试
def show(row, col, action):
    # □ 口
    graph = [
        '口', '口', '口', '口', '口', '口', '口', '口', '口', '口', '口', '口',
        '口', '口', '口', '口', '口', '口', '口', '口', '口', '口', '口', '口',
        '口', '口', '口', '口', '口', '口', '口', '口', '口', '口', '口', '口',
        '口', '○', '○', '○', '○', '○', '○', '○', '○', '○', '○', '❤'
    ]
    action = {0: '↑', 1: '↓', 2: '←', 3: '→'}[action]
    graph[row * 12 + col] = action
    graph = ''.join(graph)
    for i in range(0, 4 * 12, 12):
        print(graph[i:i + 12])
    print("--------------------")


def test():
    # 起点
    row = random.choice(range(4))
    col = 0
    # 最多玩N步
    for _ in range(200):
        # 获取当前状态，如果状态是终点或者掉陷阱则终止
        if get_state(row, col) in ['trap', 'terminal']:
            break
        # 选择最优动作
        action = Q[row, col].argmax()
        # 打印这个动作
        display.clear_output(wait=True)
        time.sleep(0.1)
        show(row, col, action)
        # 执行动作
        row, col, reward = move(row, col, action)


# 初始化在每一个格子里采取每个动作的分数,初始化都是0,因为没有任何的知识
Q = np.zeros([4, 12, 4])


def main():
    train()
    test()
    # 打印所有格子的动作倾向
    for row in range(4):
        line = ''
        for col in range(12):
            action = Q[row, col].argmax()
            action = {0: '↑', 1: '↓', 2: '←', 3: '→'}[action]
            line += action
        print(line)


if __name__ == '__main__':
    main()
