"""
Markov Reward Process, MRP
"""
import numpy as np

np.random.seed(0)
# 定义状态转移概率矩阵P
P = [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]
P = np.array(P)

rewards = [-1, -2, -2, 10, 1, 0]  # 定义奖励函数
gamma = 0.5  # 定义折扣因子


# 给定一条序列,计算从某个索引（起始状态）开始到序列最后（终止状态）得到的回报
def compute_return(start_index, chain, _gamma):
    g = 0
    # for i in reversed(range(start_index, len(chain))):
    #     print(f"i = {i}, chain[i] = {chain[i]}, g = {g}", end=", ")
    #     g = _gamma * g + rewards[chain[i] - 1]
    #     print(f"{g} = {_gamma} * {g} + {rewards[chain[i] - 1]}")
    """
    书中的源码如上（循环内只有中间那一行），他是翻转了列表，函数体并不便于理解。
    所以我重新写了一下for循环内的函数体，这样可以更好的跟计算回报的公式契合。
    """
    for i in range(start_index, len(chain) - 1):
        print(f"{g} += pow({_gamma}, {i}) * {rewards[chain[i] - 1]}",
              end=", ")
        g += pow(_gamma, i) * rewards[chain[i] - 1]
        print(f"g = {g}")
    return g


def main_return():
    # 一个状态序列,s1-s2-s3-s6
    chain = [1, 2, 3, 6]
    start_index = 0
    g = compute_return(start_index, chain, gamma)
    print("根据本序列计算得到回报为：%s。" % g)


def compute(p, _rewards, _gamma, states_num):
    """
    利用贝尔曼方程的矩阵形式计算解析解,states_num是MRP的状态数
    """
    # 将rewards写成列向量形式
    _rewards = np.array(_rewards).reshape((-1, 1))
    value = np.dot(np.linalg.inv(
        np.eye(states_num, states_num) - _gamma * p), _rewards)
    return value


def main_value_function():
    v = compute(P, rewards, gamma, 6)
    print("MRP中每个状态价值分别为\n", v)


if __name__ == '__main__':
    # main_return()
    main_value_function()
