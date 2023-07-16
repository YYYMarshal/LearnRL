"""
递减的贪婪算法，探索的欲望逐渐降低
"""

import numpy as np
import random

# 每个老虎机的中奖概率,0-1之间的均匀分布
probs = np.random.uniform(size=10)
# 记录每个老虎机的返回值
rewards = [[1] for _ in range(10)]
print(f"probs = {probs}")
print(f"rewards = {rewards}")


# 随机选择的概率递减的贪婪算法
# 只有这个函数做了改动
def choose_one():
    # 求出现在已经玩了多少次了
    played_count = sum([len(i) for i in rewards])

    # 随机选择的概率逐渐下降
    if random.random() < 1 / played_count:
        return random.randint(0, 9)

    # 计算每个老虎机的奖励平均
    rewards_mean = [np.mean(i) for i in rewards]

    # 选择期望奖励估值最大的拉杆
    return np.argmax(rewards_mean)


def try_and_play():
    i = choose_one()

    # 玩老虎机,得到结果
    reward = 0
    if random.random() < probs[i]:
        reward = 1

    # 记录玩的结果
    rewards[i].append(reward)


def get_result():
    # 玩N次
    for _ in range(5000):
        try_and_play()

    # 期望的最好结果
    target = max(probs) * 5000

    # 实际玩出的结果
    result = sum([sum(i) for i in rewards])

    return target, result


def main():
    target, result = get_result()
    print(f"target = {target}, result = {result}")
    print(f"difference value = {target - result}")


main()
