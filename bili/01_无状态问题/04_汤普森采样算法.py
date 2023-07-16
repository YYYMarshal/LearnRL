"""
使用beta分布衡量期望
"""
import numpy as np
import random

# 每个老虎机的中奖概率,0-1之间的均匀分布
probs = np.random.uniform(size=10)
# 记录每个老虎机的返回值
rewards = [[1] for _ in range(10)]
print(f"probs = {probs}")
print(f"rewards = {rewards}")

# beta分布测试
print('当数字小的时候，beta分布的概率有很大的随机性')
for _ in range(5):
    print(np.random.beta(1, 1))

print('当数字大时，beta分布逐渐稳定')
for _ in range(5):
    print(np.random.beta(1e5, 1e5))


def choose_one():
    # 求出每个老虎机出1的次数+1
    count_1 = [sum(i) + 1 for i in rewards]

    # 求出每个老虎机出0的次数+1
    count_0 = [sum(1 - np.array(i)) + 1 for i in rewards]

    # 按照beta分布计算奖励分布,这可以认为是每一台老虎机中奖的概率
    beta = np.random.beta(count_1, count_0)

    return beta.argmax()


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
