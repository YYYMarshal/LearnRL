"""
贪婪算法：大概率选择目前中奖率最高的，小概率随机探索
"""
import numpy as np
import random

# 每个老虎机的中奖概率,0-1之间的均匀分布
probs = np.random.uniform(size=10)
# 记录每个老虎机的返回值
rewards = [[1] for _ in range(10)]
print(f"probs = {probs}")
print(f"rewards = {rewards}")


# 贪婪算法（动作函数）
def choose_one():
    # 有小概率随机选择一根拉杆
    # 随机生成（0,1）之间的浮点数
    if random.random() < 0.01:
        # 左右都闭，均可取到
        return random.randint(0, 9)

    # 计算每个老虎机的奖励平均，rewards 是一个二维列表，所以 i = rewards[index] 是一个列表
    rewards_mean = [np.mean(i) for i in rewards]

    # 选择期望奖励估值最大的拉杆
    # np.argmax()是numpy中获取array的某一个维度中数值最大的那个元素的索引
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
    # print(f"choose_one = {choose_one()}")
    # try_and_play()
    # print(rewards)
    target, result = get_result()
    print(f"target = {target}, result = {result}")
    print(f"difference value = {target - result}")


main()
