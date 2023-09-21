import gym
import torch
import random
from IPython import display
from matplotlib import pyplot as plt

env = gym.make("CartPole-v1", render_mode="human")


def test_env():
    state = env.reset()
    print(f"state = {state}")
    print(f"env.action_space = {env.action_space}")
    action = env.action_space.sample()
    print(f"随机一个动作：{action}")
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"observation = {observation}")
    over = terminated or truncated
    print(f"reward = {reward}, over = {over}")
    # 游戏结束了就重置
    if over:
        env.reset()


def create_model():
    result_model = torch.nn.Sequential(
        torch.nn.Linear(4, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 2),
    )
    return result_model


# 计算动作的模型，也是真正要用的模型
model = create_model()
# 经验网络，用于评估一个状态的分数
next_model = create_model()
# 样本池
datas = []


def get_action(state):
    if random.random() < 0.01:
        return random.choice([0, 1])
    # 走神经网络，得到一个动作
    state = torch.FloatTensor(state).reshape(1, 4)
    return model(state).argmax().item()


# 向样本池中添加N条数据,删除M条最古老的数据
def update_data():
    old_count = len(datas)
    # 玩到新增了N个数据为止
    while len(datas) - old_count < 200:
        # 初始化游戏
        state = env.reset()
        # 玩到游戏结束为止
        over = False
        while not over:
            # 根据当前状态得到一个动作
            action = get_action(state)
            # 执行动作,得到反馈
            next_state, reward, over, truncated, info = env.step(action)
            # 记录数据样本
            datas.append((state, action, reward, next_state, over))
            # 更新游戏状态,开始下一个动作
            state = next_state
    update_count = len(datas) - old_count
    drop_count = max(len(datas) - 10000, 0)
    # 数据上限,超出时从最古老的开始删除
    while len(datas) > 10000:
        datas.pop(0)
    return update_count, drop_count


# 获取一批数据样本
def get_sample():
    # 从样本池中采样
    samples = random.sample(datas, 64)

    # [b, 4]
    state = torch.FloatTensor([i[0] for i in samples]).reshape(-1, 4)
    # [b, 1]
    action = torch.LongTensor([i[1] for i in samples]).reshape(-1, 1)
    # [b, 1]
    reward = torch.FloatTensor([i[2] for i in samples]).reshape(-1, 1)
    # [b, 4]
    next_state = torch.FloatTensor([i[3] for i in samples]).reshape(-1, 4)
    # [b, 1]
    over = torch.LongTensor([i[4] for i in samples]).reshape(-1, 1)

    return state, action, reward, next_state, over


def get_value(state, action):
    # 使用状态计算出动作的 logic
    # [b, 4] -> [b, 2]
    value = model(state)

    # 根据实际使用的action取出每一个值
    # 这个值就是模型评估的在该状态下,执行动作的分数
    # 在执行动作前,显然并不知道会得到的反馈和next_state
    # 所以这里不能也不需要考虑next_state和reward
    # [b, 2] -> [b, 1]
    value = value.gather(dim=1, index=action)

    return value


def get_target(reward, next_state, over):
    # 上面已经把模型认为的状态下执行动作的分数给评估出来了
    # 下面使用next_state和reward计算真实的分数
    # 针对一个状态,它到底应该多少分,可以使用以往模型积累的经验评估
    # 这也是没办法的办法,因为显然没有精确解,这里使用延迟更新的next_model评估

    # 使用next_state计算下一个状态的分数
    # [b, 4] -> [b, 2]
    with torch.no_grad():
        target = next_model(next_state)

    # 取所有动作中分数最大的
    # [b, 2] -> [b, 1]
    target = target.max(dim=1)[0]
    target = target.reshape(-1, 1)

    # 下一个状态的分数乘以一个系数,相当于权重
    target *= 0.98

    # 如果next_state已经游戏结束,则next_state的分数是0
    # 因为如果下一步已经游戏结束,显然不需要再继续玩下去,也就不需要考虑next_state了.
    # [b, 1] * [b, 1] -> [b, 1]
    target *= (1 - over)

    # 加上reward就是最终的分数
    # [b, 1] + [b, 1] -> [b, 1]
    target += reward

    return target


# 打印游戏
def show():
    plt.imshow(env.render())
    plt.show()


def test(play):
    # 初始化游戏
    state = env.reset()

    # 记录反馈值的和,这个值越大越好
    reward_sum = 0

    # 玩到游戏结束为止
    over = False
    while not over:
        # 根据当前状态得到一个动作
        action = get_action(state)

        # 执行动作,得到反馈
        state, reward, over, truncated, info = env.step(action)
        reward_sum += reward

        # 打印动画
        if play and random.random() < 0.2:  # 跳帧
            display.clear_output(wait=True)
            show()

    return reward_sum


def train():
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = torch.nn.MSELoss()

    # 训练N次
    for epoch in range(500):
        # 更新N条数据
        update_count, drop_count = update_data()

        # 每次更新过数据后,学习N次
        for i in range(200):
            # 采样一批数据
            state, action, reward, next_state, over = get_sample()

            # 计算一批样本的value和target
            value = get_value(state, action)
            target = get_target(reward, next_state, over)

            # 更新参数
            loss = loss_fn(value, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 把model的参数复制给next_model
            if (i + 1) % 10 == 0:
                next_model.load_state_dict(model.state_dict())

        if epoch % 50 == 0:
            test_result = sum([test(play=False) for _ in range(20)]) / 20
            print(epoch, len(datas), update_count, drop_count, test_result)


def main():
    test_env()

    # 把 model 的参数复制给 next_model
    next_model.load_state_dict(model.state_dict())
    print(model)
    print(next_model)
    # state, action, reward, next_state, over = get_sample()
    test(False)
    train()


if __name__ == '__main__':
    main()
