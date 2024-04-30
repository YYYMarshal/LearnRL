import numpy as np
import matplotlib.pyplot as plt
import gym


class Qlearning:
    def __init__(self, state_dim, action_dim, epsilon, alpha, gamma):
        self.Q_table = np.zeros([state_dim, action_dim])  # 初始化Q(s,a)表格
        self.action_dim = action_dim  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数

    def take_action(self, state):  # 选取下一步的操作,具体实现为epsilon-贪婪
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            action = np.argmax(self.Q_table[state])
        # print(action)
        return action

    def update(self, s0, a0, r, s1):
        # td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

    def show_result(self, row, col):
        sym = ["←", "↓", "→", "↑"]
        best_actions = np.argmax(self.Q_table, axis=1)
        for i in range(row):
            for j in range(col):
                best_action = best_actions[row * i + j]
                print(sym[best_action], end=" ")
            print()


class FrozenLakeWrapper(gym.Wrapper):
    def __init__(self):
        # is_slippery 控制会不会滑
        env = gym.make('FrozenLake-v0', is_slippery=False)
        super().__init__(env)
        self.env = env

    def step(self, action):
        """
        Reward schedule:
        Reach goal(G): +1
        Reach hole(H): 0
        Reach frozen(F): 0
        """
        next_state, reward, done, info = self.env.step(action)
        # 走一步扣一分,逼迫智能体尽快结束游戏
        if not done:
            reward = -1
        # 掉坑(H, hole)扣100分
        if done and reward == 0:
            reward = -100
        # 走到终点，得一分，修改为得100分。
        if reward == 1:
            reward = 100
        return next_state, reward, done, info


def main():
    env = FrozenLakeWrapper()
    env.render()
    np.random.seed(0)
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    print(state_dim, action_dim)
    agent = Qlearning(state_dim, action_dim, epsilon, alpha, gamma)
    # print(agent.Q_table)

    num_episodes = 500  # 智能体在环境中运行的序列的数量
    return_list = []  # 记录每一条序列的回报
    for i_episode in range(num_episodes):
        episode_return = 0
        state = env.reset()
        done = False
        while not done:
            # env.render()
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            episode_return += reward
            agent.update(state, action, reward, next_state)
            state = next_state
        # print(episode_return)
        return_list.append(episode_return)

    # print(agent.Q_table)
    agent.show_result(4, 4)
    # 35417
    print(np.sum(return_list))
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Qlearning on {}'.format('FrozenLake-v0'))
    plt.show()


if __name__ == "__main__":
    main()
