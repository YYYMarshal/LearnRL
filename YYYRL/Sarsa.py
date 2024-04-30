import numpy as np
from Utility.EnvWrappers import FrozenLakeWrapper
from Utility.Plot import plot


class Sarsa:
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

    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        # td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

    def show_result(self, row, col):
        sym = ["←", "↓", "→", "↑"]
        best_actions = np.argmax(self.Q_table, axis=1)
        for i in range(row):
            for j in range(col):
                best_action = best_actions[row * i + j]
                print(sym[best_action], end=" ")
            print()


def main():
    env = FrozenLakeWrapper()
    # env = gym.make("CliffWalking-v0")
    env.render()

    np.random.seed(0)
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    num_episodes = 500  # 智能体在环境中运行的序列的数量

    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    print(f"状态空间数量：{state_dim}，动作空间数量：{action_dim}")

    agent = Sarsa(state_dim, action_dim, epsilon, alpha, gamma)
    # print(agent.Q_table)

    all_episode_reward_list = []  # 记录每一条序列的总奖励
    for episode in range(num_episodes):
        episode_reward_list = 0
        state = env.reset()
        action = agent.take_action(state)
        done = False
        while not done:
            # print("--------------")
            # env.render()
            next_state, reward, done, info = env.step(action)
            next_action = agent.take_action(next_state)
            episode_reward_list += reward
            agent.update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
        # print(episode_reward_list)
        all_episode_reward_list.append(episode_reward_list)

    # print(agent.Q_table)
    # CliffWalking-v0: 4, 12
    agent.show_result(4, 4)
    # 31804
    print(f"总奖励 = {np.sum(all_episode_reward_list)}")
    plot(all_episode_reward_list, "Sarsa", "FrozonLake-v0", is_plot_average=True)


if __name__ == "__main__":
    main()
