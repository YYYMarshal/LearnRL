import random
import datetime
import gym
import numpy as np
import torch
import torch.nn.functional as fun
from YYYRL.utility import HyperParameters, train_off_policy_agent, plot, moving_average


class QNet(torch.nn.Module):
    """
    只有一层隐藏层的Q网络
    """

    # dimensionality：维度
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, out):
        # 隐藏层使用ReLU激活函数
        out = fun.relu(self.fc1(out))
        return self.fc2(out)


class DQN:
    def __init__(self, state_dim, action_dim, device, params: HyperParameters):
        self.action_dim = action_dim
        # Q网络
        self.q_net = QNet(state_dim, params.hidden_dim, action_dim).to(device)
        # 目标网络
        self.target_q_net = QNet(state_dim, params.hidden_dim, action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=params.lr)
        self.gamma = params.gamma  # 折扣因子
        self.epsilon = params.epsilon  # epsilon-贪婪策略
        self.target_update = params.target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        # 探索
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        # 利用
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, b_states, b_actions, b_rewards, b_next_states, b_dones):
        states = torch.tensor(b_states, dtype=torch.float).to(self.device)
        actions = torch.tensor(b_actions).view(-1, 1).to(self.device)
        rewards = torch.tensor(b_rewards, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(b_next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(b_dones, dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        # TD误差目标
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(fun.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1


def main():
    print(datetime.datetime.now())
    params = HyperParameters()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device.type = {device.type}, device = {device}")

    env_name = "CartPole-v0"
    # env_name = "Acrobot-v1"
    # env_name = "MountainCar-v0"
    # env_name = "LunarLander-v2"
    env = gym.make(env_name)

    num_seed = params.num_seed
    random.seed(num_seed)
    np.random.seed(num_seed)
    env.seed(num_seed)
    torch.manual_seed(num_seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"env_name = {env_name}\n"
          f"env.observation_space = {env.observation_space}\n"
          f"env.observation_space.shape = {env.observation_space.shape}\n"
          f"env.observation_space.shape[0] = {env.observation_space.shape[0]}\n"
          f"env.action_space.n = {env.action_space.n}")

    agent = DQN(state_dim, action_dim, device, params)
    params.num_episodes = 500
    return_list = train_off_policy_agent(env, agent, params, True)
    print("---------------------")
    print(f"mean = {np.mean(return_list)}")

    xlabel = "Episodes"
    ylabel = "Returns"
    title = f"DQN on {env_name}"

    episodes_list = list(range(len(return_list)))
    plot(episodes_list, return_list, xlabel, ylabel, title)
    mv_return = moving_average(return_list, 9)
    plot(episodes_list, mv_return, xlabel, ylabel, title)


if __name__ == '__main__':
    main()
