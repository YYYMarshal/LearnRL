import random
import datetime
import gym
import numpy as np
import torch
import torch.nn.functional as fun
from YYYRL.Version1.utility import HyperParameters, train_off_policy_agent, moving_average, OffPolicyTransition, plot
import matplotlib.pyplot as plt


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


class VANet(torch.nn.Module):
    """
    只有一层隐藏层的A网络和V网络
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VANet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 共享网络部分
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        a = self.fc_A(fun.relu(self.fc1(x)))
        v = self.fc_V(fun.relu(self.fc1(x)))
        q = v + a - a.mean(1).view(-1, 1)  # Q值由V值和A值计算得到
        return q


class DQN:
    def __init__(self, state_dim, action_dim, device, params: HyperParameters, dqn_type="DQN"):
        self.action_dim = action_dim
        hidden_dim = params.hidden_dim
        if dqn_type == "DuelingDQN":
            # Dueling DQN采取不一样的网络框架
            self.q_net = VANet(state_dim, hidden_dim, self.action_dim).to(device)
            self.target_q_net = VANet(state_dim, hidden_dim, self.action_dim).to(device)
        else:
            self.q_net = QNet(state_dim, hidden_dim, self.action_dim).to(device)
            self.target_q_net = QNet(state_dim, hidden_dim, self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=params.lr)
        self.gamma = params.gamma  # 折扣因子
        self.epsilon = params.epsilon  # epsilon-贪婪策略
        self.target_update = params.target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        self.dqn_type = dqn_type

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        # 探索
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        # 利用
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition: OffPolicyTransition):
        states = torch.tensor(transition.states, dtype=torch.float).to(self.device)
        actions = torch.tensor(transition.actions).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition.rewards, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition.next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(transition.dones, dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        if self.dqn_type == "DoubleDQN":
            # DQN 与 Double DQN 的区别
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:
            # DQN的情况
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


def all_dqn():
    print(datetime.datetime.now())
    params = HyperParameters()
    params.num_episodes = 1000
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
    return_list_dqn = train_off_policy_agent(env, agent, params)
    print(f"Time: {datetime.datetime.now()}")
    print("---------------------")
    agent = DQN(state_dim, action_dim, device, params, dqn_type="DoubleDQN")
    return_list_double_dqn = train_off_policy_agent(env, agent, params)
    print(f"Time: {datetime.datetime.now()}")
    print("---------------------")
    agent = DQN(state_dim, action_dim, device, params, dqn_type="DuelingDQN")
    return_list_dueling_dqn = train_off_policy_agent(env, agent, params)
    print(f"Time: {datetime.datetime.now()}")
    print("---------------------")

    print(f"mean_DQN = {np.mean(return_list_dqn)}")
    print(f"mean_DoubleDQN = {np.mean(return_list_double_dqn)}")
    print(f"mean_DuelingDQN = {np.mean(return_list_dueling_dqn)}")

    xlabel = "Episodes"
    ylabel = "Returns"
    title = f"DQN/DoubleDQN/DuleingDQN on {env_name}"

    episodes_list = list(range(len(return_list_dqn)))

    plt.plot(episodes_list, return_list_dqn, label="DQN", color="blue")
    plt.plot(episodes_list, return_list_double_dqn, label="DoubleDQN", color="green")
    plt.plot(episodes_list, return_list_dueling_dqn, label="DuelingDQN", color="red")
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

    mv_return_dqn = moving_average(return_list_dqn, 9)
    mv_return_double_dqn = moving_average(return_list_double_dqn, 9)
    mv_return_dueling_dqn = moving_average(return_list_dueling_dqn, 9)

    plt.plot(episodes_list, mv_return_dqn, label="DQN", color="blue")
    plt.plot(episodes_list, mv_return_double_dqn, label="DoubleDQN", color="green")
    plt.plot(episodes_list, mv_return_dueling_dqn, label="DuelingDQN", color="red")
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def single_dqn(dqn_type="DQN"):
    print(f"Time: {datetime.datetime.now()}")
    params = HyperParameters()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device.type = {device.type}, device = {device}")

    env_name = "CartPole-v0"
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

    agent = DQN(state_dim, action_dim, device, params, dqn_type=dqn_type)
    return_list = train_off_policy_agent(env, agent, params)

    print("---------------------")
    print(f"Time: {datetime.datetime.now()}")
    print(f"mean_{dqn_type} = {np.mean(return_list)}")

    xlabel = "Episodes"
    ylabel = "Returns"
    title = f"{dqn_type} on {env_name}"

    episodes_list = list(range(len(return_list)))
    plot(episodes_list, return_list, xlabel, ylabel, title)

    mv_return_dqn = moving_average(return_list, 9)
    plot(episodes_list, mv_return_dqn, xlabel, ylabel, title)


def main():
    # DQN DoubleDQN DuelingDQN
    dqn_type = "DQN"
    choice = "x"
    all_dqn() if choice == "all" else single_dqn(dqn_type)


if __name__ == '__main__':
    main()
