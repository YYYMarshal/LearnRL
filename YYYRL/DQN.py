import random
import gym
import numpy as np
import torch
import torch.nn.functional as func
import utility


class QNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        """
        只有一层隐藏层（fc1）的Q网络，fc2是输出层
        """
        super(QNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        定义了前向传播方法，用于定义模型的正向计算
        """
        x = func.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate,
                 gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = QNet(state_dim, hidden_dim, self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = QNet(state_dim, hidden_dim, self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):
        """
        epsilon-贪婪策略采取动作
        """
        # 探索
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        # 利用
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(func.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1


def train(env_name: str):
    lr = 2e-3
    num_episodes = 500
    # 当 hidden_dim = 128 时：mean_DQN = 141.316
    # 当 hidden_dim = 512 时：mean_DQN = 153.216
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    # 当 buffer 数据的数量超过 minimal_size 后,才进行Q网络训练
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = gym.make(env_name)
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"device: {device}\n"
          f"env_name = {env_name}, state_dim = {state_dim}, action_dim = {action_dim}")

    agent = DQN(state_dim, hidden_dim, action_dim, lr,
                gamma, epsilon, target_update, device)
    replay_buffer = utility.ReplayBuffer(buffer_size)
    return_list = utility.train_off_policy_agent(env, agent, num_episodes, replay_buffer,
                                                 minimal_size, batch_size, True)
    return return_list


def main():
    start_time = utility.get_current_time()
    algorithm = "DQN"
    env_name = "CartPole-v0"
    return_list = train(env_name)
    utility.time_difference(start_time)
    utility.plot(return_list, algorithm, env_name)


if __name__ == '__main__':
    main()
