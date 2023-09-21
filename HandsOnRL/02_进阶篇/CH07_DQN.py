import random
import gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as fun
import matplotlib.pyplot as plt
from HandsOnRL.rl_utils import ReplayBuffer, moving_average


class QNet(torch.nn.Module):
    """
    只有一层隐藏层的Q网络
    """

    # dimensionality：维度
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # x = fun.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        # return self.fc2(x)
        out = self.fc1(x)
        out = fun.relu(out)
        out = self.fc2(out)
        return out


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim,
                 learning_rate, gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        # Q网络
        self.q_net = QNet(state_dim, hidden_dim, action_dim).to(device)
        # 目标网络
        self.target_q_net = QNet(state_dim, hidden_dim, action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        # 探索
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        # 利用
        else:
            # UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow.
            # Please consider converting the list to a single numpy.ndarray with
            # numpy.array() before converting to a tensor.
            # [state] ---> np.array([state])
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
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device.type, device)

    env_name = "CartPole-v0"
    env = gym.make(env_name)
    num_seed = 0
    random.seed(num_seed)
    np.random.seed(num_seed)
    env.seed(num_seed)
    torch.manual_seed(num_seed)

    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"env.observation_space = {env.observation_space}\n"
          f"env.observation_space.shape = {env.observation_space.shape}\n"
          f"env.observation_space.shape[0] = {env.observation_space.shape[0]}\n"
          f"env.action_space.n = {env.action_space.n}")
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)

    return_list = []
    # 进度条的总数量
    num_tqdm = 10
    # 每一个进度条的长度
    length_tqdm = int(num_episodes / num_tqdm)
    for i in range(num_tqdm):
        with tqdm(total=length_tqdm, desc='Iteration %d' % i) as pbar:
            for i_episode in range(length_tqdm):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    # observation, reward, done, info
                    next_state, reward, done, info = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        # 选择 return_list 最后 10 个数据，然后计算平均值
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    print("---------------------")
    episodes_list = list(range(len(return_list)))
    # episodes_list = np.arange(1, len(return_list) + 1)
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()


class ConvolutionalQnet(torch.nn.Module):
    """
    加入卷积层的Q网络
    """

    def __init__(self, action_dim, in_channels=4):
        super(ConvolutionalQnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = torch.nn.Linear(7 * 7 * 64, 512)
        self.head = torch.nn.Linear(512, action_dim)

    def forward(self, x):
        x = x / 255
        x = fun.relu(self.conv1(x))
        x = fun.relu(self.conv2(x))
        x = fun.relu(self.conv3(x))
        x = fun.relu(self.fc4(x))
        return self.head(x)


if __name__ == '__main__':
    main()
