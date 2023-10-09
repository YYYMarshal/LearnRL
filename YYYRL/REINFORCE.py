import gym
import torch
import torch.nn.functional as fun
import numpy as np
import datetime
from YYYRL.utility import HyperParameters, train_on_policy_agent, moving_average, OnPolicyTransition, plot


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = fun.relu(self.fc1(x))
        return fun.softmax(self.fc2(x), dim=1)


class REINFORCE:
    def __init__(self, state_dim, action_dim, device, params: HyperParameters):
        self.policy_net = PolicyNet(state_dim, params.hidden_dim, action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=params.lr)
        self.gamma = params.gamma  # 折扣因子
        self.device = device

    def take_action(self, state):  # 根据动作概率分布随机采样
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition: OnPolicyTransition):
        g = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(transition.reward_list))):  # 从最后一步算起
            reward = transition.reward_list[i]
            state = torch.tensor(np.array([transition.state_list[i]]), dtype=torch.float).to(self.device)
            action = torch.tensor([transition.action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            g = self.gamma * g + reward
            loss = -log_prob * g  # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降


def main():
    print(datetime.datetime.now())

    params = HyperParameters()
    params.num_episodes = 1000
    params.lr = 1e-3

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device.type = {device.type}, device = {device}")

    env_name = "CartPole-v0"
    env = gym.make(env_name)

    env.seed(params.num_seed)
    torch.manual_seed(params.num_seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCE(state_dim, action_dim, device, params)

    return_list = train_on_policy_agent(env, agent, params)

    print(datetime.datetime.now())

    xlabel = "Episodes"
    ylabel = "Returns"
    title = f"REINFORCE on {env_name}"

    episodes_list = list(range(len(return_list)))
    plot(episodes_list, return_list, xlabel, ylabel, title)

    mv_return = moving_average(return_list, 9)
    plot(episodes_list, mv_return, xlabel, ylabel, title)


if __name__ == '__main__':
    main()
