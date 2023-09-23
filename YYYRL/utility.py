import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import collections
import random

from tqdm import tqdm
import torch


class ReplayBuffer:
    """
    经验回放池
    """

    def __init__(self, capacity):
        # 队列,先进先出
        self.buffer = collections.deque(maxlen=capacity)

    # 将数据加入buffer
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # 从buffer中采样数据,数量为batch_size
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return np.array(states), actions, rewards, np.array(next_states), dones

    # 目前buffer中数据的数量
    def size(self):
        return len(self.buffer)


class HyperParameters:
    """
    超参数，每个超参数都有默认值，也可以在声明对象后，再指定其某个属性的值。
    """
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    num_seed = 0


def moving_average(y_list, window_size):
    cumulative_sum = np.cumsum(np.insert(y_list, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(y_list[:window_size - 1])[::2] / r
    end = (np.cumsum(y_list[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def render(env, episode: int, is_render=False, interval_render=1):
    """
    注意：需要在调用该函数的循环体的外部执行 env.close()，以便关闭最后一次的 env.render()。
    """
    # interval_render 默认为 1，也就是每一次的运行画面都显示。
    if is_render:
        if episode % interval_render == 0:
            env.render()
        else:
            env.close()


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def train_off_policy_agent(env, agent, params: HyperParameters, is_render=False, interval_render=50):
    replay_buffer = ReplayBuffer(params.buffer_size)
    return_list = []
    for episode in range(1, params.num_episodes + 1):
        episode_return = 0
        state = env.reset()
        done = False
        while not done:
            render(env=env, episode=episode, is_render=is_render, interval_render=interval_render)

            action = agent.take_action(state)
            # observation, reward, done, info
            next_state, reward, done, info = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            # 当buffer数据的数量超过一定值后,才进行Q网络训练
            if replay_buffer.size() > params.minimal_size:
                b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample(params.batch_size)
                agent.update(b_states, b_actions, b_rewards, b_next_states, b_dones)
        print(f"episode = {episode}, episode_return = {episode_return}")
        return_list.append(episode_return)
    # 最后一次的 env.render() 没有关闭，所以在这里关闭一下。
    env.close()
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


def plot(x, y, xlabel, ylabel, title):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_multi(plot_list: [matplotlib.lines.Line2D], xlabel, ylabel, title):
    for item in plot_list:
        plt.plot(item.x, item.y, label=item.label, color=item.color, linestyle=item.linestyle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
