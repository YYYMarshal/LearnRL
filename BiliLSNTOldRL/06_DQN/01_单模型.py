import gym
from matplotlib import pyplot as plt


# 定义环境
class MyWrapper(gym.Wrapper):

    def __init__(self):
        env = gym.make('CartPole-v1', render_mode='rgb_array')
        super().__init__(env)
        self.env = env
        self.step_n = 0

    def reset(self):
        state = self.env.reset()
        self.step_n = 0
        return state

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.step_n += 1
        if self.step_n >= 200:
            done = True
        return state, reward, done, info


# 打印游戏
def show(env):
    plt.imshow(env.render())
    plt.show()


def main():
    env = MyWrapper()
    env.reset()
    show(env)


if __name__ == '__main__':
    main()
