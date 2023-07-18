import gym
import pygame
from gym.utils.play import play


def show_info(env):
    # 游戏的动作空间
    print(env.action_space)
    # 游戏的状态空间
    print(env.observation_space.low)
    print(env.observation_space.high)
    # 反馈值空间
    print(env.reward_range)


def play_game(name):
    # 创建游戏华景
    env = gym.make(name, render_mode='human')
    show_info(env)
    # 初始化游戏
    env.reset()
    # 随机玩n个动作
    n = 200
    for i in range(n):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        over = terminated or truncated
        # 游戏结束了就重置
        if over:
            env.reset()
    # 关闭游戏
    env.close()


def main():
    # game_name = "CartPole-v1"
    game_name = "LunarLander-v2"
    play_game(game_name)


if __name__ == '__main__':
    main()
