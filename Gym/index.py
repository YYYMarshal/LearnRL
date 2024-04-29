import gym
from gym.utils.env_checker import check_env
from gym.spaces import Box, Discrete, Dict, Tuple, MultiBinary, MultiDiscrete
import numpy as np
from gym.wrappers import RescaleAction  # , ResizeObservation
from gym.utils.play import play  # , PlayPlot
import pygame


def play_game(env_name: str, num_episodes=100):
    # 创建游戏场景
    env = gym.make(env_name, render_mode='human')
    for episode in range(num_episodes):
        print(f"episode = {episode + 1}")
        env.reset()
        while True:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            over = terminated or truncated
            if over:
                break
    # 关闭游戏
    env.close()


def interacting_with_the_environment():
    env = gym.make("LunarLander-v2", render_mode="human")
    env.action_space.seed(42)
    env.reset(seed=42)
    for _ in range(1000):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
    env.close()


def checking_api_conformity():
    env = gym.make("LunarLander-v2", render_mode="human")
    check_env(env)


def spaces():
    observation_space = Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float32)
    print(observation_space.sample())
    observation_space = Discrete(4)
    print(observation_space.sample())
    observation_space = Discrete(5, start=-2)
    print(observation_space.sample())
    observation_space = Dict({"position": Discrete(2), "velocity": Discrete(3)})
    print(observation_space.sample())
    observation_space = Tuple((Discrete(2), Discrete(3)))
    print(observation_space.sample())
    observation_space = MultiBinary(5)
    print(observation_space.sample())
    observation_space = MultiDiscrete([5, 2, 2])
    print(observation_space.sample())


def wrappers():
    base_env = gym.make("BipedalWalker-v3")
    print(base_env.action_space)
    print(base_env.action_space.sample())
    wrapped_env = RescaleAction(base_env, min_action=0, max_action=1)
    print(wrapped_env.action_space)


def playing_within_an_environment():
    # env = gym.make('Pong-v4', render_mode="rgb_array_list")
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array_list")
    env.metadata['render_fps'] = 30
    # play(env)
    mapping = {(pygame.K_SPACE,): 1,
               (pygame.K_UP,): 2,
               (pygame.K_DOWN,): 3}
    play(env, keys_to_action=mapping)


def playing_cart_pole():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
    play(env, keys_to_action=mapping)


def playing_car_racing():
    env = gym.make("CarRacing-v2", render_mode="rgb_array", continuous=False)
    # The discrete action space has 5 actions: [do nothing, left, right, gas, brake].
    mapping = {(pygame.K_LEFT,): 2,
               (pygame.K_RIGHT,): 1,
               (pygame.K_UP,): 3,
               (pygame.K_DOWN,): 4,
               }
    play(env, keys_to_action=mapping)


def playing_ms_pacman():
    env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
    mapping = {
        (pygame.K_SPACE,): 0,

        (pygame.K_UP,): 1,
        (pygame.K_RIGHT,): 2,
        (pygame.K_LEFT,): 3,
        (pygame.K_DOWN,): 4,

        (pygame.K_UP, pygame.K_RIGHT): 5,
        (pygame.K_UP, pygame.K_LEFT): 6,
        (pygame.K_DOWN, pygame.K_RIGHT): 7,
        (pygame.K_DOWN, pygame.K_LEFT): 8,
    }
    play(env, keys_to_action=mapping, zoom=5)


def playing_mountain_car():
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    mapping = {
        (pygame.K_RIGHT,): 2,
        (pygame.K_LEFT,): 0,
        (pygame.K_DOWN,): 1
    }
    play(env, keys_to_action=mapping)
    # play(env)


def env_info():
    # env_name = "Acrobot-v1"
    env_name = "CartPole-v1"
    # env_name = "MountainCarContinuous-v0"
    env = gym.make(env_name)
    print(env.observation_space.shape, env.action_space)
    state_dim = env.observation_space.shape[0]
    # 连续动作空间的环境不能使用下面的 .n
    action_dim = env.action_space.n
    state_high_bound = env.observation_space.high
    state_low_bound = env.observation_space.low
    print(state_dim, action_dim)
    print(state_high_bound)
    print(state_low_bound)


def main():
    # play_game("LunarLander-v2", 10)
    # play_game("Adventure-v4", 10)
    # interacting_with_the_environment()
    # checking_api_conformity()
    # spaces()
    # wrappers()
    # playing_within_an_environment()
    # playing_cart_pole()
    # play_game("CarRacing-v2")
    # playing_car_racing()
    # playing_ms_pacman()
    # playing_mountain_car()
    env_info()


if __name__ == '__main__':
    main()
