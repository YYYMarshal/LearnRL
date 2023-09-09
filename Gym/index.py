import gym


def play_game(name: str, num_episodes=100):
    # 创建游戏场景
    env = gym.make(name, render_mode='human')
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


def main():
    play_game("LunarLander-v2", 20)


if __name__ == '__main__':
    main()
