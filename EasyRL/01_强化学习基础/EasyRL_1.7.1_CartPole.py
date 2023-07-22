import gym  # 导入Gym 的Python 接口环境包

env = gym.make('CartPole-v1', render_mode='human')  # 构建实验环境
env.reset()  # 重置一个回合
for _ in range(1000):
    env.render()  # 显示图形界面
    action = env.action_space.sample()  # 从动作空间中随机选取一个动作
    observation, reward, done, info, _ = env.step(action)  # 用于提交动作，括号内是具体的动作
    if done:
        break
env.close()  # 关闭环境
