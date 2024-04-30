import numpy as np


def train_on_policy_agent(env, agent, num_episodes, is_render=False, interval=10):
    # num_episodes 次 Episode 的 Reward 的列表集合
    episode_reward_list = []
    # 每运行 num_episodes * (1/interval) 次打印一次信息、显示画面（可选）
    part = num_episodes / interval
    for i_episode in range(num_episodes):
        # 每一个 Episode 的总 Reward
        episode_reward = 0
        state = env.reset()
        done = False
        is_print = i_episode % part == 0
        transition_dict = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
        while not done:
            if is_render and is_print:
                env.render()
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['rewards'].append(reward)
            transition_dict['next_states'].append(next_state)
            transition_dict['dones'].append(done)
            state = next_state
            episode_reward += reward
        episode_reward_list.append(episode_reward)
        agent.update(transition_dict)
        if is_print:
            print(f"{i_episode}/{num_episodes}, episode_reward = {episode_reward}")

    print("---------------------")
    print(f"Episode Reward List 的平均值 = {np.mean(episode_reward_list)}")
    return episode_reward_list


def train_off_policy_agent(env, agent, num_episodes,
                           replay_buffer, minimal_size, batch_size,
                           is_render=False, interval=10):
    episode_reward_list = []
    part = num_episodes / interval
    for i_episode in range(num_episodes):
        episode_reward = 0
        state = env.reset()
        done = False
        is_print = i_episode % part == 0
        while not done:
            if is_render and is_print:
                env.render()
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            # 当buffer数据的数量超过一定值后,才进行Q网络训练
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'rewards': b_r,
                    'next_states': b_ns,
                    'dones': b_d}
                agent.update(transition_dict)
        episode_reward_list.append(episode_reward)
        env.close()
        if is_print:
            print(f"{i_episode}/{num_episodes}, episode_reward = {episode_reward}")

    print("---------------------")
    print(f"Episode Reward List 的平均值 = {np.mean(episode_reward_list)}")
    return episode_reward_list
