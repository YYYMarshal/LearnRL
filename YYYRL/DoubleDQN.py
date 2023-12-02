from DQN import train
import utility
import matplotlib.pyplot as plt


def main():
    start_time = utility.get_current_time()
    algorithm = "DoubleDQN"
    env_name = "CartPole-v0"
    return_list = train(env_name, algorithm)
    utility.time_difference(start_time)
    utility.plot(return_list, algorithm, env_name)


if __name__ == '__main__':
    main()
