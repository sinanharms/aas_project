import gym
import numpy as np
import matplotlib.pyplot as plt
from duelingagent import DuelingAgent


def plot_results(data, x_lab: str, y_lab: str, title: str):
    """
    Function to plot the results of the training
    :param data: list of the training results
    :param x_lab: name of the x-axis
    :param y_lab: name of y-axis
    :param title: title of the plot
    :return: plot
    """
    plt.figure()
    x = np.arange(0, len(data))
    y = data
    plt.plot(x, y)
    plt.xlabel(xlabel=x_lab)
    plt.ylabel(ylabel=y_lab)
    plt.title(title)
    # if needed change filepath
    plt.savefig(f"./MiniProjectReport/img/{title.strip(' ')}.png")
    plt.show(block=False)


if __name__ == "__main__":
    # initialize the environment
    env = gym.make("LunarLander-v2")
    # initialize agent with default args
    agent = DuelingAgent(env)
    # start training
    total_reward_history, avg_reward_history = agent.train()

    # plot the training results
    plot_results(total_reward_history, "Episode", "Reward", "Total Reward History")
    plot_results(avg_reward_history, "Episode", "Reward", "Average Reward History")

    # save weights of the trained network
    agent.network.save_weights("./modelweights/dueling")

