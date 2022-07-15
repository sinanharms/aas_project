import gym
import numpy as np
from duelingagent import DuelingAgent

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    # set up agent with epsilon = 0
    agent = DuelingAgent(env, epsilon=0)
    # load the trained network
    agent.network.load_weights("./modelweights/dueling")

    total_reward_history = []
    # play 10 episodes of a trained agent
    for i in range(10):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.epsilon_greedy(state)
            state, reward, term, info = env.step(action)
            total_reward += reward
            env.render()
            if term:
                total_reward_history.append(total_reward)
                break
    env.close()

    avg_reward = np.mean(total_reward_history)
    print(f"Average Reward of trained Agent: {avg_reward}")
