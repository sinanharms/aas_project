import gym
import numpy as np

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    # play 10 episodes of Lunar Lander with a random agent
    total_reward_history = []
    for i in range(10):
        state = env.reset()
        total_reward = 0
        while True:
            action = env.action_space.sample()
            state, reward, term, info = env.step(action)
            total_reward += reward
            env.render()
            if term:
                total_reward_history.append(total_reward)
                break
    env.close()

    avg_reward = np.mean(total_reward_history)
    print(f"Average Reward before training: {avg_reward}")

