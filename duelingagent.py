import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import deque
from experiencereplaymemory import ExperienceReplay
from duelingnetwork import SimpleDuelingNetwork


class DuelingAgent:
    def __init__(self, env, batch_size=64, memory_size=100000, gamma=0.999, min_epsilon=0.01,
                 epsilon=0.6, replacement_freq=4, episodes=1000, learning_rate=0.0005, tau=0.001):
        """
        :param env: openAI gym environment
        :param batch_size: size of sample from memory default 64
        :param memory_size: default 100000
        :param gamma: reward decay parameter default 0.999
        :param min_epsilon: default 0.01
        :param epsilon: default 0.6
        :param replacement_freq: update frequency of the target network default 4
        :param episodes: default 1000
        :param learning_rate: default 0.0005
        :param tau: soft-update parameter default 0.001
        """
        self.env = env
        self.input_dims = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.replay_memory = ExperienceReplay(self.input_dims, memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.replacement_freq = replacement_freq
        self.batch_size = batch_size
        self.episodes = episodes
        self.learning_rate = learning_rate
        self.tau = tau
        self.step = 0

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.huber = tf.keras.losses.Huber()

        # initialize the dueling networks
        # two are needed one is a copy the original network named as target network. this is not trainable as its
        # weights get updated according to the replacement frequency
        self.network = SimpleDuelingNetwork(n_actions=self.n_actions)
        self.target_network = SimpleDuelingNetwork(n_actions=self.n_actions)

    def update_target(self):
        # soft update of the target network
        for t, e in zip(self.target_network.trainable_variables, self.network.trainable_variables):
            e = self.tau*e + (1-self.tau)*t
            t.assign(e)

    def epsilon_greedy(self, state):
        # epsilon greedy policy implementation
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
            return action
        else:
            q_values = self.network(tf.expand_dims(state, axis=0))
            return np.argmax(q_values)

    def epsilon_decay(self):
        self.epsilon = max(self.epsilon * 0.99, self.min_epsilon)

    def add_to_memory(self, state, action, reward, next_state, term):
        self.replay_memory.add_experience(state, action, reward, next_state, term)

    def get_sample_experience(self):
        state, action, reward, next_state, term = self.replay_memory.get_experience(self.batch_size)
        return state, action, reward, next_state, term

    def get_indices(self, actions):
        # helper function that maps the indices of the actions to their Q values to retrieve them
        arr = np.arange(self.batch_size)
        indices = tf.stack([arr, actions], axis=1)
        return indices

    def gradient_step(self):
        """
        Implementation true to the double dqn gradient descent step as in using the definition of
        a^max(s') = argmax_a' Q(s', a')
        Meaning it decouples the action selection from action evaluation -> reduces overestimating the Q-values
        """
        if self.replay_memory.counter < self.batch_size:
            return
        states, actions, rewards, next_states, term = self.replay_memory.get_experience(self.batch_size)
        # find next best action among all available actions
        next_actions = tf.argmax(self.network(next_states), axis=1, output_type=tf.dtypes.int32)
        n_a_i = self.get_indices(next_actions)
        # estimate Q-values for next states
        q_next = self.target_network(next_states)
        # match the Q-values to the next action
        q_next = tf.gather_nd(q_next, n_a_i)
        # calculate the target values y_i
        q_target = rewards + (1.0-term) * tf.math.multiply(self.gamma, q_next)
        a_i = self.get_indices(actions)
        with tf.GradientTape() as tape:
            q_current = self.network(states)
            q_current = tf.gather_nd(q_current, a_i)
            mse = tf.reduce_mean(self.huber(q_target, q_current))

        gradients = tape.gradient(mse, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        self.update_target()

    def train(self):
        """
        Main training loop of the agent. A few changes were made to optimize the procedure, the most significant being
        limiting the length of one episode to 500 instead of default 1000 steps
        :return: list of total reward and average reward history
        """
        solved = False
        # initializing moving average of the reward history
        mov_avg_reward = deque(maxlen=100)
        # initialize outputs
        total_reward_history = []
        avg_reward_history = []

        print("Starting Training ...")
        for episode in range(self.episodes):
            state = self.env.reset()
            total_reward = 0
            # restrict the maximum number of steps in the environment to 500
            for t in range(500):
                # choose an action according to policy
                action = self.epsilon_greedy(state)
                # make an action step in the env
                next_state, reward, term, _ = self.env.step(action)
                # sum up the rewards
                total_reward += reward
                # add current step to memory
                self.add_to_memory(state, action, reward, next_state, term)
                # increase the step
                self.step += 1
                # do a gradient step and replace the target network according to the update frequency
                if self.step % self.replacement_freq == 0:
                    self.gradient_step()
                # set state s as the next state s'
                state = next_state
                # end episode if terminal state is reached
                if term:
                    break
            # decrease the epsilon after each episode
            self.epsilon_decay()
            # save the results of the episode
            total_reward_history.append(total_reward)
            mov_avg_reward.append(total_reward)
            # calculate the average reward
            avg_reward = np.mean(mov_avg_reward)
            avg_reward_history.append(avg_reward)
            # print results of the episode
            print(f"episodes: {episode}/{self.episodes}, epsilon: {self.epsilon}, total_reward: {total_reward},"
                  f" avg_reward: {avg_reward}, "
                  f"gain: {avg_reward_history[-1] - avg_reward_history[-2] if len(avg_reward_history) >= 2 else 0}")
            # if the agent has accumulated an average reward of over 200 over the last 100 episodes the game is solved
            if episode >= 100 and avg_reward > 200:
                solved = True
                print(f"Game solved after {episode} episodes!")
                break
        if not solved:
            print("The game was not solved yet!")
        print(f"Average reward = {avg_reward}")
        print(f"Training ended! \n")

        return total_reward_history, avg_reward_history


# def grad_desc_step(self):
#     """
#     basically the same function as gradient_desc except for matching the next Q-values to the specific next action.
#     so it's just a regular dqn step , but does not decouple the action selection from the action
#     evaluation. Surprisingly it works just as well as
#     """
#     # gradient descent step
#     if self.replay_memory.counter < self.batch_size:
#         return
#
#     states, actions, rewards, next_states, term = self.replay_memory.get_experience(self.batch_size)
#     ind = self.get_indices(actions)
#
#     q_next = self.target_network(next_states)
#     max_q_next = tf.reduce_max(q_next, axis=1)
#
#     q_target = rewards + (1-term) * self.gamma * max_q_next
#
#     with tf.GradientTape() as tape:
#         q_pred = tf.gather_nd(self.network(states), ind)
#         loss = tf.reduce_mean(self.huber(q_target, q_pred))
#
#     gradients = tape.gradient(loss, self.network.trainable_variables)
#     self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
#     self.update_target()

