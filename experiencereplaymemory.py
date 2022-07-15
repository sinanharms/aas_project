import numpy as np


class ExperienceReplay:
    def __init__(self, input_dims, memory_size):
        super(ExperienceReplay, self).__init__()
        self.maximum_mem_size = memory_size
        self.counter = 0

        # create transition memories
        self.state_mem = np.zeros((memory_size, input_dims), dtype=np.float32)
        self.action_mem = np.zeros(memory_size, dtype=np.int32)
        self.reward_mem = np.zeros(memory_size, dtype=np.float32)
        self.next_state_mem = np.zeros((memory_size, input_dims), dtype=np.float32)
        self.terminal_mem = np.zeros(memory_size, dtype=bool)

    def add_experience(self, state, action, reward, next_state, term):
        """
        function that adds new experience from episode to the memory. as long as the maximum memory size is not exceeded
        the experience gets added to the end of the memory after that the oldest experience gets replaced by new
        experience.
        """
        # while self.counter <= self.maximum_mem_size:
        #     index = self.counter
        #
        #     self.state_mem[index] = state
        #     self.next_state_mem[index] = next_state
        #     self.reward_mem[index] = reward
        #     self.action_mem[index] = action
        #     self.terminal_mem[index] = term
        #
        #     self.counter += 1
        # else:
        #     self.state_mem = np.append(self.state_mem[1:], state)
        #     self.next_state_mem = np.append(self.next_state_mem[1:], next_state)
        #     self.reward_mem = np.append(self.reward_mem[1:], reward)
        #     self.action_mem = np.append(self.action_mem[1:], action)
        #     self.terminal_mem = np.append(self.terminal_mem[1:], term)

        index = self.counter % self.maximum_mem_size

        self.state_mem[index] = state
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.next_state_mem[index] = next_state
        self.terminal_mem[index] = term

        self.counter += 1

    def get_experience(self, batch_size: int):
        """
        function that fetches random experience from the memory according to the batch size
        """
        random_index = np.random.choice(min(self.counter, self.maximum_mem_size), batch_size)

        rand_state = self.state_mem[random_index]
        rand_action = self.action_mem[random_index]
        rand_reward = self.reward_mem[random_index]
        rand_next_state = self.next_state_mem[random_index]
        rand_term = self.terminal_mem[random_index]

        return rand_state, rand_action, rand_reward, rand_next_state, rand_term

