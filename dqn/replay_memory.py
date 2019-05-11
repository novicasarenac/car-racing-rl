import random
from collections import deque


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return states, actions, rewards, next_states, dones


if __name__ == '__main__':
    memory = ReplayMemory(5)
    for i in range(5):
        memory.add(i, i, i, i, False)
    states, actions, rewards, next_states, dones = memory.sample(2)
    print(states)
    print(next_states)
    print(dones)
