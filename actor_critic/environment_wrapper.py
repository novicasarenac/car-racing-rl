import gym
import numpy as np
from PIL import Image
from collections import deque
from utils.image_utils import to_grayscale, zero_center, crop


class EnvironmentWrapper(gym.Wrapper):
    def __init__(self, env, stack_size):
        super().__init__(env)
        self.stack_size = stack_size
        self.frames = deque([], maxlen=stack_size)

    def reset(self):
        state = self.env.reset()
        for _ in range(self.stack_size):
            self.frames.append(self.preprocess(state))
        return self.state()

    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        self.env.env.viewer.window.dispatch_events()
        preprocessed_state = self.preprocess(state)

        self.frames.append(preprocessed_state)
        return self.state(), reward, done

    def state(self):
        return np.stack(self.frames, axis=0)

    def preprocess(self, state):
        preprocessed_state = to_grayscale(state)
        preprocessed_state = zero_center(preprocessed_state)
        preprocessed_state = crop(preprocessed_state)
        return preprocessed_state

    def get_state_shape(self):
        return (self.stack_size, 84, 84)


if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    env_wrapper = EnvironmentWrapper(env, 5)
    env_wrapper.reset()
    action = [0, 0, 0]
    for i in range(100):
        s, _, _ = env_wrapper.step(action)
        print(s.shape)
