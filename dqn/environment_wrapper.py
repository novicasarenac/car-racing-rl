import gym
from utils.image_utils import to_grayscale, crop, save


class EnvironmentWrapper(gym.Wrapper):
    def __init__(self, env, skip_steps):
        super().__init__(env)
        self.skip_steps = skip_steps

    def reset(self):
        state = self.env.reset()
        preprocessed_state = self.preprocess(state)
        return [preprocessed_state]

    def step(self, action):
        total_reward = 0
        for i in range(self.skip_steps):
            state, reward, done, _ = self.env.step(action)
            self.env.env.viewer.window.dispatch_events()
            total_reward += reward
            if done:
                break
        preprocessed_state = self.preprocess(state)
        return [preprocessed_state], total_reward, done

    def preprocess(self, state):
        preprocessed_state = to_grayscale(state)
        preprocessed_state = crop(preprocessed_state)
        return preprocessed_state


if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    env_wrapper = EnvironmentWrapper(env, 4)
    s = env_wrapper.reset()
    print(s.shape)
    action = [0, 1, 0]
    for i in range(10):
        s, r, _ = env_wrapper.step(action)
        print(s.shape)
        print(r)
