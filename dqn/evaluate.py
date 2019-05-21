import gym
import torch
from dqn.dqn import DQN
from dqn.actions import get_action_space, get_action
from dqn.environment_wrapper import EnvironmentWrapper


def evaluate_dqn(path):
    model = DQN(input_shape=1, num_of_actions=get_action_space())
    model.load_state_dict(torch.load(path))
    model.eval()

    env = gym.make('CarRacing-v0')
    env_wrapper = EnvironmentWrapper(env, 1)

    total_reward = 0
    num_of_episodes = 100

    for episode in range(num_of_episodes):
        state = env_wrapper.reset()
        state = torch.tensor(state, dtype=torch.float)
        done = False
        score = 0
        while not done:
            q_value = model(torch.stack([state]))
            _, action = get_action(q_value, train=False)
            print(action)
            state, reward, done = env_wrapper.step(action)
            state = torch.tensor(state, dtype=torch.float32)
            score += reward
            env_wrapper.render()
        print('Episode: {0} Score: {1:.2f}'.format(episode, score))
        total_reward += score
    return total_reward / num_of_episodes
