import gym
import torch
from dqn.dqn import DQN
from dqn.actions import get_action, get_action_space
from dqn.environment_wrapper import EnvironmentWrapper


def dqn_inference(path):
    model = DQN(input_shape=1, num_of_actions=get_action_space())
    model.load_state_dict(torch.load(path))
    model.eval()

    env = gym.make('CarRacing-v0')
    env_wrapper = EnvironmentWrapper(env, 1)

    state = env_wrapper.reset()
    state = torch.tensor(state, dtype=torch.float32)
    done = False
    total_score = 0
    while not done:
        q_value = model(torch.stack([state]))
        _, action = get_action(q_value, train=False)
        print(action)
        state, reward, done = env_wrapper.step(action)
        state = torch.tensor(state, dtype=torch.float32)
        total_score += reward
        env_wrapper.render()
    return total_score
