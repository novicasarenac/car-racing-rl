import gym
import torch
from a2c.actor_critic import ActorCritic
from a2c.actions import get_action_space, get_actions
from a2c.environment_wrapper import EnvironmentWrapper


def a2c_inference(params, path):
    model = ActorCritic(params.stack_size, get_action_space())
    model.load_state_dict(torch.load(path))
    model.eval()

    env = gym.make('CarRacing-v0')
    env_wrapper = EnvironmentWrapper(env, params.stack_size)

    state = env_wrapper.reset()
    state = torch.Tensor([state])
    terminal = False
    total_score = 0
    while not terminal:
        probs, _, _ = model(state)
        action = get_actions(probs)
        print(action)
        state, reward, done = env_wrapper.step(action[0])
        state = torch.Tensor([state])
        total_score += reward
        env_wrapper.render()
    return total_score
