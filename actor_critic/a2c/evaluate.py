import gym
import torch
from actor_critic.actor_critic import ActorCritic
from actor_critic.actions import get_action_space, get_actions
from actor_critic.environment_wrapper import EnvironmentWrapper


def evaluate_a2c(params, path):
    model = ActorCritic(params.stack_size, get_action_space())
    model.load_state_dict(torch.load(path))
    model.eval()

    env = gym.make('CarRacing-v0')
    env_wrapper = EnvironmentWrapper(env, params.stack_size)

    total_reward = 0
    num_of_episodes = 100

    for episode in range(num_of_episodes):
        state = env_wrapper.reset()
        state = torch.Tensor([state])
        done = False
        score = 0
        while not done:
            probs, _, _ = model(state)
            action = get_actions(probs)
            state, reward, done = env_wrapper.step(action[0])
            state = torch.Tensor([state])
            score += reward
            env_wrapper.render()
        print('Episode: {0} Score: {1:.2f}'.format(episode, score))
        total_reward += score
    return total_reward / num_of_episodes
