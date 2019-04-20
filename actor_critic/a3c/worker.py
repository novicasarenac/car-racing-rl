import torch
import gym
import torch.multiprocessing as mp
import torch.nn as nn
from torch.optim import Adam
from actor_critic.environment_wrapper import EnvironmentWrapper
from actor_critic.actor_critic import ActorCritic
from actor_critic.actions import get_action_space, get_actions
from actor_critic.a3c.storage import Storage


class Worker(mp.Process):
    def __init__(self, process_num, global_model, params):
        super().__init__()

        self.process_num = process_num
        self.global_model = global_model
        self.params = params
        env = gym.make('CarRacing-v0')
        self.environment = EnvironmentWrapper(env, self.params.stack_size)
        self.model = ActorCritic(self.params.stack_size, get_action_space())
        self.optimizer = Adam(self.global_model.parameters(), lr=self.params.lr)
        self.storage = Storage(self.params.steps_per_update)
        self.current_observation = torch.zeros(1, *self.environment.get_state_shape())

    def run(self):
        num_of_updates = self.params.num_of_steps / self.params.steps_per_update
        self.current_observation = torch.Tensor([self.environment.reset()])

        for update in range(int(num_of_updates)):
            self.storage.reset_storage()
            # synchronize with global model
            self.model.load_state_dict(self.global_model.state_dict())
            for step in range(self.params.steps_per_update):
                probs, log_probs, value = self.model(self.current_observation)
                action = get_actions(probs)[0]
                action_log_prob, entropy = self.compute_action_log_and_entropy(probs, log_probs)

                state, reward, done = self.environment.step(action)
                if done:
                    state = self.environment.reset()
                done = torch.Tensor([done])
                self.current_observation = torch.Tensor([state])
                self.storage.add(step, value, reward, action_log_prob, entropy, done)

            _, _, last_value = self.model(self.current_observation)
            expected_reward = self.storage.compute_expected_reward(last_value,
                                                                   self.params.discount_factor)
            advantages = torch.tensor(expected_reward) - self.storage.values
            value_loss = advantages.pow(2).mean()
            if self.params.use_gae:
                gae = self.storage.compute_gae(last_value,
                                               self.params.discount_factor,
                                               self.params.gae_coef)
                policy_loss = -(torch.tensor(gae) * self.storage.action_log_probs).mean()
            else:
                policy_loss = -(advantages * self.storage.action_log_probs).mean()

            self.optimizer.zero_grad()
            loss = policy_loss - self.params.entropy_coef * self.storage.entropies.mean() + \
                self.params.value_loss_coef * value_loss
            loss.backward()
            nn.utils.clip_grad_norm(self.model.parameters(), self.params.max_norm)
            self._share_gradients()
            self.optimizer.step()

            if update % 20 == 0:
                print('Process: {}. Update: {}. Loss: {}'.format(self.process_num,
                                                                 update,
                                                                 loss))

    def compute_action_log_and_entropy(self, probs, log_probs):
        values, indices = probs.max(1)
        indices = indices.view(-1, 1)
        action_log_prob = log_probs.gather(1, indices)

        entropy = -(log_probs * probs).sum(-1)

        return action_log_prob, entropy

    def _share_gradients(self):
        for local_param, global_param in zip(self.model.parameters(),
                                             self.global_model.parameters()):
            global_param._grad = local_param.grad
