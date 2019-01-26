import multiprocessing
import torch
import torch.nn as nn
from torch.optim import Adam
from a2c.parallel_environments import ParallelEnvironments
from a2c.actor_critic import ActorCritic
from a2c.actions import get_action_space, get_actions
from a2c.storage import Storage


class A2CTrainer:
    def __init__(self, params, model_path):
        self.params = params
        self.model_path = model_path
        self.num_of_processes = multiprocessing.cpu_count()
        self.parallel_environments = ParallelEnvironments(self.params.stack_size,
                                                          number_of_processes=self.num_of_processes)
        self.actor_critic = ActorCritic(self.params.stack_size, get_action_space())
        self.optimizer = Adam(self.actor_critic.parameters(), lr=self.params.lr)
        self.storage = Storage(self.params.steps_per_update, self.num_of_processes)
        self.current_observations = torch.zeros(self.num_of_processes,
                                                *self.parallel_environments.get_state_shape())

    def run(self):
        # num of updates per environment
        num_of_updates = self.params.num_of_steps / self.params.steps_per_update
        self.current_observations = self.parallel_environments.reset()

        print(self.current_observations.size())

        for update in range(int(num_of_updates)):
            self.storage.reset_storage()
            for step in range(self.params.steps_per_update):
                probs, log_probs, value = self.actor_critic(self.current_observations)
                actions = get_actions(probs)
                action_log_probs, entropies = self.compute_action_logs_and_entropies(probs, log_probs)

                states, rewards, dones = self.parallel_environments.step(actions)
                rewards = rewards.view(-1, 1)
                dones = dones.view(-1, 1)
                self.current_observations = states
                self.storage.add(step, value, rewards, action_log_probs, entropies, dones)

            _, _, last_values = self.actor_critic(self.current_observations)
            expected_rewards = self.storage.compute_expected_rewards(last_values,
                                                                     self.params.discount_factor)
            advantages = torch.tensor(expected_rewards) - self.storage.values
            value_loss = advantages.pow(2).mean()
            policy_loss = -(advantages * self.storage.action_log_probs).mean()

            self.optimizer.zero_grad()
            loss = policy_loss - self.params.entropy_coef * self.storage.entropies.mean() + \
                self.params.value_loss_coef * value_loss
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm(self.actor_critic.parameters(), self.params.max_norm)
            self.optimizer.step()

            if update % 300 == 0:
                torch.save(self.actor_critic.state_dict(), self.model_path)

            if update % 100 == 0:
                print('Update: {}. Loss: {}'.format(update, loss))

    def compute_action_logs_and_entropies(self, probs, log_probs):
        values, indices = probs.max(1)
        indices = indices.view(-1, 1)
        action_log_probs = log_probs.gather(1, indices)

        entropies = -(log_probs * probs).sum(-1)

        return action_log_probs, entropies
