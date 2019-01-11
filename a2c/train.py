import multiprocessing
import torch
from torch.optim import Adam
from a2c.utils import params
from a2c.parallel_environments import ParallelEnvironments
from a2c.actor_critic import ActorCritic
from a2c.actions import get_action_space, get_actions
from a2c.storage import Storage


class Trainer:
    def __init__(self):
        self.num_of_processes = multiprocessing.cpu_count()
        self.parallel_environments = ParallelEnvironments(number_of_processes=self.num_of_processes)
        self.actor_critic = ActorCritic(params.stack_size, get_action_space())
        self.optimizer = Adam(self.actor_critic.parameters(), lr=params.lr)
        self.storage = Storage(params.steps_per_update, self.num_of_processes)
        self.current_observations = torch.zeros(self.num_of_processes,
                                                *self.parallel_environments.get_state_shape())

    def run(self):
        # num of updates per environment
        num_of_updates = params.num_of_steps / params.steps_per_update
        self.current_observations = self.parallel_environments.reset()

        print(self.current_observations.size())

        for update in range(1): #range(int(num_of_updates)):
            probs, log_probs, value = self.actor_critic(self.current_observations)
            actions = get_actions(probs)
            states, rewards, dones = self.parallel_environments.step(actions)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()
