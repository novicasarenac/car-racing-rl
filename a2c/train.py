import multiprocessing
import torch
from torch.optim import Adam
from a2c.utils import params
from a2c.parallel_environments import ParallelEnvironments
from a2c.actor_critic import ActorCritic
from a2c.actions import get_action_space
from a2c.storage import Storage


class Trainer:
    def __init__(self):
        self.num_of_processes = multiprocessing.cpu_count()
        self.parallel_environments = ParallelEnvironments(number_of_processes=num_of_processes)
        self.actor_critic = ActorCritic(params.stack_size, get_action_space())
        self.optimizer = Adam(self.actor_critic.parameters(), lr=params.lr)
        self.storage = Storage(params.steps_per_update, self.num_of_processes)
        self.current_observations = torch.zeros(self.num_of_processes,
                                                *self.parallel_environments.get_state_shape())

    def run(self):
        # num of updates per environment
        num_of_updates = params.num_of_steps / params.steps_per_update
        self.current_observations = self.parallel_environments.reset()

        for update in num_of_updates:
            
