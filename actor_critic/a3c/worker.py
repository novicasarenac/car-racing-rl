import gym
import torch.multiprocessing as mp
from torch.optim import Adam
from actor_critic.environment_wrapper import EnvironmentWrapper
from actor_critic.actor_critic import ActorCritic
from actor_critic.a2c.actions import get_action_space
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

    def run(self):
        num_of_updates = self.params.num_of_steps / self.params.steps_per_update

        for update in range(int(num_of_updates)):
            print(update)
