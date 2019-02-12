import torch
import gym
import torch.multiprocessing as mp
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

                action_log_probs, entropy = self.compute_action_logs_and_entropies(probs, log_probs)
                break
            break

    def compute_action_logs_and_entropies(self, probs, log_probs):
        # TODO
        return None
