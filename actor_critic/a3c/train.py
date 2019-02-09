import torch.multiprocessing as mp
from actor_critic.actor_critic import ActorCritic
from actor_critic.a2c.actions import get_action_space
from actor_critic.a3c.worker import Worker


class A3CTrainer:
    def __init__(self, params, model_path):
        self.params = params
        self.model_path = model_path
        self.num_of_processes = mp.cpu_count()
        self.global_model = ActorCritic(self.params.stack_size,
                                        get_action_space())
        self.global_model.share_memory()

    def run(self):
        processes = []
        for process_num in range(self.num_of_processes):
            worker = Worker(process_num, self.global_model, self.params)
            processes.append(worker)
            worker.start()

        for process in processes:
            process.join()
