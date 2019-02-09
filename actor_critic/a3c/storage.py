import torch


class Storage:

    def __init__(self, steps_per_update):
        self.steps_per_update = steps_per_update
        self.reset_storage()

    def reset_storage(self):
        self.values = torch.zeros(self.steps_per_update, 1)
        self.rewards = torch.zeros(self.steps_per_update, 1)
        self.action_log_probs = torch.zeros(self.steps_per_update, 1)
        self.entropies = torch.zeros(self.steps_per_update)
        self.dones = torch.zeros(self.steps_per_update, 1)

    def add(self, step, value, reward, action_log_prob, entropy, done):
        self.values[step] = value
        self.rewards[step] = reward
        self.action_log_probs[step] = action_log_prob
        self.entropies[step] = entropy
        self.dones[step] = done
