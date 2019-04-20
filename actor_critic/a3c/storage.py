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

    def compute_expected_reward(self, last_value, discount_factor):
        expected_reward = torch.zeros(self.steps_per_update + 1, 1)
        expected_reward[-1] = last_value
        for step in reversed(range(self.rewards.size(0))):
            expected_reward[step] = self.rewards[step] + \
                                    expected_reward[step + 1] * discount_factor * (1.0 - self.dones[step])
        return expected_reward[:-1]

    def compute_gae(self, last_value, discount_factor, gae_coef):
        gae = torch.zeros(self.steps_per_update + 1, 1)
        next_value = last_value
        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + discount_factor * next_value - self.values[step]
            gae[step] = gae[step + 1] * discount_factor * gae_coef + delta
            next_value = self.values[step]
        return gae[:-1]
