import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_shape, num_of_actions):
        super().__init__()

        self.conv1 = nn.Conv2d(input_shape, 16, kernel_size=5, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(32)

        self.linear1 = nn.Linear(32*7*7, 256)
        self.linear2 = nn.Linear(256, num_of_actions)

    def forward(self, x):
        conv1_out = F.relu(self.batch_norm1(self.conv1(x)))
        conv2_out = F.relu(self.batch_norm2(self.conv2(conv1_out)))
        conv3_out = F.relu(self.batch_norm3(self.conv3(conv2_out)))

        flattened = torch.flatten(conv3_out, start_dim=1)
        linear1_out = self.linear1(flattened)
        q_value = self.linear2(linear1_out)

        return q_value


if __name__ == '__main__':
    x = torch.rand(1, 1, 84, 84)
    dqn = DQN(input_shape=1, num_of_actions=4)
    a = dqn(x)
    m = a.max(1)[1]
    print(a)
    print(m)
